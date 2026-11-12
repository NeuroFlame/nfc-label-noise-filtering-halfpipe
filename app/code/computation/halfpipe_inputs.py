"""Convert nested HALFpipe derivatives into the flat per-subject TSV layout.

Sites point NeuroFLAME at a folder holding a participants file, an optional
QC file, and HALFpipe derivatives at any depth (``sub-*/[ses-*/]func/task-*/``).
At the start of the run this module averages each subject's included runs into
one correlation matrix and writes them, with a ``labels.csv``, into a staging
folder in the flat layout that ``validation.validate_and_get_inputs`` already
reads. The rest of the pipeline is unchanged.
"""

import copy
import os
import re
import shutil
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import constants as ct

# Feature/atlas names used for the staged flat files. The flat reader appends
# "CombinedDseg" to the atlas when it builds each filename.
STAGED_FEATURE = "corrMatrix"
STAGED_ATLAS = "halfpipe"
STAGED_LABEL_FILE = "labels.csv"
STAGED_DIR_NAME = "flat_halfpipe_inputs"
QC_SUMMARY_FILE = "qc_summary.csv"

PARTICIPANT_FILE_CANDIDATES = ("participants.csv", "participants.tsv")
PARTICIPANT_ID_COLUMNS = (ct.PARAM_KEY_SUBJECT_ID, "participant_id")

MATRIX_SUFFIX = "_desc-correlation_matrix.tsv"
# BIDS-style name: sub-<label>(_<key>-<value>)*_desc-correlation_matrix.tsv,
# with alphanumeric labels only, so stray copies ("... copy.tsv") never match.
_MATRIX_NAME = re.compile(
    r"^sub-[A-Za-z0-9]+(?:_[A-Za-z]+-[A-Za-z0-9]+)*" + re.escape(MATRIX_SUFFIX) + r"$"
)
_FISHER_CLIP = 0.999999
_MATRIX_TOLERANCE = 1e-6
# Filename entities that are not part of a run's identity: who/which task, and
# which matrix. Every other entity (dir-AP/PA, acq-, echo-, ...) distinguishes runs.
_NON_RUN_ENTITIES = {"sub", "ses", "task", "run", "feature", "atlas", "desc"}
# Entity columns qc.csv may carry next to session/run (other columns, such as
# reviewer notes, are ignored).
QC_ENTITY_COLUMNS = ("dir", "acq", "echo", "rec", "ce", "part", "inv", "mt", "flip")
_SKIPPED_DIRS = {"__MACOSX"}
_MISSING_DIAGNOSIS = {"", "n/a", "na", "nan", "none", "null"}
_INCLUDE_VALUES = {
    "yes": True,
    "y": True,
    "true": True,
    "1": True,
    "no": False,
    "n": False,
    "false": False,
    "0": False,
}


class HalfpipeInputError(ValueError):
    """A site's nested HALFpipe inputs cannot be converted as configured."""


@dataclass(frozen=True)
class RunKey:
    """One subject/session/run, normalized for matching.

    ``extras`` holds any other filename entities (for example ``dir-AP`` versus
    ``dir-PA``) as sorted ``(entity, value)`` pairs, so runs that share a run
    number but differ in phase-encoding direction or echo stay separate.
    """

    subject: str
    session: str
    run: Optional[int]
    extras: Tuple[Tuple[str, str], ...] = ()


@dataclass
class MatrixFile:
    """One discovered correlation matrix and its parsed BIDS entities."""

    path: str
    key: RunKey
    entities: Dict[str, str]


@dataclass
class SubjectSummary:
    """What went into one subject's averaged matrix (written to qc_summary.csv)."""

    subject_id: str
    status: str
    sessions: List[str] = field(default_factory=list)
    runs_used: List[str] = field(default_factory=list)
    runs_excluded: List[str] = field(default_factory=list)
    matrix_size: str = ""


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def normalize_subject(value) -> str:
    """Lower-case a subject ID and drop an optional ``sub-`` prefix."""
    text = str(value).strip()
    if text.lower().startswith("sub-"):
        text = text[4:]
    return text.lower()


def normalize_session(value) -> str:
    """Lower-case a session label and drop an optional ``ses-`` prefix; blank -> ''."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.lower().startswith("ses-"):
        text = text[4:]
    return text.lower()


def normalize_run(value) -> Optional[int]:
    """Parse ``01``, ``1`` or ``run-01`` as the integer run number; blank -> None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if text.lower().startswith("run-"):
        text = text[4:]
    if text == "":
        return None
    if not text.isdigit():
        raise HalfpipeInputError(
            f"Invalid run value {value!r}; expected a run number such as 1, 01 or run-01"
        )
    return int(text)


def normalize_entity(name, value) -> str:
    """Lower-case another entity's value and drop an optional ``<name>-`` prefix."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    if text.startswith(f"{name}-"):
        text = text[len(name) + 1 :]
    return text


def _run_extras(entities) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, normalize_entity(name, value))
            for name, value in entities.items()
            if name not in _NON_RUN_ENTITIES
        )
    )


def _label(item: "MatrixFile") -> str:
    """``sub-X[_ses-Y][_run-Z][_dir-AP...]`` spelled as in the source filename."""
    keys = ["sub", "ses", "run"] + [name for name, _ in item.key.extras]
    return "_".join(f"{k}-{item.entities[k]}" for k in keys if k in item.entities)


def _describe(key: RunKey) -> str:
    parts = [f"sub-{key.subject}"]
    if key.session:
        parts.append(f"ses-{key.session}")
    if key.run is not None:
        parts.append(f"run-{key.run}")
    parts += [f"{name}-{value}" for name, value in key.extras]
    return "_".join(parts)


# --------------------------------------------------------------------------
# Detection and discovery
# --------------------------------------------------------------------------


def get_halfpipe_settings(parameters) -> Tuple[str, str, Optional[str]]:
    """Return ``(atlas, task, feature)`` from ``ReadDataFileFromHalfpipeOutputs``."""
    settings = parameters.get("ReadDataFileFromHalfpipeOutputs") or {}
    atlas = str(settings.get("atlas", "")).strip()
    if not atlas:
        raise HalfpipeInputError(
            "ReadDataFileFromHalfpipeOutputs.atlas must name the HALFpipe atlas to use (for example NeuroMark1)"
        )
    task = str(settings.get("task", "rest")).strip() or "rest"
    feature = settings.get("feature", settings.get("corr"))
    feature = str(feature).strip() if feature else None
    return atlas, task, feature


def _atlas_matches(file_atlas: str, atlas: str) -> bool:
    file_atlas, atlas = file_atlas.lower(), atlas.lower()
    return file_atlas in (atlas, atlas + "combineddseg")


def parse_matrix_filename(filename: str) -> Optional[Dict[str, str]]:
    """Return the BIDS entities of a HALFpipe correlation-matrix filename, or None."""
    if not _MATRIX_NAME.match(filename):
        return None
    stem = filename[: -len(MATRIX_SUFFIX)]
    entities = {}
    for part in stem.split("_"):
        key, value = part.split("-", 1)
        entities[key.lower()] = value
    return entities


def discover_matrices(
    data_dir, atlas, task, feature=None, *, nested_only=False
) -> List[MatrixFile]:
    """Find every correlation matrix for the configured atlas/task (and feature, if set).

    Follows symlinked folders (derivatives are often linked in from scratch
    space) but visits each real folder once, and skips hidden folders and
    ``__MACOSX`` copies left by unzipping on a Mac.
    """
    found = []
    visited = set()
    for root, dirs, files in os.walk(data_dir, followlinks=True):
        real_root = os.path.realpath(root)
        if real_root in visited:
            dirs[:] = []
            continue
        visited.add(real_root)
        dirs[:] = sorted(
            d for d in dirs if not d.startswith(".") and d not in _SKIPPED_DIRS
        )
        if nested_only and os.path.samefile(root, data_dir):
            continue
        for name in sorted(files):
            entities = parse_matrix_filename(name)
            if entities is None:
                continue
            if not _atlas_matches(entities.get("atlas", ""), atlas):
                continue
            if entities.get("task", "").lower() != task.lower():
                continue
            if feature and entities.get("feature", "").lower() != feature.lower():
                continue
            key = RunKey(
                subject=normalize_subject(entities["sub"]),
                session=normalize_session(entities.get("ses")),
                run=normalize_run(entities.get("run")),
                extras=_run_extras(entities),
            )
            found.append(
                MatrixFile(path=os.path.join(root, name), key=key, entities=entities)
            )
    return found


def explain_unmatched_derivatives(data_dir, parameters) -> Optional[str]:
    """Describe nested HALFpipe matrices that the configured atlas/task do not match.

    Returns None when there are none (so the site's inputs are flat or absent).
    Called when ``is_nested_halfpipe_dataset`` is False, to turn an atlas or task
    typo into a clear message instead of a flat-reader failure.
    """
    if not parameters.get("ReadDataFileFromHalfpipeOutputs"):
        return None
    atlas, task, feature = get_halfpipe_settings(parameters)
    found = {}
    for root, dirs, files in os.walk(data_dir, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SKIPPED_DIRS]
        if os.path.samefile(root, data_dir):
            continue
        for name in files:
            entities = parse_matrix_filename(name)
            if entities:
                combo = (
                    entities.get("task", "?"),
                    entities.get("atlas", "?"),
                    entities.get("feature", "?"),
                )
                found[combo] = found.get(combo, 0) + 1
    if not found:
        return None
    available = "; ".join(
        f"task-{t} atlas-{a} feature-{f} ({n} files)"
        for (t, a, f), n in sorted(found.items())
    )
    wanted = f"task-{task} atlas-{atlas}" + (f" feature-{feature}" if feature else "")
    return (
        f"Found HALFpipe correlation matrices, but none for {wanted}. Available: {available}. "
        "Check ReadDataFileFromHalfpipeOutputs (atlas, task, feature)."
    )


def is_nested_halfpipe_dataset(data_dir, parameters) -> bool:
    """True when matching correlation matrices exist below the site folder's top level."""
    if not parameters.get("ReadDataFileFromHalfpipeOutputs"):
        return False
    atlas, task, feature = get_halfpipe_settings(parameters)
    return bool(discover_matrices(data_dir, atlas, task, feature, nested_only=True))


# --------------------------------------------------------------------------
# Participants and QC files
# --------------------------------------------------------------------------


def _read_table(path) -> pd.DataFrame:
    sep = "\t" if path.lower().endswith(".tsv") else ","
    return pd.read_csv(
        path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )


def find_participants_file(data_dir, parameters) -> str:
    """Use ``LabelFile`` if it exists in the site folder, else participants.csv/.tsv."""
    candidates = [parameters.get("LabelFile")] + list(PARTICIPANT_FILE_CANDIDATES)
    for name in candidates:
        if name and os.path.isfile(os.path.join(data_dir, name)):
            return os.path.join(data_dir, name)
    raise HalfpipeInputError(
        "No participants file found: expected LabelFile, participants.csv or participants.tsv in the site folder"
    )


def load_participants(path) -> pd.DataFrame:
    """Return columns ``subject_id`` (as given), ``key`` (normalized) and ``diagnosis``."""
    table = _read_table(path)
    table.columns = [c.strip() for c in table.columns]
    id_column = next((c for c in PARTICIPANT_ID_COLUMNS if c in table.columns), None)
    if id_column is None or ct.PARAM_KEY_DIAGNOSIS not in table.columns:
        raise HalfpipeInputError(
            f"{os.path.basename(path)} needs a 'subject_id' (or 'participant_id') column and a 'diagnosis' column; "
            f"found {list(table.columns)}"
        )
    participants = pd.DataFrame(
        {
            "subject_id": table[id_column].str.strip(),
            "key": table[id_column].map(normalize_subject),
            "diagnosis": table[ct.PARAM_KEY_DIAGNOSIS].str.strip(),
        }
    )
    participants = participants[participants["subject_id"] != ""]
    duplicates = participants[participants["key"].duplicated(keep=False)][
        "subject_id"
    ].tolist()
    if duplicates:
        raise HalfpipeInputError(
            f"Duplicate subjects in {os.path.basename(path)}: {sorted(set(duplicates))}"
        )
    return participants.reset_index(drop=True)


def load_qc(path) -> Dict[RunKey, bool]:
    """Read qc.csv into ``{RunKey: include}``.

    Required columns: ``subject_id`` and ``include``; ``session`` and ``run`` are
    optional. Entity columns from ``QC_ENTITY_COLUMNS`` (``dir``, ``echo``, ...)
    tell apart runs that share a run number; other columns are ignored.
    """
    table = _read_table(path)
    table.columns = [c.strip().lower() for c in table.columns]
    missing = {"subject_id", "include"} - set(table.columns)
    if missing:
        raise HalfpipeInputError(
            f"{os.path.basename(path)} is missing required column(s): {sorted(missing)}"
        )
    entity_columns = [c for c in QC_ENTITY_COLUMNS if c in table.columns]

    decisions: Dict[RunKey, bool] = {}
    errors = []
    for line_number, row in enumerate(table.to_dict("records"), start=2):
        if not str(row.get("subject_id", "")).strip():
            continue
        try:
            key = RunKey(
                subject=normalize_subject(row["subject_id"]),
                session=normalize_session(row.get("session", "")),
                run=normalize_run(row.get("run", "")),
                extras=tuple(
                    sorted(
                        (c, normalize_entity(c, row[c]))
                        for c in entity_columns
                        if normalize_entity(c, row[c])
                    )
                ),
            )
        except HalfpipeInputError as error:
            errors.append(f"line {line_number}: {error}")
            continue
        include = _INCLUDE_VALUES.get(str(row["include"]).strip().lower())
        if include is None:
            errors.append(
                f"line {line_number} ({_describe(key)}): include must be yes or no, got {row['include']!r}"
            )
        elif key in decisions:
            errors.append(f"line {line_number}: duplicate row for {_describe(key)}")
        else:
            decisions[key] = include
    if errors:
        raise HalfpipeInputError(
            f"{os.path.basename(path)} has problems:\n  " + "\n  ".join(errors)
        )
    return decisions


# --------------------------------------------------------------------------
# Averaging
# --------------------------------------------------------------------------


def load_matrix(path) -> np.ndarray:
    """Read one correlation matrix TSV and check it is one.

    It must be square (no header row), symmetric and within [-1, 1]; NaN marks
    ROIs without data and is allowed.
    """
    if os.path.getsize(path) == 0:
        raise HalfpipeInputError(f"{path} is empty")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = np.genfromtxt(path, delimiter="\t")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise HalfpipeInputError(
            f"{path} is not a square matrix (shape {matrix.shape}); "
            "correlation matrices must have no header row or index column"
        )
    finite = np.isfinite(matrix) & np.isfinite(matrix.T)
    if np.any(np.abs(matrix - matrix.T)[finite] > _MATRIX_TOLERANCE):
        raise HalfpipeInputError(
            f"{path} is not symmetric, so it is not a correlation matrix"
        )
    values = matrix[np.isfinite(matrix)]
    if values.size and np.max(np.abs(values)) > 1 + _MATRIX_TOLERANCE:
        raise HalfpipeInputError(
            f"{path} has values outside [-1, 1] (largest {np.max(np.abs(values)):.3f}), "
            "so it is not a correlation matrix"
        )
    return matrix


def fisher_z_average(matrices: List[np.ndarray]) -> np.ndarray:
    """Average correlation matrices in Fisher-z space, ignoring NaNs per ROI pair.

    A single matrix is returned unchanged. A pair that is NaN in every run stays NaN.
    """
    if len(matrices) == 1:
        return matrices[0].copy()
    stacked = np.arctanh(np.clip(np.stack(matrices), -_FISHER_CLIP, _FISHER_CLIP))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN pairs
        averaged = np.tanh(np.nanmean(stacked, axis=0))
    np.fill_diagonal(averaged, 1.0)
    return averaged


def _write_matrix(path, matrix):
    rows = [
        "\t".join("nan" if np.isnan(x) else f"{x:.10f}" for x in row) for row in matrix
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows))


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------


def stage_flat_dataset(data_dir, parameters, staging_dir, logger, summary_dir=None):
    """Convert one site's nested HALFpipe inputs into a flat folder.

    Returns ``(staged_dir, staged_parameters)`` for ``validate_and_get_inputs``.
    """
    atlas, task, feature = get_halfpipe_settings(parameters)
    ignore_missing = str(
        parameters.get(
            "IgnoreSubjectsWithMissingData", ct.DEFAULT_IgnoreSubjectsWithMissingData
        )
    ).strip().lower() in ("true", "1", "yes")

    participants_path = find_participants_file(data_dir, parameters)
    participants = _code_diagnoses(
        load_participants(participants_path),
        parameters,
        ignore_missing,
        os.path.basename(participants_path),
        logger,
    )
    logger.info(
        f"Reading nested HALFpipe derivatives (atlas={atlas}, task={task}"
        f"{', feature=' + feature if feature else ''}) for {len(participants)} participants "
        f"from {os.path.basename(participants_path)}"
    )

    matrices = discover_matrices(data_dir, atlas, task, feature)
    by_key: Dict[RunKey, List[MatrixFile]] = {}
    for item in matrices:
        by_key.setdefault(item.key, []).append(item)

    ambiguous = {k: v for k, v in by_key.items() if len(v) > 1}
    if ambiguous:
        details = [
            f"{_describe(k)}: " + ", ".join(sorted(os.path.basename(m.path) for m in v))
            for k, v in sorted(ambiguous.items(), key=lambda kv: _describe(kv[0]))
        ]
        raise HalfpipeInputError(
            f"More than one correlation matrix for atlas {atlas} per run. Set "
            "ReadDataFileFromHalfpipeOutputs.feature to choose one:\n  "
            + "\n  ".join(details)
        )

    labeled = set(participants["key"])
    runs_by_subject: Dict[str, List[RunKey]] = {}
    for key in sorted(by_key, key=_describe):
        runs_by_subject.setdefault(key.subject, []).append(key)
    unlabeled = sorted(set(runs_by_subject) - labeled)
    if unlabeled:
        logger.info(
            f"Ignoring derivatives for {len(unlabeled)} subject(s) not in the participants file: "
            f"{', '.join('sub-' + s for s in unlabeled)}"
        )

    qc_path = os.path.join(data_dir, parameters.get("QCFile", "qc.csv"))
    if os.path.isfile(qc_path):
        qc = load_qc(qc_path)
        _check_qc(qc, by_key, labeled, os.path.basename(qc_path), logger)
        logger.info(f"Applying QC decisions from {os.path.basename(qc_path)}")
    else:
        qc = None
        logger.warning(
            f"No QC file ({os.path.basename(qc_path)}) found: every run found for each "
            "participant is included"
        )

    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)

    summaries, staged_rows, missing, matrix_shape = [], [], [], None
    empty_rois = {}  # staged subject -> ROIs with no data in any included run
    for participant in participants.to_dict("records"):
        subject_key = participant["key"]
        staged_id = "sub-" + re.sub(r"(?i)^sub-", "", participant["subject_id"])
        keys = runs_by_subject.get(subject_key, [])
        used = [k for k in keys if qc is None or qc.get(k)]
        excluded = [k for k in keys if k not in used]
        summary = SubjectSummary(
            subject_id=participant["subject_id"],
            status="",
            sessions=sorted({by_key[k][0].entities["ses"] for k in used if k.session}),
            runs_used=[_label(by_key[k][0]) for k in used],
            runs_excluded=[_label(by_key[k][0]) for k in excluded],
        )
        summaries.append(summary)
        if not used:
            summary.status = "no included runs" if keys else "no matching files"
            missing.append(participant["subject_id"])
            continue

        loaded = [load_matrix(by_key[k][0].path) for k in used]
        shapes = {m.shape for m in loaded}
        if len(shapes) != 1:
            raise HalfpipeInputError(
                f"Runs for {participant['subject_id']} have different matrix sizes: {sorted(shapes)}"
            )
        shape = loaded[0].shape
        if matrix_shape is None:
            matrix_shape = shape
        elif shape != matrix_shape:
            raise HalfpipeInputError(
                f"{participant['subject_id']} has a {shape[0]}x{shape[1]} matrix but other subjects at this site "
                f"have {matrix_shape[0]}x{matrix_shape[1]}; all subjects must use the same atlas"
            )
        averaged = fisher_z_average(loaded)
        nan_pairs = np.isnan(averaged) & ~np.eye(shape[0], dtype=bool)
        if nan_pairs.any():
            # ROIs with no data at all; if gaps are only scattered pairs, the ROIs involved
            empty = np.where(nan_pairs.sum(axis=1) == shape[0] - 1)[0]
            rois = empty if empty.size else np.where(nan_pairs.any(axis=1))[0]
            empty_rois[participant["subject_id"]] = frozenset(rois.tolist())
        _write_matrix(
            os.path.join(staging_dir, _staged_matrix_name(staged_id)), averaged
        )
        staged_rows.append((staged_id, participant["code"]))
        summary.status = "included"
        summary.matrix_size = f"{shape[0]}x{shape[1]}"
        if len(summary.sessions) > 1:
            logger.info(
                f"{participant['subject_id']}: averaged {len(used)} runs across sessions "
                f"{', '.join('ses-' + s for s in summary.sessions)}"
            )

    if missing:
        message = (
            f"{len(missing)} participant(s) have no usable correlation matrices for atlas {atlas}: "
            f"{', '.join(missing)}"
        )
        if not ignore_missing:
            _write_summary(summaries, summary_dir)
            raise HalfpipeInputError(
                message + ". Set IgnoreSubjectsWithMissingData to true to skip them."
            )
        logger.info(message + " (skipped: IgnoreSubjectsWithMissingData is true)")
    if not staged_rows:
        _write_summary(summaries, summary_dir)
        raise HalfpipeInputError(
            f"No participants have usable correlation matrices for atlas {atlas}"
        )
    _check_roi_coverage(
        empty_rois,
        len(staged_rows),
        atlas,
        ignore_missing,
        logger,
        summaries,
        summary_dir,
    )

    pd.DataFrame(
        staged_rows, columns=[ct.PARAM_KEY_SUBJECT_ID, ct.PARAM_KEY_DIAGNOSIS]
    ).to_csv(os.path.join(staging_dir, STAGED_LABEL_FILE), index=False)
    _write_summary(summaries, summary_dir)
    logger.info(
        f"Converted {len(staged_rows)} participant(s) to the flat layout "
        f"({matrix_shape[0]}x{matrix_shape[1]} matrices)"
    )

    # Staged labels are already the internal codes, so the flat reader maps them
    # to themselves (and never has to parse text labels).
    staged_parameters = copy.deepcopy(parameters)
    staged_parameters[ct.PARAM_KEY_PATIENT_LABEL] = ct.PATIENT_LABEL_VALUE
    staged_parameters[ct.PARAM_KEY_CONTROL_LABEL] = ct.CONTROL_LABEL_VALUE
    staged_parameters["LabelFile"] = STAGED_LABEL_FILE
    staged_parameters.pop("ExistingDataFile", None)
    staged_parameters["ReadDataFileFromHalfpipeOutputs"] = {
        "atlas": STAGED_ATLAS,
        "corr": STAGED_FEATURE,
    }
    return staging_dir, staged_parameters


def _staged_matrix_name(staged_id) -> str:
    return (
        f"{staged_id}_task-rest_feature-{STAGED_FEATURE}_atlas-{STAGED_ATLAS}CombinedDseg"
        f"{MATRIX_SUFFIX}"
    )


def _check_qc(qc, by_key, labeled, qc_name, logger):
    """Every labeled run on disk needs a QC decision; every included QC row needs a file."""
    unreviewed = sorted(
        _label(v[0]) for k, v in by_key.items() if k.subject in labeled and k not in qc
    )
    no_file = sorted(
        _describe(k) for k, include in qc.items() if include and k not in by_key
    )
    problems = []
    if unreviewed:
        message = (
            f"Runs found on disk with no row in {qc_name} (add them with include yes or no):\n    "
            + "\n    ".join(unreviewed)
        )
        qc_columns = {name for key in qc for name, _ in key.extras}
        needed = sorted(
            {
                name
                for key in by_key
                if key.subject in labeled and key not in qc
                for name, _ in key.extras
                if name not in qc_columns
            }
        )
        if needed:
            message += (
                f"\n  These runs are told apart by {', '.join(needed)} in their filenames; "
                f"add {'a column' if len(needed) == 1 else 'columns'} named "
                f"{', '.join(needed)} to {qc_name}."
            )
        problems.append(message)
    if no_file:
        problems.append(
            f"Rows in {qc_name} marked include=yes with no matching correlation matrix:\n    "
            + "\n    ".join(no_file)
        )
    if problems:
        raise HalfpipeInputError("\n  ".join(problems))
    not_labeled = sorted({k.subject for k in qc} - labeled)
    if not_labeled:
        logger.warning(
            f"{qc_name} lists subject(s) not in the participants file: "
            f"{', '.join('sub-' + s for s in not_labeled)}"
        )


def _code_diagnoses(participants, parameters, ignore_missing, file_name, logger):
    """Map each diagnosis to the internal patient/control code (column ``code``).

    Matching ignores case and surrounding spaces, and treats numbers by value
    (``1``, ``1.0``). Blank or n/a diagnoses are missing; any other value is an
    error, since it would silently be neither group.
    """
    patient = parameters.get(ct.PARAM_KEY_PATIENT_LABEL)
    control = parameters.get(ct.PARAM_KEY_CONTROL_LABEL)

    def canonical(value):
        text = str(value).strip().lower()
        try:
            return repr(float(text))
        except ValueError:
            return text

    codes = {
        canonical(patient): ct.PATIENT_LABEL_VALUE,
        canonical(control): ct.CONTROL_LABEL_VALUE,
    }
    coded, missing, unknown = [], [], {}
    for row in participants.to_dict("records"):
        value = row["diagnosis"]
        if str(value).strip().lower() in _MISSING_DIAGNOSIS:
            missing.append(row["subject_id"])
        elif canonical(value) in codes:
            coded.append({**row, "code": codes[canonical(value)]})
        else:
            unknown.setdefault(str(value).strip(), []).append(row["subject_id"])
    if unknown:
        details = "; ".join(
            f"{value!r} ({len(ids)}: {', '.join(ids[:5])}{', ...' if len(ids) > 5 else ''})"
            for value, ids in sorted(unknown.items())
        )
        raise HalfpipeInputError(
            f"{file_name} has diagnosis values that are neither PatientLabel {patient!r} nor "
            f"ControlLabel {control!r}: {details}"
        )
    if missing:
        message = f"{len(missing)} participant(s) in {file_name} have no diagnosis: {', '.join(missing)}"
        if not ignore_missing:
            raise HalfpipeInputError(
                message + ". Set IgnoreSubjectsWithMissingData to true to skip them."
            )
        logger.info(message + " (skipped: IgnoreSubjectsWithMissingData is true)")
    return pd.DataFrame(coded, columns=list(participants.columns) + ["code"])


def _check_roi_coverage(
    empty_rois, n_staged, atlas, ignore_missing, logger, summaries, summary_dir
):
    """Explain subjects whose averaged matrix still has ROIs without data.

    The flat reader drops (or, if missing data is not ignored, rejects) every
    such subject. When that is every subject, the likely cause is an atlas
    reaching outside the scans' field of view, so stop with that explanation.
    """
    if not empty_rois:
        return
    shared = frozenset.intersection(*empty_rois.values())
    examples = ", ".join(str(i) for i in sorted(shared)[:10])
    if len(empty_rois) == n_staged:
        _write_summary(summaries, summary_dir)
        raise HalfpipeInputError(
            f"Every participant has ROIs of atlas {atlas} with no data in any included run "
            f"(present in all of them: ROI indices {examples or 'none in common'}), so none can be used. "
            "The atlas may extend beyond these scans' coverage; check the coverage or choose another atlas."
        )
    per_subject = "; ".join(f"{s}: {len(r)}" for s, r in sorted(empty_rois.items()))
    action = "they will be skipped" if ignore_missing else "the run will stop on them"
    logger.warning(
        f"{len(empty_rois)} of {n_staged} participant(s) have ROIs with no data in any included run "
        f"({per_subject} ROIs); {action}"
    )


def _write_summary(summaries, summary_dir):
    if not summary_dir:
        return
    os.makedirs(summary_dir, exist_ok=True)
    pd.DataFrame(
        [
            {
                "subject_id": s.subject_id,
                "status": s.status,
                "sessions": ";".join("ses-" + x for x in s.sessions),
                "runs_used": ";".join(s.runs_used),
                "runs_excluded": ";".join(s.runs_excluded),
                "matrix_size": s.matrix_size,
            }
            for s in summaries
        ]
    ).to_csv(os.path.join(summary_dir, QC_SUMMARY_FILE), index=False)
