"""Varied, partly speculative HALFpipe site layouts for hardening the input conversion.

Each ``Scenario`` describes one or more sites (participants file, optional
qc.csv, nested derivatives) and what ``load_inputs`` should do at each site:
convert successfully with a given number of subjects, or stop with an error
mentioning a given phrase. ``write_scenario`` materializes a scenario on disk;
``tests/test_scenarios.py`` checks every scenario, and
``scripts/generate_scenarios.py`` writes them out for manual or simulator runs.

Several scenarios are speculative: they model what real sites could plausibly
send (AP/PA runs, multi-echo, symlinked scratch folders, macOS copy debris,
atlases that exceed scan coverage, inconsistent diagnosis coding) rather than
anything already seen.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

N_ROI = 53
DOMAIN_BOUNDS = [0, 5, 7, 16, 25, 42, 49, 53]
PATIENT, CONTROL = "1", "2"


@dataclass
class Run:
    """One correlation-matrix file: its filename entities and how to corrupt it."""

    entities: Dict[
        str, str
    ]  # ses, run, dir, acq, echo, ... (sub/task/feature/atlas added)
    feature: str = "neuromark"
    atlas: str = "NeuroMark1"
    task: str = "rest"
    n_roi: int = N_ROI
    nan_rois: Tuple[int, ...] = ()
    malformed: Optional[str] = (
        None  # header | empty | nonsquare | asymmetric | out_of_range
    )
    include: bool = True  # decision written to an auto-generated qc.csv


@dataclass
class Subject:
    label: str  # as it appears after "sub-" in filenames
    diagnosis: str
    runs: List[Run]
    participants_id: Optional[str] = None  # spelling in the participants file
    in_participants: bool = True


@dataclass
class Site:
    subjects: List[Subject]
    expect: Tuple  # ("ok", n_subjects) or ("error", "phrase")
    top: str = "halfpipe_derivatives"
    participants_name: Optional[str] = "participants.csv"
    id_column: str = "subject_id"
    bom_crlf: bool = False
    drop_diagnosis_column: bool = False
    qc: Optional[str] = "auto"  # "auto" | None
    qc_entity_columns: Tuple[str, ...] = ()  # e.g. ("dir",) for AP/PA runs
    qc_mutate: Optional[Callable[[List[Dict[str, str]]], List[Dict[str, str]]]] = None
    qc_bom_crlf: bool = False
    junk: bool = False  # .DS_Store, AppleDouble files, __MACOSX copies
    symlink_top: bool = (
        False  # derivatives live elsewhere; the site folder holds a symlink
    )
    check: Optional[Callable[[np.ndarray, pd.DataFrame], None]] = None


@dataclass
class Scenario:
    name: str
    description: str
    sites: Dict[str, Site]
    parameters: Dict = field(default_factory=dict)
    cross_site_mismatch: bool = (
        False  # sites convert fine but send different matrix sizes
    )


# --------------------------------------------------------------------------
# Matrix generation
# --------------------------------------------------------------------------


def _domains(n_roi):
    bounds = DOMAIN_BOUNDS if n_roi == N_ROI else np.linspace(0, n_roi, 8).astype(int)
    labels = np.empty(n_roi, dtype=int)
    for k in range(len(bounds) - 1):
        labels[bounds[k] : bounds[k + 1]] = k
    return labels


def _run_matrix(rng, is_patient, n_roi):
    """Correlations of simulated time series; patients couple less within two domains."""
    domains = _domains(n_roi)
    loading = np.full(n_roi, 0.8) + rng.normal(0, 0.1, n_roi)
    if is_patient:
        loading[np.isin(domains, [2, 4])] -= 0.35
    shared = rng.normal(0, 1, (150, domains.max() + 1))
    ts = shared[:, domains] * loading + rng.normal(0, 1, (150, n_roi))
    corr = np.corrcoef(ts, rowvar=False)
    np.fill_diagonal(corr, 1.0)
    return corr


def _write_matrix(path, matrix, malformed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if malformed == "empty":
        open(path, "w").close()
        return
    if malformed == "nonsquare":
        matrix = matrix[:, :-1]
    if malformed == "asymmetric":
        matrix = matrix.copy()
        matrix[0, 1] += 0.3
    if malformed == "out_of_range":
        matrix = matrix.copy()
        matrix[0, 1] = matrix[1, 0] = 3.33
    lines = [
        "\t".join("nan" if np.isnan(x) else f"{x:.10f}" for x in row) for row in matrix
    ]
    if malformed == "header":
        lines.insert(0, "\t".join(f"ROI{i + 1}" for i in range(matrix.shape[1])))
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def _filename(subject, run):
    order = ["ses", "task", "run", "dir", "acq", "echo"]
    entities = {"sub": subject.label, "task": run.task, **run.entities}
    parts = [f"sub-{entities.pop('sub')}"]
    for key in order:
        if key in entities:
            parts.append(f"{key}-{entities.pop(key)}")
    parts += [f"{k}-{v}" for k, v in entities.items()]
    parts += [f"feature-{run.feature}", f"atlas-{run.atlas}"]
    return "_".join(parts) + "_desc-correlation_matrix.tsv"


def _write_table(path, rows, columns, bom_crlf):
    sep = "\t" if path.endswith(".tsv") else ","
    lines = [sep.join(columns)] + [
        sep.join(str(r.get(c, "")) for c in columns) for r in rows
    ]
    text = ("\r\n" if bom_crlf else "\n").join(lines)
    with open(
        path, "w", encoding="utf-8-sig" if bom_crlf else "utf-8", newline=""
    ) as handle:
        handle.write(text)


def write_scenario(scenario: Scenario, root: str, seed: int = 0) -> Dict[str, str]:
    """Write every site of ``scenario`` under ``root``; return {site: site_dir}."""
    rng = np.random.default_rng(seed)
    site_dirs = {}
    for site_name, site in scenario.sites.items():
        site_dir = os.path.join(root, scenario.name, site_name)
        os.makedirs(site_dir, exist_ok=True)
        site_dirs[site_name] = site_dir
        top = os.path.join(site_dir, site.top)
        if site.symlink_top:
            real = os.path.join(root, scenario.name, f"_scratch_{site_name}")
            os.makedirs(real, exist_ok=True)
            os.makedirs(os.path.dirname(top), exist_ok=True)
            os.symlink(real, top)
        qc_rows = []
        for subject in site.subjects:
            for run in subject.runs:
                ses = run.entities.get("ses")
                func = os.path.join(
                    top,
                    f"sub-{subject.label}",
                    *([f"ses-{ses}"] if ses else []),
                    "func",
                )
                matrix = _run_matrix(
                    rng, subject.diagnosis.strip().lower() in (PATIENT, "sz"), run.n_roi
                )
                for roi in run.nan_rois:
                    matrix[roi, :] = np.nan
                    matrix[:, roi] = np.nan
                path = os.path.join(func, f"task-{run.task}", _filename(subject, run))
                _write_matrix(path, matrix, run.malformed)
                row = {
                    "subject_id": f"sub-{subject.label}",
                    "session": f"ses-{ses}" if ses else "",
                    "run": run.entities.get("run", ""),
                    "include": "yes" if run.include else "no",
                }
                for column in site.qc_entity_columns:
                    row[column] = run.entities.get(column, "")
                qc_rows.append(row)
                if site.junk:
                    open(os.path.join(func, ".DS_Store"), "w").close()
                    with open(
                        os.path.join(
                            os.path.dirname(path), "._" + os.path.basename(path)
                        ),
                        "wb",
                    ) as f:
                        f.write(b"\x00\x05\x16\x07AppleDouble")
                    mac = os.path.join(
                        site_dir,
                        "__MACOSX",
                        os.path.relpath(os.path.dirname(path), site_dir),
                    )
                    _write_matrix(
                        os.path.join(mac, os.path.basename(path)), matrix, None
                    )
        if site.participants_name:
            columns = [site.id_column, "sex", "age"] + (
                [] if site.drop_diagnosis_column else ["diagnosis"]
            )
            rows = [
                {
                    site.id_column: s.participants_id or f"sub-{s.label}",
                    "sex": "MF"[i % 2],
                    "age": 20 + i,
                    "diagnosis": s.diagnosis,
                }
                for i, s in enumerate(site.subjects)
                if s.in_participants
            ]
            _write_table(
                os.path.join(site_dir, site.participants_name),
                rows,
                columns,
                site.bom_crlf,
            )
        if site.qc == "auto":
            unique = {}
            for (
                row
            ) in qc_rows:  # one row per run, even when it has several feature files
                unique.setdefault(
                    tuple(sorted((k, v) for k, v in row.items() if k != "include")), row
                )
            qc_rows = list(unique.values())
            if site.qc_mutate:
                qc_rows = site.qc_mutate(qc_rows)
            columns = [
                "subject_id",
                "session",
                "run",
                *site.qc_entity_columns,
                "include",
            ]
            _write_table(
                os.path.join(site_dir, "qc.csv"), qc_rows, columns, site.qc_bom_crlf
            )
    return site_dirs


def scenario_parameters(scenario: Scenario) -> Dict:
    params = {
        "SamplingThreshold": 0.7,
        "Iteration": 3,
        "NTree": 5,
        "LabelThreshold": 2,
        "TypicalThreshold": 0.8,
        "TruncationParameter": 0.2,
        "LabelFile": "participants.csv",
        "QCFile": "qc.csv",
        "ReadDataFileFromHalfpipeOutputs": {"atlas": "NeuroMark1", "task": "rest"},
        "ControlLabel": 2,
        "PatientLabel": 1,
        "FNCDomainNames": DOMAIN_BOUNDS,
        "IgnoreSubjectsWithMissingData": True,
        "LogLevel": "info",
    }
    for key, value in scenario.parameters.items():
        if isinstance(value, dict) and isinstance(params.get(key), dict):
            params[key] = {**params[key], **value}
        else:
            params[key] = value
    return params


# --------------------------------------------------------------------------
# Scenario catalogue
# --------------------------------------------------------------------------


def _runs(sessions=(None,), runs=("1", "2"), dirs=(None,), extra=None, **run_kwargs):
    out = []
    for ses in sessions:
        for d in dirs:
            for r in runs:
                entities = {}
                if ses:
                    entities["ses"] = ses
                if r is not None:
                    entities["run"] = r
                if d:
                    entities["dir"] = d
                entities.update(extra or {})
                out.append(Run(entities=dict(entities), **run_kwargs))
    return out


def _group(
    prefix, n_patients=3, n_controls=3, patient=PATIENT, control=CONTROL, **run_args
):
    subjects = []
    for i in range(n_patients + n_controls):
        dx = patient if i < n_patients else control
        subjects.append(
            Subject(label=f"{prefix}{i + 1:03d}", diagnosis=dx, runs=_runs(**run_args))
        )
    return subjects


def _exclude(subject, **match):
    for run in subject.runs:
        if all(run.entities.get(k) == v for k, v in match.items()):
            run.include = False
    return subject


def _summary_runs(expected_per_subject):
    def check(data, summary):
        used = summary[summary.status == "included"].runs_used.str.split(";").str.len()
        assert used.tolist() == expected_per_subject, used.tolist()

    return check


def build_scenarios() -> List[Scenario]:
    s = []

    hcp = _group("HCP", sessions=("V1A", "V1B"), runs=("1", "2"), dirs=("AP", "PA"))
    _exclude(hcp[0], ses="V1B")  # follow-up half of the visit failed QC
    _exclude(hcp[1], dir="PA", run="2")  # one PA run failed in each session
    s.append(
        Scenario(
            "hcp_ap_pa",
            "HCP-style: 2 sessions x AP/PA x 2 runs, qc.csv with a dir column",
            {
                "site1": Site(
                    hcp,
                    ("ok", 6),
                    qc_entity_columns=("dir",),
                    check=_summary_runs([4, 6, 8, 8, 8, 8]),
                )
            },
        )
    )
    s.append(
        Scenario(
            "hcp_ap_pa_no_qc",
            "HCP-style runs without a QC file: all 8 runs averaged",
            {
                "site1": Site(
                    _group("HCP", sessions=("V1A", "V1B"), dirs=("AP", "PA")),
                    ("ok", 6),
                    qc=None,
                    check=_summary_runs([8] * 6),
                )
            },
        )
    )
    s.append(
        Scenario(
            "hcp_ap_pa_qc_without_dir",
            "AP/PA runs but qc.csv has no dir column: must say so",
            {
                "site1": Site(
                    _group("HCP", dirs=("AP", "PA")),
                    ("error", "add a column named dir"),
                )
            },
        )
    )
    s.append(
        Scenario(
            "multi_echo_acq",
            "acq-mb8 and echo-1/echo-2 entities, no QC file",
            {
                "site1": Site(
                    _group("ME", runs=("1",), extra={"acq": "mb8"}), ("ok", 6), qc=None
                )
            },
        )
    )
    me = _group("ME", runs=("1",))
    for subject in me:
        subject.runs = [Run(entities={"run": "1", "echo": e}) for e in ("1", "2")]
    s.append(
        Scenario(
            "multi_echo_qc",
            "Two echoes per run, qc.csv with an echo column",
            {
                "site1": Site(
                    me,
                    ("ok", 6),
                    qc_entity_columns=("echo",),
                    check=_summary_runs([2] * 6),
                )
            },
        )
    )
    s.append(
        Scenario(
            "no_run_entity",
            "Single run without a run- entity, with and without sessions",
            {
                "site1": Site(_group("NR", runs=(None,)), ("ok", 6)),
                "site2": Site(_group("NS", sessions=("1",), runs=(None,)), ("ok", 6)),
            },
        )
    )
    s.append(
        Scenario(
            "many_runs",
            "Six runs in one session",
            {
                "site1": Site(
                    _group(
                        "MR", sessions=("1",), runs=tuple(str(i) for i in range(1, 7))
                    ),
                    ("ok", 6),
                    check=_summary_runs([6] * 6),
                )
            },
        )
    )
    longitudinal = _group("LG", sessions=("baseline", "followup2yr"))
    for subject in longitudinal:
        _exclude(subject, ses="followup2yr")

    def baseline_only(data, summary):
        assert set(summary.sessions) == {"ses-baseline"}, set(summary.sessions)

    s.append(
        Scenario(
            "longitudinal_baseline_only",
            "Visits years apart; qc.csv keeps only baseline",
            {"site1": Site(longitudinal, ("ok", 6), check=baseline_only)},
        )
    )
    s.append(
        Scenario(
            "longitudinal_no_qc",
            "Visits years apart, no QC file: averaged (and logged)",
            {
                "site1": Site(
                    _group("LG", sessions=("baseline", "followup2yr")),
                    ("ok", 6),
                    qc=None,
                )
            },
        )
    )

    ids = _group("ID")
    ids[0].label, ids[0].participants_id = "007", "007"
    ids[1].label, ids[1].participants_id = "ABC12", "SUB-abc12"
    ids[2].label, ids[2].participants_id = "Pt3", "pt3"
    s.append(
        Scenario(
            "subject_id_variants",
            "Leading zeros, case differences, missing sub- prefix",
            {"site1": Site(ids, ("ok", 6))},
        )
    )

    bids = _group("BD")
    bids[0].diagnosis = "n/a"
    s.append(
        Scenario(
            "bids_participants_tsv",
            "participants.tsv with participant_id, BOM, CRLF, a n/a diagnosis",
            {
                "site1": Site(
                    bids,
                    ("ok", 5),
                    participants_name="participants.tsv",
                    id_column="participant_id",
                    bom_crlf=True,
                ),
                "site2": Site(
                    _group("BE"),
                    ("ok", 6),
                    participants_name="participants.tsv",
                    id_column="participant_id",
                    bom_crlf=True,
                ),
            },
        )
    )
    strict = _group("BS")
    strict[0].diagnosis = ""
    s.append(
        Scenario(
            "missing_diagnosis_strict",
            "Blank diagnosis with IgnoreSubjectsWithMissingData false",
            {"site1": Site(strict, ("error", "diagnosis"))},
            parameters={"IgnoreSubjectsWithMissingData": False},
        )
    )
    s.append(
        Scenario(
            "diagnosis_strings",
            "Text labels; one site writes them in lower case with spaces",
            {
                "site1": Site(_group("TX", patient="SZ", control="HC"), ("ok", 6)),
                "site2": Site(_group("TY", patient="sz", control=" hc "), ("ok", 6)),
            },
            parameters={"PatientLabel": "SZ", "ControlLabel": "HC"},
        )
    )
    unknown = _group("UK")
    unknown[0].diagnosis = "BP"
    s.append(
        Scenario(
            "diagnosis_unknown_value",
            "A third diagnosis value (BP) at one site",
            {"site1": Site(unknown, ("error", "BP"))},
        )
    )
    s.append(
        Scenario(
            "diagnosis_numeric_formats",
            "Diagnosis written as 1.0 / 2.0",
            {"site1": Site(_group("NF", patient="1.0", control="2.0"), ("ok", 6))},
        )
    )
    s.append(
        Scenario(
            "one_group_only",
            "A site with only controls",
            {
                "site1": Site(
                    _group("OG", n_patients=0, n_controls=6), ("error", "Both groups")
                )
            },
        )
    )

    partial = _group("NP")
    partial[0].runs[0].nan_rois = (10, 11)  # filled from run 2
    s.append(
        Scenario(
            "nan_one_run",
            "ROIs missing in one run only are filled from the other run",
            {"site1": Site(partial, ("ok", 6))},
        )
    )
    allruns = _group("NA")
    for subject in allruns[:2]:
        for run in subject.runs:
            run.nan_rois = (20,)
    s.append(
        Scenario(
            "nan_all_runs_some_subjects",
            "Two subjects missing an ROI in every run",
            {"site1": Site(allruns, ("ok", 4)), "site2": Site(_group("NB"), ("ok", 6))},
        )
    )
    s.append(
        Scenario(
            "nan_all_runs_strict",
            "As above with IgnoreSubjectsWithMissingData false",
            {"site1": Site(allruns, ("error", "ROIs with no data"))},
            parameters={"IgnoreSubjectsWithMissingData": False},
        )
    )
    coverage = _group("CV", nan_rois=(48, 49, 50))
    s.append(
        Scenario(
            "atlas_exceeds_coverage",
            "Same ROIs empty for every subject (e.g. cerebellum not scanned)",
            {"site1": Site(coverage, ("error", "ROI indices 48, 49, 50"))},
        )
    )

    s.append(
        Scenario(
            "qc_formatting",
            "qc.csv with BOM, CRLF, YES/No, run-01, SES- prefixes",
            {
                "site1": Site(
                    _group("QF", sessions=("V1",)),
                    ("ok", 6),
                    qc_bom_crlf=True,
                    qc_mutate=lambda rows: [
                        {
                            **r,
                            "include": "YES" if r["include"] == "yes" else " No ",
                            "run": f"run-0{r['run']}",
                            "session": r["session"].upper(),
                        }
                        for r in rows
                    ],
                )
            },
        )
    )
    s.append(
        Scenario(
            "qc_missing_rows",
            "qc.csv lacks rows for some runs",
            {
                "site1": Site(
                    _group("QM"),
                    ("error", "no row in"),
                    qc_mutate=lambda rows: rows[:-3],
                )
            },
        )
    )
    s.append(
        Scenario(
            "qc_typo_subject",
            "qc.csv has a mistyped subject marked yes",
            {
                "site1": Site(
                    _group("QT"),
                    ("error", "no matching correlation matrix"),
                    qc_mutate=lambda rows: rows
                    + [{**rows[0], "subject_id": "sub-QT0O1"}],
                )
            },
        )
    )
    s.append(
        Scenario(
            "qc_duplicate_rows",
            "qc.csv repeats a run",
            {
                "site1": Site(
                    _group("QD"),
                    ("error", "duplicate"),
                    qc_mutate=lambda rows: rows + rows[:1],
                )
            },
        )
    )
    s.append(
        Scenario(
            "qc_extra_columns",
            "qc.csv with reviewer notes and rater columns",
            {
                "site1": Site(
                    _group("QX"),
                    ("ok", 6),
                    qc_mutate=lambda rows: [
                        {**r, "notes": "ok", "rater": "JR"} for r in rows
                    ],
                )
            },
        )
    )

    extra_participants = _group("EP")
    extra_participants.append(Subject(label="EP099", diagnosis=CONTROL, runs=[]))
    s.append(
        Scenario(
            "participants_without_data",
            "A participant with no derivatives",
            {"site1": Site(extra_participants, ("ok", 6))},
        )
    )
    s.append(
        Scenario(
            "participants_without_data_strict",
            "As above with IgnoreSubjectsWithMissingData false",
            {"site1": Site(extra_participants, ("error", "EP099"))},
            parameters={"IgnoreSubjectsWithMissingData": False},
        )
    )
    unlabeled = _group("UL")
    unlabeled.append(
        Subject(label="UL099", diagnosis=CONTROL, runs=_runs(), in_participants=False)
    )
    s.append(
        Scenario(
            "derivatives_without_participant",
            "Derivatives for a subject not in participants",
            {"site1": Site(unlabeled, ("ok", 6), qc=None)},
        )
    )

    s.append(
        Scenario(
            "atlas_size_mismatch_across_sites",
            "Same atlas name, different matrix sizes at two sites",
            {
                "site1": Site(_group("AS"), ("ok", 6)),
                "site2": Site(_group("AT", n_roi=100), ("ok", 6)),
            },
            cross_site_mismatch=True,
        )
    )
    within = _group("AW")
    within[0].runs = _runs(n_roi=100)
    s.append(
        Scenario(
            "atlas_size_mismatch_within_site",
            "One subject's matrices have a different size",
            {"site1": Site(within, ("error", "same atlas"))},
        )
    )

    for kind, phrase in [
        ("header", "square"),
        ("empty", "empty"),
        ("nonsquare", "square"),
        ("asymmetric", "symmetric"),
        ("out_of_range", "outside"),
    ]:
        bad = _group("MF")
        bad[0].runs[0].malformed = kind
        s.append(
            Scenario(
                f"malformed_{kind}",
                f"One matrix file is malformed ({kind})",
                {"site1": Site(bad, ("error", phrase))},
            )
        )

    s.append(
        Scenario(
            "feature_name_variants",
            "Sites named the HALFpipe feature differently",
            {
                "site1": Site(_group("FA", feature="neuromark"), ("ok", 6)),
                "site2": Site(_group("FB", feature="fcNeuromark"), ("ok", 6)),
            },
        )
    )
    two = _group("TF")
    for subject in two:
        subject.runs += [
            Run(entities=dict(r.entities), feature="corrNM") for r in subject.runs
        ]
    s.append(
        Scenario(
            "two_features_same_atlas",
            "The atlas exists under two feature names",
            {"site1": Site(two, ("error", "feature to choose"))},
        )
    )
    s.append(
        Scenario(
            "two_features_same_atlas_feature_set",
            "As above with feature set in the parameters",
            {"site1": Site(two, ("ok", 6))},
            parameters={"ReadDataFileFromHalfpipeOutputs": {"feature": "neuromark"}},
        )
    )

    s.append(
        Scenario(
            "symlinked_derivatives",
            "Derivatives folder is a symlink into scratch space",
            {"site1": Site(_group("SL"), ("ok", 6), symlink_top=True)},
        )
    )
    s.append(
        Scenario(
            "macos_junk",
            ".DS_Store, AppleDouble ._ files and a __MACOSX copy of the tree",
            {"site1": Site(_group("MJ"), ("ok", 6), junk=True)},
        )
    )
    s.append(
        Scenario(
            "deep_nesting",
            "Derivatives several folders deep",
            {
                "site1": Site(
                    _group("DN"),
                    ("ok", 6),
                    top="outputs/2026-run2/derivatives/halfpipe",
                )
            },
        )
    )
    s.append(
        Scenario(
            "participants_missing_diagnosis_column",
            "participants file without a diagnosis column",
            {
                "site1": Site(
                    _group("PD"), ("error", "diagnosis"), drop_diagnosis_column=True
                )
            },
        )
    )
    s.append(
        Scenario(
            "no_participants_file",
            "Derivatives and qc.csv but no participants file",
            {
                "site1": Site(
                    _group("NP"), ("error", "participants"), participants_name=None
                )
            },
        )
    )
    s.append(
        Scenario(
            "wrong_task",
            "Only task-faces derivatives; parameters ask for rest",
            {
                "site1": Site(
                    _group("WT", task="faces"),
                    ("error", "Available: task-faces atlas-NeuroMark1"),
                    qc=None,
                )
            },
        )
    )
    s.append(
        Scenario(
            "atlas_name_typo",
            "Parameters name an atlas the sites do not have (NeuroMark2)",
            {
                "site1": Site(
                    _group("AN"), ("error", "none for task-rest atlas-NeuroMark2")
                )
            },
            parameters={"ReadDataFileFromHalfpipeOutputs": {"atlas": "NeuroMark2"}},
        )
    )
    s.append(
        Scenario(
            "atlas_name_case",
            "Parameters spell the atlas in different case (neuromark1)",
            {"site1": Site(_group("AC"), ("ok", 6))},
            parameters={"ReadDataFileFromHalfpipeOutputs": {"atlas": "neuromark1"}},
        )
    )
    return s


def describe_all() -> str:
    """One line per scenario, for the generator script."""
    return "\n".join(f"{sc.name:40s} {sc.description}" for sc in build_scenarios())


if __name__ == "__main__":
    print(describe_all())
