"""Generate nested HALFpipe-style test sites for local simulation.

Writes test_data/site1..site3, each with a participants file and HALFpipe
derivatives nested as sub-*/[ses-*/]func/task-rest/, covering the layouts
the computation must handle:

    site1  sessions ses-V1A/ses-V1B, runs 1-2, participants.csv, qc.csv
           (some runs and one follow-up session excluded)
    site2  no session level, runs 1-2, participants.tsv (participant_id), no qc.csv
    site3  one session, runs 01-02, qc.csv, plus distractor files
           (other atlas, covariance, timeseries, Finder-style copies)

Matrices are NeuroMark1-sized (53x53) correlations of simulated time series.
Patients (diagnosis 1) have weaker within-domain coupling in two domains and
stronger coupling between two others. Some labels are flipped to create known
label noise; the truth is written to test_data/ground_truth.csv, outside the
site folders.

Usage: python3 scripts/generate_nested_test_data.py [output_dir]
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

DOMAIN_BOUNDS = [0, 5, 7, 16, 25, 42, 49, 53]  # NeuroMark1 domains (FNCDomainNames)
N_ROI = DOMAIN_BOUNDS[-1]
N_TIMEPOINTS = 300
PATIENT, CONTROL = 1, 2
ATLAS, FEATURE = "NeuroMark1", "neuromark"

SITES = {
    "site1": dict(
        top="halfpipe_derivatives",
        n=24,
        n_patients=12,
        sessions=["V1A", "V1B"],
        runs=["1", "2"],
        participants="participants.csv",
        qc=True,
        distractors=False,
        seed=101,
    ),
    "site2": dict(
        top="derivatives/halfpipe",
        n=24,
        n_patients=12,
        sessions=[None],
        runs=["1", "2"],
        participants="participants.tsv",
        qc=False,
        distractors=False,
        seed=202,
    ),
    "site3": dict(
        top="returned_from_scratch",
        n=20,
        n_patients=10,
        sessions=["1"],
        runs=["01", "02"],
        participants="participants.csv",
        qc=True,
        distractors=True,
        seed=303,
    ),
}
FLIP_RATE = 0.12
EFFECT = 0.35


def domains():
    """Map each of the 53 ROIs to its NeuroMark1 domain index."""
    labels = np.empty(N_ROI, dtype=int)
    for k in range(len(DOMAIN_BOUNDS) - 1):
        labels[DOMAIN_BOUNDS[k] : DOMAIN_BOUNDS[k + 1]] = k
    return labels


def subject_loadings(rng, is_patient, roi_domains):
    """Per-ROI coupling to its domain signal; weaker in two domains for patients."""
    loading = np.full(N_ROI, 0.8) + rng.normal(0, 0.1, N_ROI)
    if is_patient:
        loading[np.isin(roi_domains, [2, 4])] -= EFFECT
    return loading


def run_matrix(rng, loading, is_patient, roi_domains):
    """Correlation matrix of one simulated resting-state run."""
    shared = rng.normal(0, 1, (N_TIMEPOINTS, roi_domains.max() + 1))
    if is_patient:
        shared[:, 5] += EFFECT * 1.5 * shared[:, 1]
    ts = shared[:, roi_domains] * loading + rng.normal(0, 1, (N_TIMEPOINTS, 1)) * 0.3
    ts += rng.normal(0, 1, (N_TIMEPOINTS, N_ROI))
    corr = np.corrcoef(ts, rowvar=False)
    np.fill_diagonal(corr, 1.0)
    return corr


def write_tsv(path, matrix):
    """Write a matrix as tab-separated text, like HALFpipe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("\t".join(f"{x:.10f}" for x in row) for row in matrix))


def name(
    sub, ses, run, feature=FEATURE, atlas=ATLAS, suffix="desc-correlation_matrix.tsv"
):
    """HALFpipe-style filename for one run."""
    parts = (
        [f"sub-{sub}"]
        + ([f"ses-{ses}"] if ses else [])
        + ["task-rest", f"run-{run}", f"feature-{feature}", f"atlas-{atlas}"]
    )
    return "_".join(parts) + "_" + suffix


def generate_site(out, site, cfg, roi_domains, truth):
    """Write one site's derivatives, participants file and (optional) qc.csv."""
    rng = np.random.default_rng(cfg["seed"])
    site_dir = out / site
    true_labels = np.array(
        [PATIENT] * cfg["n_patients"] + [CONTROL] * (cfg["n"] - cfg["n_patients"])
    )
    rng.shuffle(true_labels)
    given = true_labels.copy()
    for idx in rng.choice(cfg["n"], int(round(FLIP_RATE * cfg["n"])), replace=False):
        given[idx] = CONTROL if given[idx] == PATIENT else PATIENT

    participants, qc_rows = [], []
    for i in range(cfg["n"]):
        sub = f"{site.upper()}{i + 1:03d}"
        is_patient = true_labels[i] == PATIENT
        loading = subject_loadings(rng, is_patient, roi_domains)
        for s_index, ses in enumerate(cfg["sessions"]):
            func = (
                site_dir
                / cfg["top"]
                / f"sub-{sub}"
                / (f"ses-{ses}" if ses else "")
                / "func"
            )
            for r_index, run in enumerate(cfg["runs"]):
                path = func / "task-rest" / name(sub, ses, run)
                write_tsv(path, run_matrix(rng, loading, is_patient, roi_domains))
                (
                    func
                    / name(sub, ses, run, suffix="stat-tsnr_boldmap.nii.gz").replace(
                        f"_feature-{FEATURE}_atlas-{ATLAS}", ""
                    )
                ).write_bytes(b"")
                include = "yes"
                if cfg["qc"] and i % 7 == 3 and r_index == 1:
                    include = "no"  # a run that failed QC
                if cfg["qc"] and s_index == 1 and i % 5 == 0:
                    include = "no"  # follow-up session not used for this subject
                qc_rows.append(
                    (f"sub-{sub}", f"ses-{ses}" if ses else "", run, include)
                )
                if cfg["distractors"] and i < 3:
                    task = func / "task-rest"
                    write_tsv(
                        task / name(sub, ses, run, "connectivity", "Schaefer200"),
                        run_matrix(rng, np.full(N_ROI, 0.5), False, roi_domains),
                    )
                    write_tsv(
                        task / name(sub, ses, run, suffix="desc-covariance_matrix.tsv"),
                        run_matrix(rng, loading, is_patient, roi_domains) * 3,
                    )
                    (task / name(sub, ses, run, suffix="timeseries.tsv")).write_text(
                        "0\t0\n"
                    )
                    shutil.copyfile(
                        path, path.with_name(path.name.replace(".tsv", " copy.tsv"))
                    )
        participants.append((f"sub-{sub}", "MF"[i % 2], 20 + i % 30, int(given[i])))
        truth.append(
            f"{site},sub-{sub},{true_labels[i]},{given[i]},{given[i] != true_labels[i]}"
        )

    if cfg["participants"].endswith(".tsv"):
        lines = ["participant_id\tsex\tage\tdiagnosis"] + [
            "\t".join(map(str, p)) for p in participants
        ]
    else:
        lines = ["subject_id,sex,age,diagnosis"] + [
            ",".join(map(str, p)) for p in participants
        ]
    (site_dir / cfg["participants"]).write_text("\n".join(lines) + "\n")
    if cfg["qc"]:
        lines = ["subject_id,session,run,include"] + [",".join(r) for r in qc_rows]
        (site_dir / "qc.csv").write_text("\n".join(lines) + "\n")


def main():
    """Generate all sites, ground_truth.csv and server/parameters.json."""
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1] / "test_data"
    )
    roi_domains = domains()
    for site in SITES:
        shutil.rmtree(out / site, ignore_errors=True)
    truth = ["site,subject_id,true_diagnosis,provided_diagnosis,label_flipped"]
    for site, cfg in SITES.items():
        generate_site(out, site, cfg, roi_domains, truth)
    (out / "ground_truth.csv").write_text("\n".join(truth) + "\n")
    (out / "server").mkdir(parents=True, exist_ok=True)
    (out / "server" / "parameters.json").write_text(
        json.dumps(
            {
                "SamplingThreshold": 0.7,
                "Iteration": 101,
                "NTree": 201,
                "LabelThreshold": 2,
                "TypicalThreshold": 0.8,
                "TruncationParameter": 0.2,
                "LabelFile": "participants.csv",
                "QCFile": "qc.csv",
                "ReadDataFileFromHalfpipeOutputs": {"atlas": ATLAS, "task": "rest"},
                "ControlLabel": CONTROL,
                "PatientLabel": PATIENT,
                "FNCDomainNames": DOMAIN_BOUNDS,
                "IgnoreSubjectsWithMissingData": True,
                "LogLevel": "info",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {', '.join(SITES)} and server/parameters.json to {out}")


if __name__ == "__main__":
    main()
