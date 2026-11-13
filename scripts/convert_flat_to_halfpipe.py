"""Convert a flat site (data.csv + labels.csv) into nested HALFpipe derivatives.

Each row of data.csv holds one subject's FNC upper triangle (np.triu_indices
order, diagonal excluded), as the original computation reads it. The script
rebuilds each symmetric matrix and writes it where stand-alone HALFpipe would:

    <out>/<site>/derivatives/halfpipe/sub-<ID>/[ses-<S>/]func/task-rest/
        sub-<ID>[_ses-<S>]_task-rest_run-1_feature-<f>_atlas-<atlas>_desc-correlation_matrix.tsv

plus participants.csv (subject_id, diagnosis as given) and qc.csv including
every run. With one run per subject the HALFpipe reader returns the original
values, so a run on the converted sites should reproduce a run on data.csv.

Usage:
    python3 scripts/convert_flat_to_halfpipe.py <flat test_data dir> <out dir>
        [--atlas NeuroMark1] [--session 1] [--sites site1 site2]

<flat test_data dir> is laid out like nfc-label-noise-filtering/test_data
(site*/data.csv, site*/labels.csv, server/parameters.json). The output gets the
same sites and a server/parameters.json set up for the HALFpipe reader.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE = "fnc"


def matrix_from_upper_triangle(values):
    """Symmetric matrix with unit diagonal from np.triu_indices(k=1) values."""
    n = int((1 + np.sqrt(1 + 8 * len(values))) / 2)
    if n * (n - 1) // 2 != len(values):
        raise ValueError(f"{len(values)} values are not the upper triangle of a matrix")
    matrix = np.eye(n)
    rows, cols = np.triu_indices(n, k=1)
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    return matrix


def write_matrix(path, matrix):
    """Write a matrix as tab-separated text with full float precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("\t".join(repr(float(x)) for x in row) for row in matrix))


def convert_site(flat_dir, out_dir, prefix, atlas, session):
    """Convert one flat site; return (subjects, diagnosis counts)."""
    data = np.loadtxt(
        open(flat_dir / "data.csv", encoding="utf-8-sig"), delimiter=",", ndmin=2
    )
    labels = pd.read_csv(flat_dir / "labels.csv", encoding="utf-8-sig", dtype=str)
    labels.columns = [c.strip() for c in labels.columns]
    if len(labels) != len(data):
        raise ValueError(
            f"{flat_dir}: {len(data)} data rows but {len(labels)} label rows"
        )
    derivatives = out_dir / "derivatives" / "halfpipe"
    participants, qc = [], []
    for index, (row, diagnosis) in enumerate(
        zip(data, labels["diagnosis"], strict=True), start=1
    ):
        subject = f"{prefix}{index:03d}"
        parts = [f"sub-{subject}"] + ([f"ses-{session}"] if session else [])
        folder = derivatives.joinpath(*parts, "func", "task-rest")
        name = "_".join(
            parts
            + ["task-rest", "run-1", f"feature-{FEATURE}", f"atlas-{atlas}"]
            + ["desc-correlation_matrix.tsv"]
        )
        write_matrix(folder / name, matrix_from_upper_triangle(row))
        participants.append((f"sub-{subject}", diagnosis.strip()))
        qc.append((f"sub-{subject}", f"ses-{session}" if session else "", "1", "yes"))
    pd.DataFrame(participants, columns=["subject_id", "diagnosis"]).to_csv(
        out_dir / "participants.csv", index=False
    )
    pd.DataFrame(qc, columns=["subject_id", "session", "run", "include"]).to_csv(
        out_dir / "qc.csv", index=False
    )
    return len(participants), pd.Series([d for _, d in participants]).value_counts()


def main():
    """Convert every requested site and write server/parameters.json."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("flat_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--atlas", default="NeuroMark1")
    parser.add_argument("--session", default="1", help="'' for no session level")
    parser.add_argument("--sites", nargs="*")
    args = parser.parse_args()

    sites = args.sites or sorted(
        p.name for p in args.flat_dir.iterdir() if (p / "data.csv").is_file()
    )
    for number, site in enumerate(sites, start=1):
        count, groups = convert_site(
            args.flat_dir / site,
            args.out_dir / site,
            f"S{number}R",
            args.atlas,
            args.session,
        )
        print(
            f"{site}: {count} subjects ({', '.join(f'{k} {v}' for k, v in groups.items())})"
        )

    params = json.loads((args.flat_dir / "server" / "parameters.json").read_text())
    params.pop("ExistingDataFile", None)
    params["LabelFile"] = "participants.csv"
    params["QCFile"] = "qc.csv"
    params["ReadDataFileFromHalfpipeOutputs"] = {"atlas": args.atlas, "task": "rest"}
    (args.out_dir / "server").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "server" / "parameters.json").write_text(
        json.dumps(params, indent=2) + "\n"
    )
    print(f"Wrote {', '.join(sites)} and server/parameters.json to {args.out_dir}")


if __name__ == "__main__":
    main()
