"""Write the HALFpipe input scenarios from tests/scenario_data.py to disk.

Usage:
    python3 scripts/generate_scenarios.py --list
    python3 scripts/generate_scenarios.py --out <dir> [--only name ...]
    python3 scripts/generate_scenarios.py --federation <dir>

--out writes each scenario as <dir>/<scenario>/<site>/ plus
<dir>/<scenario>/server/parameters.json and expected.json (what each site
should do). --federation writes a three-site set in the test_data layout
(site1..site3 + server/parameters.json) combining HCP-style AP/PA runs with
QC, multi-echo runs, and a longitudinal site that keeps only baseline, sized
for a full federated simulation.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
)
from scenario_data import (  # noqa: E402
    Run,
    Scenario,
    Site,
    _exclude,
    _group,
    build_scenarios,
    describe_all,
    scenario_parameters,
    write_scenario,
)


def write_all(out, only):
    """Write every (or the named) scenario with its parameters and expectations."""
    for scenario in build_scenarios():
        if only and scenario.name not in only:
            continue
        write_scenario(scenario, out)
        server = os.path.join(out, scenario.name, "server")
        os.makedirs(server, exist_ok=True)
        params = {**scenario_parameters(scenario), "Iteration": 101, "NTree": 201}
        with open(os.path.join(server, "parameters.json"), "w") as handle:
            json.dump(params, handle, indent=2)
        expected = {
            "description": scenario.description,
            "sites": {name: list(site.expect) for name, site in scenario.sites.items()},
            "cross_site_mismatch": scenario.cross_site_mismatch,
        }
        with open(os.path.join(out, scenario.name, "expected.json"), "w") as handle:
            json.dump(expected, handle, indent=2)
        print(f"wrote {scenario.name}")


def write_federation(out):
    """Three realistic, larger sites in the test_data layout for a federated run."""
    hcp = _group(
        "HCP", 6, 6, sessions=("V1A", "V1B"), runs=("1", "2"), dirs=("AP", "PA")
    )
    _exclude(hcp[0], ses="V1B")
    _exclude(hcp[7], dir="PA", run="2")
    echo = _group("ECHO", 6, 6, runs=("1",))
    for subject in echo:
        subject.runs = [
            Run(entities={"run": r, "echo": e}) for r in ("1", "2") for e in ("1", "2")
        ]
    longitudinal = _group("LONG", 7, 7, sessions=("baseline", "followup2yr"))
    for subject in longitudinal:
        _exclude(subject, ses="followup2yr")
    scenario = Scenario(
        "federation",
        "HCP AP/PA + multi-echo + longitudinal baseline-only",
        {
            "site1": Site(hcp, ("ok", 12), qc_entity_columns=("dir",)),
            "site2": Site(echo, ("ok", 12), qc_entity_columns=("echo",)),
            "site3": Site(longitudinal, ("ok", 14)),
        },
    )
    tmp = os.path.join(out, "_tmp")
    write_scenario(scenario, tmp, seed=7)
    for site in scenario.sites:
        shutil.rmtree(os.path.join(out, site), ignore_errors=True)
        shutil.move(os.path.join(tmp, "federation", site), os.path.join(out, site))
    shutil.rmtree(tmp)
    os.makedirs(os.path.join(out, "server"), exist_ok=True)
    params = {**scenario_parameters(scenario), "Iteration": 101, "NTree": 201}
    with open(os.path.join(out, "server", "parameters.json"), "w") as handle:
        json.dump(params, handle, indent=2)
    print(f"wrote federation sites to {out}")


def main():
    """Parse arguments and write the requested scenarios."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="list scenarios")
    parser.add_argument("--out", help="write every scenario under this folder")
    parser.add_argument(
        "--only", nargs="*", default=[], help="with --out: only these scenarios"
    )
    parser.add_argument("--federation", help="write a 3-site federated test set here")
    args = parser.parse_args()
    if args.list or not (args.out or args.federation):
        print(describe_all())
    if args.out:
        write_all(args.out, set(args.only))
    if args.federation:
        write_federation(args.federation)


if __name__ == "__main__":
    main()
