"""Run every scenario in scenario_data through the site input step."""

import logging
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from computation.halfpipe_inputs import QC_SUMMARY_FILE
from computation.inputs import load_inputs
from computation.remote_math import check_matching_fnc_shapes
from computation.types import CentroidSet, CrfLocalResult

sys.path.insert(0, os.path.dirname(__file__))
from scenario_data import (  # noqa: E402
    build_scenarios,
    scenario_parameters,
    write_scenario,
)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def run_site(site_dir, parameters):
    """Run load_inputs at one site; return (data or None, error text, summary, log text)."""
    logger = logging.getLogger(f"scenario.{site_dir}")
    logger.propagate = False
    capture = _Capture()
    logger.handlers = [capture]
    logger.setLevel(logging.DEBUG)
    with (
        tempfile.TemporaryDirectory() as cache,
        tempfile.TemporaryDirectory() as output,
    ):
        data, error = None, ""
        try:
            data = load_inputs(site_dir, parameters, logger, cache, output).data
        except Exception as exc:  # noqa: BLE001 - every failure is reported to the test
            error = f"{exc} {exc.__cause__ or ''}"
        summary_path = os.path.join(output, QC_SUMMARY_FILE)
        summary = (
            pd.read_csv(summary_path, dtype=str, keep_default_na=False)
            if os.path.exists(summary_path)
            else None
        )
        leftovers = [f for _, _, files in os.walk(cache) for f in files]
    return data, error, summary, "\n".join(capture.messages), leftovers


class ScenarioTests(unittest.TestCase):
    """Each scenario's sites convert, or fail with the expected message."""


def _make_test(scenario):
    def test(self):
        with tempfile.TemporaryDirectory() as root:
            site_dirs = write_scenario(scenario, root)
            params = scenario_parameters(scenario)
            widths = {}
            for site_name, site in scenario.sites.items():
                with self.subTest(site=site_name):
                    data, error, summary, log, _ = run_site(
                        site_dirs[site_name], params
                    )
                    outcome, value = site.expect
                    if outcome == "ok":
                        self.assertIsNotNone(
                            data, f"expected success, got: {error}\nlog:\n{log}"
                        )
                        self.assertEqual(data.shape[0], value, f"subjects\nlog:\n{log}")
                        widths[site_name] = data.shape[1] - 1
                        if site.check:
                            site.check(data, summary)
                    else:
                        self.assertIsNone(
                            data, f"expected an error mentioning {value!r}\nlog:\n{log}"
                        )
                        self.assertIn(
                            value.lower(),
                            (error + "\n" + log).lower(),
                            f"\nerror: {error}\nlog:\n{log}",
                        )
            if scenario.cross_site_mismatch:
                self.assertGreater(len(set(widths.values())), 1)
                results = {name: _fake_result(width) for name, width in widths.items()}
                with self.assertRaisesRegex(ValueError, "different sizes"):
                    check_matching_fnc_shapes(results)

    test.__doc__ = scenario.description
    return test


def _fake_result(n_features):
    n = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
    centroids = CentroidSet(
        center_sz=np.zeros(1), center_hc=np.zeros(1), selected_features=np.zeros(1)
    )
    return CrfLocalResult(
        self_token="t",
        centroids=centroids,
        fnc_sum={"1": np.zeros((n, n))},
        fnc_count={"1": np.ones((n, n))},
    )


for _scenario in build_scenarios():
    setattr(ScenarioTests, f"test_{_scenario.name}", _make_test(_scenario))


if __name__ == "__main__":
    unittest.main()
