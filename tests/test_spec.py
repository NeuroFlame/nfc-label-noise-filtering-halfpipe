"""Smoke tests for the computation spec and its pure-math helpers."""

import tempfile
import unittest

import numpy as np
import pandas as pd
from computation import constants
from computation.local_math import (
    LOCAL_CACHE_DATA_FILE,
    compute_dimensional_scores,
    cumulative_features_selection,
)
from computation.reports import _build_distribution_section
from computation.spec import SPEC
from computation.types import CentroidSet, GlobalCentroidResult
from framework import cache_file
from framework.workflow import get_task_names


class SpecTests(unittest.TestCase):
    """The stepped workflow builds and exposes the expected site task names."""

    def test_task_names(self):
        """Every local and output step is a distinct site task, in declared order."""
        self.assertEqual(
            get_task_names(SPEC.workflow),
            [
                "compute_local_crf",
                "compute_dimensional_scores",
                "relabel_subjects",
                "build_site_report",
            ],
        )


class CumulativeFeaturesSelectionTests(unittest.TestCase):
    """Features are kept up to (and excluding) the first non-significant p-value."""

    def test_keeps_features_below_threshold(self):
        p_values = np.array([0.5, 0.001, 0.2, 0.0001])
        selected = cumulative_features_selection(p_values, 0.01)
        # sorted ascending: [0.0001, 0.001, 0.2, 0.5] -> first two are <= 0.01
        self.assertEqual(sorted(selected.tolist()), [1, 3])

    def test_keeps_all_when_all_significant(self):
        p_values = np.array([0.001, 0.002])
        selected = cumulative_features_selection(p_values, 0.01)
        self.assertEqual(sorted(selected.tolist()), [0, 1])


class ComputeDimensionalScoresTests(unittest.TestCase):
    """A site resolves its own name via self_token and excludes itself from the average."""

    def test_own_site_uses_original_label_and_is_excluded_from_average(self):
        # Two subjects; the local site is "site2" and only "site1" is a peer.
        state = {"self_token": "token-for-site2"}
        cache_dir = self.enterContext(tempfile.TemporaryDirectory())
        np.save(
            cache_file(LOCAL_CACHE_DATA_FILE, cache_dir=cache_dir),
            np.array([[1.0, 2.0], [3.0, 4.0]]),  # last column is the label
        )
        global_result = GlobalCentroidResult(
            site_centroids={
                "site1": CentroidSet(
                    center_sz=np.array([[1.0]]),
                    center_hc=np.array([[3.0]]),
                    selected_features=np.array([0]),
                ),
                "site2": CentroidSet(
                    center_sz=np.array([[0.0]]),
                    center_hc=np.array([[0.0]]),
                    selected_features=np.array([0]),
                ),
            },
            token_to_name={
                "token-for-site1": "site1",
                "token-for-site2": "site2",
            },
        )

        result = compute_dimensional_scores(
            global_result, state=state, cache_dir=cache_dir
        )
        payload, next_state = result.payload, result.state

        self.assertEqual(next_state["my_name"], "site2")
        # Own label column equals the original labels, not a distance-based score.
        np.testing.assert_array_equal(
            next_state["site_scores"]["site2"], np.array([2.0, 4.0])
        )
        # The average excludes the own-site column entirely.
        np.testing.assert_array_equal(
            payload.scores, next_state["site_scores"]["site1"]
        )


class RelabeledDistributionTests(unittest.TestCase):
    """The report counts each relabeled group by its internal label value."""

    def test_counts_and_values_per_label(self):
        scores = pd.DataFrame({"re_labeled": [1.0, 1.0, 2.0, -1.0]})
        label_def = constants.get_label_definitions(
            patient_label="SZ", control_label="HC"
        )
        html = _build_distribution_section(scores, label_def, 0.085)
        self.assertNotIn("None", html)
        cells = '<td style="font-family:monospace">{}</td>'
        # SZ: value 1, 2 subjects (50%); HC: value 2, 1 subject (25%)
        self.assertIn(cells.format(1) + cells.format(2) + cells.format("50.0%"), html)
        self.assertIn(cells.format(2) + cells.format(1) + cells.format("25.0%"), html)


if __name__ == "__main__":
    unittest.main()
