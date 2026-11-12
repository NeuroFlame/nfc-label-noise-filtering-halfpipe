"""Tests for converting nested HALFpipe derivatives into the flat layout."""

import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from computation.halfpipe_inputs import (
    QC_SUMMARY_FILE,
    STAGED_DIR_NAME,
    STAGED_LABEL_FILE,
    HalfpipeInputError,
    fisher_z_average,
    is_nested_halfpipe_dataset,
    normalize_run,
    stage_flat_dataset,
)
from computation.local_math import LOCAL_CACHE_DATA_FILE
from computation.remote_math import check_matching_fnc_shapes
from computation.results import remove_cached_site_data
from computation.types import CentroidSet, CrfLocalResult
from computation.validation import validate_and_get_inputs

LOGGER = logging.getLogger("test_halfpipe_inputs")
N_ROI = 4


def make_matrix(value, n=N_ROI):
    """A valid correlation matrix with every off-diagonal entry equal to ``value``."""
    matrix = np.full((n, n), float(value))
    np.fill_diagonal(matrix, 1.0)
    return matrix


def write_tsv(path, matrix):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, matrix, delimiter="\t", fmt="%.10f")


def run_file(
    root,
    sub,
    ses=None,
    run=None,
    feature="neuromark",
    atlas="NeuroMark1",
    desc="desc-correlation_matrix.tsv",
):
    """Path of a HALFpipe correlation matrix inside a nested derivatives tree."""
    entities = [f"sub-{sub}"] + ([f"ses-{ses}"] if ses else []) + ["task-rest"]
    entities += ([f"run-{run}"] if run else []) + [
        f"feature-{feature}",
        f"atlas-{atlas}",
    ]
    folder = os.path.join(
        root,
        "derivs_from_scratch",
        f"sub-{sub}",
        *([f"ses-{ses}"] if ses else []),
        "func",
        "task-rest",
    )
    return os.path.join(folder, "_".join(entities) + "_" + desc)


def write_participants(root, rows, name="participants.csv"):
    sep = "\t" if name.endswith(".tsv") else ","
    id_column = "participant_id" if name.endswith(".tsv") else "subject_id"
    pd.DataFrame(rows, columns=[id_column, "sex", "age", "diagnosis"]).to_csv(
        os.path.join(root, name), sep=sep, index=False
    )


def write_qc(root, rows):
    pd.DataFrame(rows, columns=["subject_id", "session", "run", "include"]).to_csv(
        os.path.join(root, "qc.csv"), index=False
    )


def parameters(**halfpipe):
    settings = {"atlas": "NeuroMark1"}
    settings.update(halfpipe)
    return {
        "LabelFile": "participants.csv",
        "ReadDataFileFromHalfpipeOutputs": settings,
        "ControlLabel": 2,
        "PatientLabel": 1,
        "IgnoreSubjectsWithMissingData": False,
    }


class HalfpipeConversionTestCase(unittest.TestCase):
    def setUp(self):
        self.root = self.enterContext(tempfile.TemporaryDirectory())
        self.cache = self.enterContext(tempfile.TemporaryDirectory())
        self.output = self.enterContext(tempfile.TemporaryDirectory())

    def stage(self, params=None):
        return stage_flat_dataset(
            self.root,
            params or parameters(),
            os.path.join(self.cache, "flat"),
            LOGGER,
            summary_dir=self.output,
        )

    def staged_labels(self, staged_dir):
        return pd.read_csv(os.path.join(staged_dir, STAGED_LABEL_FILE), dtype=str)

    def staged_matrix(self, staged_dir, staged_id):
        name = next(f for f in os.listdir(staged_dir) if f.startswith(staged_id + "_"))
        return np.genfromtxt(os.path.join(staged_dir, name), delimiter="\t")


class PhcpLayoutTests(HalfpipeConversionTestCase):
    """Two sessions and two runs per subject, renamed top folder, participants.tsv."""

    def setUp(self):
        super().setUp()
        for sub, values in {
            "PHCPP1011968": (0.1, 0.2, 0.3, 0.4),
            "PHCPP1011969": (0.5, 0.5, 0.5, 0.5),
            "PHCPP1011970": (0.2, 0.2, 0.2, 0.2),
        }.items():
            for (ses, run), value in zip(
                [("V1A", "01"), ("V1A", "02"), ("V1B", "01"), ("V1B", "02")],
                values,
                strict=True,
            ):
                write_tsv(run_file(self.root, sub, ses, run), make_matrix(value))
        write_participants(
            self.root,
            [
                ("sub-PHCPP1011968", "M", 24, 1),
                ("sub-PHCPP1011969", "F", 28, 2),
                ("sub-PHCPP1011970", "F", 30, 2),
            ],
            name="participants.tsv",
        )
        self.params = parameters()
        self.params["LabelFile"] = "participants.tsv"

    def test_is_detected_as_nested(self):
        self.assertTrue(is_nested_halfpipe_dataset(self.root, self.params))

    def test_without_qc_averages_every_run_and_session(self):
        staged_dir, staged_params = self.stage(self.params)
        labels = self.staged_labels(staged_dir)
        self.assertEqual(
            labels["subject_id"].tolist(),
            ["sub-PHCPP1011968", "sub-PHCPP1011969", "sub-PHCPP1011970"],
        )
        expected = np.tanh(np.mean(np.arctanh([0.1, 0.2, 0.3, 0.4])))
        np.testing.assert_allclose(
            self.staged_matrix(staged_dir, "sub-PHCPP1011968")[0, 1],
            expected,
            atol=1e-9,
        )

        ok, data = validate_and_get_inputs(staged_dir, staged_params, LOGGER)
        self.assertTrue(ok)
        self.assertEqual(data.shape, (3, N_ROI * (N_ROI - 1) // 2 + 1))
        self.assertEqual(data[:, -1].tolist(), [1.0, 2.0, 2.0])

    def test_qc_excludes_runs_and_sessions(self):
        rows = []
        for sub in ("PHCPP1011968", "PHCPP1011969", "PHCPP1011970"):
            for ses, run in [
                ("ses-V1A", "1"),
                ("ses-V1A", "2"),
                ("ses-V1B", "run-01"),
                ("ses-V1B", "02"),
            ]:
                rows.append((sub, ses, run, "yes"))
        rows[1] = ("PHCPP1011968", "ses-V1A", "2", "no")  # one failed run
        rows[2] = ("PHCPP1011968", "ses-V1B", "1", "no")  # follow-up visit excluded
        rows[3] = ("PHCPP1011968", "ses-V1B", "2", "No")
        write_qc(self.root, rows)
        staged_dir, _ = self.stage(self.params)
        np.testing.assert_allclose(
            self.staged_matrix(staged_dir, "sub-PHCPP1011968")[0, 1], 0.1, atol=1e-9
        )

        summary = pd.read_csv(
            os.path.join(self.output, QC_SUMMARY_FILE), dtype=str
        ).set_index("subject_id")
        self.assertEqual(
            summary.loc["sub-PHCPP1011968", "runs_used"],
            "sub-PHCPP1011968_ses-V1A_run-01",
        )
        self.assertEqual(summary.loc["sub-PHCPP1011969", "sessions"], "ses-V1A;ses-V1B")

    def test_unreviewed_run_is_an_error(self):
        write_qc(self.root, [("PHCPP1011968", "V1A", "1", "yes")])
        with self.assertRaisesRegex(HalfpipeInputError, "no row in qc.csv"):
            self.stage(self.params)

    def test_included_qc_row_without_file_is_an_error(self):
        rows = [
            (s, ses, r, "yes")
            for s in ("PHCPP1011968", "PHCPP1011969", "PHCPP1011970")
            for ses in ("V1A", "V1B")
            for r in ("1", "2")
        ]
        rows.append(("PHCPP1011968", "V2", "1", "yes"))
        write_qc(self.root, rows)
        with self.assertRaisesRegex(
            HalfpipeInputError, "with no matching correlation matrix"
        ):
            self.stage(self.params)

    def test_bad_qc_values_and_duplicates_are_errors(self):
        write_qc(
            self.root,
            [
                ("PHCPP1011968", "V1A", "1", "maybe"),
                ("PHCPP1011968", "V1A", "01", "yes"),
                ("PHCPP1011968", "V1A", "run-1", "yes"),
            ],
        )
        with self.assertRaises(HalfpipeInputError) as raised:
            self.stage(self.params)
        self.assertIn("include must be yes or no", str(raised.exception))
        self.assertIn("duplicate row", str(raised.exception))


class LayoutVariantTests(HalfpipeConversionTestCase):
    def test_no_session_level_and_text_ids_with_leading_zeros(self):
        write_tsv(run_file(self.root, "01", run="1"), make_matrix(0.3))
        write_tsv(run_file(self.root, "02", run="1"), make_matrix(0.6))
        write_participants(self.root, [("01", "M", 20, "1"), ("02", "F", 22, "2")])
        staged_dir, _ = self.stage()
        self.assertEqual(
            self.staged_labels(staged_dir)["subject_id"].tolist(), ["sub-01", "sub-02"]
        )
        # A single run is used unchanged.
        np.testing.assert_array_equal(
            self.staged_matrix(staged_dir, "sub-02"), make_matrix(0.6)
        )

    def test_distractor_files_are_ignored(self):
        write_tsv(run_file(self.root, "01", run="1"), make_matrix(0.3))
        write_tsv(
            run_file(self.root, "01", run="1", desc="desc-covariance_matrix.tsv"),
            make_matrix(9),
        )
        write_tsv(
            run_file(
                self.root, "01", run="1", feature="connectivity", atlas="Schaefer200"
            ),
            make_matrix(0.9),
        )
        copy_path = run_file(self.root, "01", run="1").replace(".tsv", " copy.tsv")
        write_tsv(copy_path, make_matrix(0.9))
        write_tsv(run_file(self.root, "02", run="1"), make_matrix(0.6))
        write_participants(self.root, [("sub-01", "M", 20, 1), ("sub-02", "F", 22, 2)])
        staged_dir, _ = self.stage()
        np.testing.assert_array_equal(
            self.staged_matrix(staged_dir, "sub-01"), make_matrix(0.3)
        )

    def test_same_atlas_under_two_feature_names_needs_feature_parameter(self):
        write_tsv(
            run_file(self.root, "01", run="1", feature="neuromark"), make_matrix(0.3)
        )
        write_tsv(run_file(self.root, "01", run="1", feature="fnc"), make_matrix(0.8))
        write_tsv(
            run_file(self.root, "02", run="1", feature="neuromark"), make_matrix(0.6)
        )
        write_participants(self.root, [("sub-01", "M", 20, 1), ("sub-02", "F", 22, 2)])
        with self.assertRaisesRegex(HalfpipeInputError, "feature to choose one"):
            self.stage()
        staged_dir, _ = self.stage(parameters(feature="neuromark"))
        np.testing.assert_array_equal(
            self.staged_matrix(staged_dir, "sub-01"), make_matrix(0.3)
        )

    def test_roi_missing_in_one_run_is_filled_from_the_other(self):
        first = make_matrix(0.3)
        first[2, :] = np.nan
        first[:, 2] = np.nan
        write_tsv(run_file(self.root, "01", run="1"), first)
        write_tsv(run_file(self.root, "01", run="2"), make_matrix(0.5))
        write_tsv(run_file(self.root, "02", run="1"), make_matrix(0.6))
        write_participants(self.root, [("sub-01", "M", 20, 1), ("sub-02", "F", 22, 2)])
        staged_dir, _ = self.stage()
        averaged = self.staged_matrix(staged_dir, "sub-01")
        self.assertAlmostEqual(averaged[0, 2], 0.5, places=6)
        self.assertAlmostEqual(
            averaged[0, 1], np.tanh(np.mean(np.arctanh([0.3, 0.5]))), places=6
        )

    def test_participant_without_files(self):
        write_tsv(run_file(self.root, "01", run="1"), make_matrix(0.3))
        write_tsv(run_file(self.root, "02", run="1"), make_matrix(0.6))
        write_participants(
            self.root,
            [("sub-01", "M", 20, 1), ("sub-02", "F", 22, 2), ("sub-03", "F", 23, 2)],
        )
        with self.assertRaisesRegex(HalfpipeInputError, "sub-03"):
            self.stage()
        params = parameters()
        params["IgnoreSubjectsWithMissingData"] = True
        staged_dir, _ = self.stage(params)
        self.assertEqual(
            self.staged_labels(staged_dir)["subject_id"].tolist(), ["sub-01", "sub-02"]
        )

    def test_different_matrix_sizes_at_one_site_are_an_error(self):
        write_tsv(run_file(self.root, "01", run="1"), make_matrix(0.3, n=4))
        write_tsv(run_file(self.root, "02", run="1"), make_matrix(0.6, n=5))
        write_participants(self.root, [("sub-01", "M", 20, 1), ("sub-02", "F", 22, 2)])
        with self.assertRaisesRegex(HalfpipeInputError, "same atlas"):
            self.stage()

    def test_flat_layout_is_not_treated_as_nested(self):
        flat = os.path.join(
            self.root,
            "sub-01_task-rest_feature-corrMatrix_atlas-schaefer2011CombinedDseg"
            "_desc-correlation_matrix.tsv",
        )
        write_tsv(flat, make_matrix(0.3))
        self.assertFalse(
            is_nested_halfpipe_dataset(self.root, parameters(atlas="schaefer2011"))
        )


class HelperTests(unittest.TestCase):
    def test_run_numbers(self):
        self.assertEqual(
            [normalize_run(v) for v in ("01", "1", "run-01", "RUN-2", "", None)],
            [1, 1, 1, 2, None, None],
        )

    def test_fisher_z_average(self):
        averaged = fisher_z_average([make_matrix(0.2), make_matrix(0.6)])
        self.assertAlmostEqual(
            averaged[0, 1], np.tanh((np.arctanh(0.2) + np.arctanh(0.6)) / 2)
        )
        self.assertEqual(averaged[0, 0], 1.0)

    def test_aggregator_rejects_mismatched_atlases(self):
        def result(n):
            centroids = CentroidSet(
                center_sz=np.zeros(1),
                center_hc=np.zeros(1),
                selected_features=np.zeros(1),
            )
            return CrfLocalResult(
                self_token="t",
                centroids=centroids,
                fnc_sum={"1": np.zeros((n, n))},
                fnc_count={"1": np.ones((n, n))},
            )

        check_matching_fnc_shapes({"site1": result(53), "site2": result(53)})
        with self.assertRaisesRegex(ValueError, "site1: 53x53, site2: 200x200"):
            check_matching_fnc_shapes({"site1": result(53), "site2": result(200)})

    def test_cached_site_data_is_removed(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            np.save(os.path.join(cache_dir, LOCAL_CACHE_DATA_FILE), np.zeros(2))
            os.makedirs(os.path.join(cache_dir, STAGED_DIR_NAME, "x"))
            remove_cached_site_data(cache_dir)
            remove_cached_site_data(cache_dir)  # safe to call again
            self.assertEqual(os.listdir(cache_dir), [])


if __name__ == "__main__":
    unittest.main()
