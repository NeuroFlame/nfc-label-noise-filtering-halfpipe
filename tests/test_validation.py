"""Flat-reader input validation."""

import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from computation.validation import _trim_all_object_columns, validate_and_get_inputs

_LOGGER = logging.getLogger("test_validation")
_LOGGER.addHandler(logging.NullHandler())
_LOGGER.propagate = False


def _write_flat_site(folder, labels):
    """Write labels.csv and one 4x4 HALFpipe matrix per subject."""
    rng = np.random.default_rng(0)
    with open(os.path.join(folder, "labels.csv"), "w") as handle:
        handle.write("subject_id,diagnosis\n")
        for index, label in enumerate(labels):
            subject = f"sub-{index:02d}"
            handle.write(f"{subject},{label}\n")
            values = rng.uniform(-0.5, 0.5, (4, 4))
            matrix = (values + values.T) / 2
            np.fill_diagonal(matrix, 1.0)
            name = (
                f"{subject}_task-rest_feature-corrMatrix_atlas-"
                "testCombinedDseg_desc-correlation_matrix.tsv"
            )
            np.savetxt(os.path.join(folder, name), matrix, delimiter="\t")


def _parameters(patient, control):
    return {
        "LabelFile": "labels.csv",
        "ReadDataFileFromHalfpipeOutputs": {"atlas": "test", "corr": "corrMatrix"},
        "PatientLabel": patient,
        "ControlLabel": control,
    }


class FlatReaderLabelTests(unittest.TestCase):
    """Patient and control labels are coded 1 and 2 whatever their spelling."""

    def _labels_read(self, labels, patient, control):
        with tempfile.TemporaryDirectory() as folder:
            _write_flat_site(folder, labels)
            ok, data = validate_and_get_inputs(
                folder, _parameters(patient, control), _LOGGER
            )
        self.assertTrue(ok)
        return data[:, -1].tolist()

    def test_text_labels(self):
        """Text diagnoses such as SZ/HC used to crash the whitespace trim."""
        labels = self._labels_read(["SZ", "HC", "sz", "HC"], "SZ", "HC")
        self.assertEqual(labels, [1, 2, 1, 2])

    def test_numeric_labels(self):
        """Numeric diagnoses keep working."""
        labels = self._labels_read([1, 2, 1, 2], 1, 2)
        self.assertEqual(labels, [1, 2, 1, 2])

    def test_mixed_case_and_whitespace(self):
        """Labels match ignoring case and surrounding spaces."""
        labels = self._labels_read(["SZ", " hc", "sz ", "Hc"], "SZ", "HC")
        self.assertEqual(labels, [1, 2, 1, 2])

    def test_numeric_labels_given_as_text(self):
        """PatientLabel "1" matches a numeric 1 in the label file."""
        labels = self._labels_read([1, 2, 1, 2], "1", "2")
        self.assertEqual(labels, [1, 2, 1, 2])

    def test_unknown_label_is_rejected(self):
        """A value that is neither label stops the site."""
        with tempfile.TemporaryDirectory() as folder:
            _write_flat_site(folder, ["SZ", "HC", "BP", "HC"])
            ok, data = validate_and_get_inputs(folder, _parameters("SZ", "HC"), _LOGGER)
        self.assertFalse(ok)
        self.assertIsNone(data)

    def test_trim_handles_non_string_object_values(self):
        """An object column holding ints (coded labels) is left as is."""
        frame = pd.DataFrame({"a": pd.Series([1, 2], dtype=object), "b": [" x", "y "]})
        trimmed = _trim_all_object_columns(frame)
        self.assertEqual(trimmed["a"].tolist(), [1, 2])
        self.assertEqual(trimmed["b"].tolist(), ["x", "y"])


if __name__ == "__main__":
    unittest.main()
