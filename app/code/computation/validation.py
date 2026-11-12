"""Validate one site's FNC data/labels CSVs against the computation parameters."""

import glob
import logging
import os
from distutils.util import strtobool

import numpy as np
import pandas as pd

from . import constants as ct


def validate_and_get_inputs(data_dir, parameters, logger):
    """Validate a site's ``data.csv``/``labels.csv`` and return the combined array.

    Returns ``(True, combined_data_and_labels)`` on success, where the
    returned array's last column is the label; on failure returns
    ``(False, None)`` after logging the reason.
    """
    try:
        # logger.info("Received parameters: ", str(parameters))
        logger.info("Performing validation of inputs...")
        ignore_subjects_with_missing_entries = parameters.get(
            "IgnoreSubjectsWithMissingData",
            ct.DEFAULT_IgnoreSubjectsWithMissingData,
        )
        ignore_subjects_with_missing_entries = bool(
            strtobool(str(ignore_subjects_with_missing_entries))
        )
        logger.debug(
            f" #### ignore_subjects_with_missing_entries = {str(ignore_subjects_with_missing_entries)}"
        )

        label_filename = parameters["LabelFile"]
        # 1. Load the labels CSV with specific columns
        labels_df = pd.read_csv(
            os.path.join(data_dir, label_filename),
            usecols=[ct.PARAM_KEY_SUBJECT_ID, ct.PARAM_KEY_DIAGNOSIS],
        )

        if not ignore_subjects_with_missing_entries:
            has_empty_cells = (
                labels_df.astype(str)
                .apply(lambda x: x.str.strip())
                .isin([""])
                .values.any()
            )
            has_nans = labels_df.isna().values.any()
            if has_nans or has_empty_cells:
                error_message = (
                    f"\n There are some NAN or cells with no values in  '{label_filename}'. "
                    f"Please correct and reupload.\n\n"
                )
                logger.error(error_message)
                return False, None
        else:
            # drop any subjects with missing values
            labels_df = labels_df.dropna()

        logger.debug("Loading_data_file")
        try:
            data_filename = parameters["ExistingDataFile"]
            data_df = pd.read_csv(os.path.join(data_dir, data_filename), header=None)

            # When the csv is provided it is assumed that the rows of csv files of data and
            # labels correspond to the same subject.
            data_df[ct.PARAM_KEY_SUBJECT_ID] = range(1, len(data_df) + 1)
            labels_df[ct.PARAM_KEY_SUBJECT_ID] = range(1, len(labels_df) + 1)

            # logger.info(f"Data file headers : {str(data_df.columns.tolist())}\n")
        except Exception:
            logger.info("Loading Subject specific FNCs data_file")

            # Data file is missing.. try loading all subject files from the directory
            # Extract unique subject IDs and
            subject_ids = labels_df["subject_id"].unique()
            data_df = load_subject_fnc_files(subject_ids, data_dir, parameters)
            logger.debug("Loading Done")

        if data_df is None:
            return False, None

        logger.debug("Checking lengths two dfs\n\n")

        if len(data_df) != len(labels_df):
            error_message = (
                "\n Number of rows in 'data' do not match with number of rows "
                "in 'labels.csv'. Please correct and reupload.\n\n"
            )
            logger.error(error_message)
            return False, None

        logger.debug("Length Passed.. two dfs\n\n")

        # validate labels
        patient_label = parameters.get(ct.PARAM_KEY_PATIENT_LABEL, None)
        control_label = parameters.get(ct.PARAM_KEY_CONTROL_LABEL, None)
        if not control_label:
            error_message = (
                f"Value provided for {ct.PARAM_KEY_CONTROL_LABEL} is empty or None.  Please provide a correct label for"
                f" controls used in the dataset.\n\n"
            )
            logger.error(error_message)
            return False, None
        if not patient_label:
            error_message = (
                f"Value provided for {ct.PARAM_KEY_PATIENT_LABEL}  is empty or None.  Please provide a correct label for"
                f" controls used in the dataset.\n\n"
            )
            logger.error(error_message)
            return False, None

        # Code patients as 1 and controls as 2. Labels match ignoring case and
        # surrounding whitespace, and numerically ("1" matches 1 and 1.0).
        codes = {
            _label_key(patient_label): ct.PATIENT_LABEL_VALUE,
            _label_key(control_label): ct.CONTROL_LABEL_VALUE,
        }
        coded = labels_df[ct.PARAM_KEY_DIAGNOSIS].map(
            lambda value: codes.get(_label_key(value))
        )
        unknown = sorted(
            {str(v) for v in labels_df.loc[coded.isna(), ct.PARAM_KEY_DIAGNOSIS]}
        )
        if unknown:
            error_message = (
                f"'{label_filename}' has diagnosis values that are neither "
                f"{ct.PARAM_KEY_PATIENT_LABEL} ({patient_label!r}) nor "
                f"{ct.PARAM_KEY_CONTROL_LABEL} ({control_label!r}): "
                f"{', '.join(repr(v) for v in unknown)}.\n\n"
            )
            logger.error(error_message)
            return False, None
        labels_df[ct.PARAM_KEY_DIAGNOSIS] = coded.astype(int)
        # logger.debug(f"updated labels : {str(labels_df.head(2))}")

        logger.debug("Merging two dfs\n\n")
        combined_df = pd.merge(
            data_df,
            labels_df,
            left_on=ct.PARAM_KEY_SUBJECT_ID,
            right_on=ct.PARAM_KEY_SUBJECT_ID,
            how="inner",
        )

        logger.debug("Completed Merging two dfs\n\n")

        combined_df.set_index(ct.PARAM_KEY_SUBJECT_ID, inplace=True)
        logger.debug("Reseting index\n\n")

        # Strip whitespace from all string/object columns
        combined_df = _trim_all_object_columns(combined_df)
        logger.debug("aFTER TRIMMING index\n\n")
        logger.debug(combined_df.head(2))

        # Convert the entire DataFrame to numeric (float), coercing errors to NaN
        combined_df = combined_df.apply(pd.to_numeric, errors="coerce")

        # Rows in nan
        all_rows_to_ignore = np.where(combined_df.isna().any(axis=1))[0].tolist()

        # Check for missing values in both data and labels
        if len(all_rows_to_ignore) > 0:
            if ignore_subjects_with_missing_entries:
                logger.info(
                    "-- Ignored following rows with incorrect/missing values: "
                    f"{str(combined_df.index[all_rows_to_ignore].tolist())}"
                )
                combined_df.drop(combined_df.index[all_rows_to_ignore], inplace=True)
            else:
                err_msg = (
                    "Following rows have empty or invalid entries. "
                    "Either choose to ignore these rows or correct the data and "
                    "try again. See log file for details: "
                    f"{str(combined_df.index[all_rows_to_ignore].tolist())}\n\n"
                )
                logger.error(err_msg)
                return False, None

        # Make sure there is data for both the labels
        logger.debug("Checking data lengths for each label")
        clean_patient_count = int(
            (combined_df[ct.PARAM_KEY_DIAGNOSIS] == ct.PATIENT_LABEL_VALUE).sum()
        )
        clean_control_count = int(
            (combined_df[ct.PARAM_KEY_DIAGNOSIS] == ct.CONTROL_LABEL_VALUE).sum()
        )

        if clean_patient_count == 0 or clean_control_count == 0:
            err_msg = "Both groups need at least one subject. "
            if clean_patient_count == 0:
                err_msg += " Number of patients after validation = 0. "
            if clean_control_count == 0:
                err_msg += " Number of controls after validation = 0. "
            logger.error(err_msg + "\n\n")
            return False, None

        combined_data_and_labels = combined_df.to_numpy()
        logger.info("\nData validation passed for the data.")
        logger.info(
            f"Number of data entries after validation: {len(combined_df)}. Running next steps.\n\n"
        )
        # logger.info(f"printing labels: {str(combined_data_and_labels[-1])}")

        return True, combined_data_and_labels

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}\n\n"
        logger.error(error_message)
        return False, None


def load_subject_fnc_files(subject_ids, data_dir, parameters):
    """Read each subject's FNC TSV and keep its upper-triangle values."""
    atlas = parameters["ReadDataFileFromHalfpipeOutputs"]["atlas"]
    corr_matrix = parameters["ReadDataFileFromHalfpipeOutputs"]["corr"]
    # Loop through and load the matching TSV files
    tsv_list = []
    missing_files = []
    for sub_id in subject_ids:
        # Construct the filename (adjust pattern if your files have a prefix/suffix)
        tsv_filename = f"{sub_id}_task-rest_feature-{corr_matrix}_atlas-{atlas}CombinedDseg_desc-correlation_matrix.tsv"

        # This works only in python 3.12+
        # files = glob.glob(os.path.join(data_dir, tsv_filename), case_sensitive=False)
        files = _case_insensitive_glob(os.path.join(data_dir, tsv_filename))
        # print(files)

        # Check if the file exists before attempting to read it
        if files:
            assert len(files) == 1, f"Found multiple files for subject. {files}\n\n"
            tsv_filename = files[0]
            # Read the TSV file using tab separation
            subj_fnc_matrix = np.genfromtxt(tsv_filename, delimiter="\t")

            # subj_fnc_matrix_df = pd.read_csv(tsv_filename, sep="\t", header=None)
            # subj_fnc_matrix = subj_fnc_matrix_df.to_numpy(dtype=float)

            # Extracts the upper triangular elements excluding the diagonal
            row_idx, col_idx = np.triu_indices(subj_fnc_matrix.shape[0], k=1)
            tsv_list.append(subj_fnc_matrix[row_idx, col_idx])
        else:
            missing_files.append(tsv_filename)

    if missing_files:
        display_missing_files = "\n".join(missing_files)
        logging.error(f"Following subject files not found: {display_missing_files}\n\n")
        return None

    # 4. Combine all individual TSV data into one DataFrame
    if tsv_list:
        data_df = pd.DataFrame(tsv_list)
        data_df[ct.PARAM_KEY_SUBJECT_ID] = subject_ids
    else:
        raise Exception("No matching TSV files were found or loaded.")

    return data_df


def get_complete_FNC_matrix_data(upper_triangle_data):
    """Reconstruct symmetric FNC matrices from their upper-triangle values."""
    n = int((1 + np.sqrt(1 + 8 * upper_triangle_data.shape[1])) / 2)

    complete_fnc_matrix = []
    for subj_id in range(upper_triangle_data.shape[0]):
        subj_fnc_matrix = np.zeros((n, n))
        triu_indices = np.triu_indices(n=n, k=1)

        subj_fnc_matrix[triu_indices] = upper_triangle_data[subj_id, :]
        subj_fnc_matrix += subj_fnc_matrix.T - np.diag(np.diag(subj_fnc_matrix))

        np.fill_diagonal(subj_fnc_matrix, 1)

        complete_fnc_matrix.append(subj_fnc_matrix)

    return np.asarray(complete_fnc_matrix)


def _trim_all_object_columns(df):
    """Trim whitespace from ends of each string value across all object columns."""
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].apply(
        lambda column: column.map(lambda v: v.strip() if isinstance(v, str) else v)
    )
    return df


def _label_key(value):
    """Comparable form of a diagnosis label: trimmed, lower case, numbers as numbers."""
    text = str(value).strip().lower()
    try:
        return repr(float(text))
    except ValueError:
        return text


def _case_insensitive_glob(pattern):
    # Example usage:
    # This will match test.txt, TEST.TXT, Test.Txt, etc.
    # files = _case_insensitive_glob("path/to/folder/*.txt")

    # Converts 'a' to '[aA]' and 'A' to '[aA]' for all alphabetical characters
    case_insensitive_pattern = "".join(
        f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in pattern
    )
    return glob.glob(case_insensitive_pattern)


def _get_user_row_numbers(df_index_list):
    return [ri + 1 for ri in df_index_list]
