"""Complete random forest (CRF) construction for label-noise detection."""

import os
from math import floor
from typing import Any, Dict

import numpy as np
from joblib import Parallel, delayed
from numpy.random import Generator
from scipy.io import savemat
from scipy.stats import zscore

from .crf_helpers import is_continuous
from .iterative_crf import build_crf_results_iter


def perform_crf(
    dataset: np.ndarray, result_path: str, parameters: Dict[str, Any], rng: Generator
):
    """Repeatedly subsample and build CRFs to flag noisy-label subjects.

    ``dataset`` holds subjects as rows, with their features as columns and
    the subject's label in the last column.
    """
    subject_count, features_count = dataset.shape

    labels = dataset[:, -1]
    label_classes = np.unique(labels)
    len_label_classes = len(label_classes)
    # class is either 1 (SZ) or 2 (HC)
    original_cls_label_indexes = dict()
    sampling_cls_label_count = [0] * len_label_classes

    for i in range(len_label_classes):
        original_cls_label_indexes[label_classes[i]] = np.where(
            labels == label_classes[i]
        )[0]
        sampling_cls_label_count[i] = floor(
            len(original_cls_label_indexes[label_classes[i]])
            * parameters.get("SamplingThreshold")
        )

    dtype = np.dtype(
        [
            ("subject_id", np.int64),  # col 0
            ("count", np.int64),  # col 1
            ("non_noise_count", np.int64),  # col 2
            ("ratio", np.float64),  # col 3
        ]
    )

    sub_noise_per_iter = np.zeros(subject_count, dtype=dtype)
    sub_noise_per_iter["subject_id"] = np.arange(
        1, subject_count + 1, dtype=int
    )  # id's of all subjects
    mean_sub_sampling_length = int(np.mean(sampling_cls_label_count))

    # for each sampling, subjects Ids which are not noise
    non_noise_sampling_subjects = []
    nltc_decisions = []
    for i in range(parameters.get("Iteration")):
        print(f"Constructing No. {i + 1} CRF")
        index_temp = []
        for cls in label_classes:
            random_sampling_indexes = rng.permutation(original_cls_label_indexes[cls])[
                :mean_sub_sampling_length
            ]
            index_temp.extend(random_sampling_indexes)

        random_indices = np.array(index_temp).astype(int)

        sub_noise_per_iter["count"][random_indices] += 1
        sampled_dataset = dataset[random_indices, :]
        attributes = zscore(sampled_dataset[:, 0 : features_count - 1], axis=0, ddof=1)
        sampled_dataset_labels = sampled_dataset[:, -1].reshape(-1, 1)

        training_data = np.hstack((sampled_dataset_labels, attributes))

        # denoise_data, non_noise_ID, NLTC_labels =  running_crf(training_data, ntree, NI_threshold)
        non_noise_ids, nltc_labels = crf_v1(training_data, parameters, rng=rng)
        nltc_decisions.append(nltc_labels)
        # print("picked sampling number: ", sampling_indexes[non_noise_ids, i])
        non_noise_sampling_subjects.append(random_indices[non_noise_ids])

    denoise_check = np.zeros((subject_count,), dtype=int)
    for i in range(parameters.get("Iteration")):
        idxs = non_noise_sampling_subjects[i]
        denoise_check[idxs] += 1

    # print("denoise count for each subject: ", denoise_check)
    sub_noise_per_iter["non_noise_count"] = denoise_check

    np.divide(
        sub_noise_per_iter["non_noise_count"],
        sub_noise_per_iter["count"],
        out=sub_noise_per_iter["ratio"],
        where=sub_noise_per_iter["count"] != 0,
    )
    sub_noise_per_iter["ratio"] = np.round(sub_noise_per_iter["ratio"], 1)

    final_mat = np.column_stack(
        (
            sub_noise_per_iter["subject_id"],
            sub_noise_per_iter["count"],
            sub_noise_per_iter["non_noise_count"],
            sub_noise_per_iter["ratio"],
        )
    )
    # print("iteration matrix: ", final_mat)
    # For non_noise_sampling_subjects, as each array is of different size,
    # convert to a 1D NumPy array of 'object' type. This acts like a container and
    # preserves the individual array shapes
    non_noise_sampling_subjects_object_array = np.empty(
        len(non_noise_sampling_subjects), dtype=object
    )

    for i, arr in enumerate(non_noise_sampling_subjects):
        non_noise_sampling_subjects_object_array[i] = arr

    output_path = os.path.join(result_path, "crf.mat")
    savemat(
        output_path,
        {
            "count": final_mat,
            "nltc_labels": np.asarray(nltc_decisions),
            "non_noise_ind": non_noise_sampling_subjects_object_array,
        },
        do_compression=True,
    )

    return final_mat


def crf_v1(train_data: np.ndarray, parameters: Dict[str, Any], rng: Generator):
    """Build one CRF pass with unique feature splits for this iteration."""
    total_trees = parameters.get("NTree")
    children = rng.bit_generator._seed_seq.spawn(2 * total_trees)
    ss_f1 = children[:total_trees]
    ss_f2 = children[total_trees : 2 * total_trees]

    subjects_count, _ = train_data.shape
    is_continuous_data = is_continuous(train_data[:, 1:])
    train_data = np.hstack(
        (train_data, np.arange(1, subjects_count + 1).reshape(-1, 1))
    )

    with Parallel(n_jobs=8, prefer="processes", batch_size="auto") as parallel:
        final_output_1 = parallel(
            delayed(build_crf_results_iter)(train_data, is_continuous_data, 1, ss_f1[_])
            for _ in range(total_trees)
        )

        final_output_2 = parallel(
            delayed(build_crf_results_iter)(train_data, is_continuous_data, 2, ss_f2[_])
            for _ in range(total_trees)
        )

    # for i in range(ntree):
    #     final_output_1[i] = crf.build_crf_results_iter(train_data, is_continuous_data, 1)
    #     final_output_2[i] = crf.build_crf_results_iter(train_data, is_continuous_data, 2)

    final_output_1 = np.hstack(final_output_1)
    final_output_2 = np.hstack(final_output_2)

    final_decisions = np.hstack([final_output_1, final_output_2])
    label_threshold = parameters.get("LabelThreshold")
    final_decisions[final_decisions < label_threshold] = 0  # non-noise
    final_decisions[final_decisions >= label_threshold] = 1  # noise

    noise_subjects = (final_decisions.sum(axis=1) > 0.5 * total_trees * 2).astype(int)

    non_noise_subject_id: np.ndarray = np.where(noise_subjects == 0)[0]

    # non_noise_data = train_data[non_noise_subject_id, :]

    return non_noise_subject_id, final_decisions
