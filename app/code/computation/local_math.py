"""Site-side math: local CRF/centroids, dimensional scoring, and relabeling."""
import os

import numpy as np
import pandas as pd
from framework import cache_file, with_state
from numpy.random import PCG64, Generator, SeedSequence
from scipy.io import savemat
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind

from . import constants
from .crf import crf
from .reports import render_fnc_heatmap
from .types import (
    CentroidSet,
    CrfLocalResult,
    GlobalCentroidResult,
    LocalRelabelResult,
    LocalScoreResult,
    ValidatedInputs,
)
from .validation import get_complete_FNC_matrix_data

# Fixed so every site's CRF draws the same sequence of random numbers against
# its own local data, matching the original single-run seeding.
_CRF_SEED = 20250908
# Site data is cached as a file in the framework's run-scoped cache_dir (removed
# when the run ends) instead of being carried through serialized step state.
LOCAL_CACHE_DATA_FILE = "local_cache_data_file.npy"


def compute_local_crf(inputs: ValidatedInputs, *, parameters, logger, output_dir, cache_dir):
    """Run the local CRF step: typical-subject centroids + local FNC averages."""
    data = inputs.data
    rng = Generator(PCG64(SeedSequence(_CRF_SEED)))

    X = get_complete_FNC_matrix_data(data[:, :-1])
    y = data[:, -1]

    patient_label =  parameters[constants.PARAM_KEY_PATIENT_LABEL] #label1
    control_label =  parameters[constants.PARAM_KEY_CONTROL_LABEL]
    label_def = constants.get_label_definitions(patient_label=patient_label, control_label=control_label)

    domain_names = parameters.get("FNCDomainNames", constants.DEFAULT_FNCDomainNames)

    t_values, p_values = compute_two_sample_ttest(X, y)
    corrected_t_values = upper_triangle_bonferroni(t_values, p_values)
    render_fnc_heatmap(
        corrected_t_values,
        output_dir,
        "original_labels_ttest.png",
        title=f"Original {control_label} Vs {patient_label} T-test values",
        colorbar_name="T Values",
        domain_names=domain_names,
    )

    unique_labels = np.unique(y).astype(int)
    avg_fnc, local_sum_count = average_fnc_by_label(X, y, unique_labels)
    for label in unique_labels:
        label_name = label_def[str(label)]["name"]
        render_fnc_heatmap(
            avg_fnc[label],
            output_dir,
            f"local_original_avg_fnc_{label_name}.png",
            title=f"Average FNC of Original {label_name} Subjects",
            colorbar_name="Avg FNC Values",
            domain_names=domain_names,
        )

    savemat(f"{output_dir}/orig.mat", {"data": data}, do_compression=True)

    subject_noise_counts = crf.perform_crf(data, output_dir, parameters, rng)
    centroids = get_centroids(
        data, subject_noise_counts, parameters.get("TypicalThreshold")
    )

    np.savez(f"{output_dir}/centers.npz", **centroids)

    payload = CrfLocalResult(
        self_token=inputs.self_token,
        centroids=CentroidSet(
            center_sz=centroids["center_sz"],
            center_hc=centroids["center_hc"],
            selected_features=centroids["selected_features"],
        ),
        fnc_sum={str(label): local_sum_count[label]["sum"] for label in unique_labels},
        fnc_count={
            str(label): local_sum_count[label]["count"] for label in unique_labels
        },
    )
    # Save data in local cache as a file
    data_path = cache_file(LOCAL_CACHE_DATA_FILE, cache_dir=cache_dir)
    if not os.path.exists(data_path):
        np.save(data_path, data)

    state = {"self_token": inputs.self_token}
    return with_state(payload, state)


def compute_dimensional_scores(global_result: GlobalCentroidResult, *, state, cache_dir):
    """Compute each subject's dimensional score against every site's centroids."""
    my_name = global_result.token_to_name[state["self_token"]]

    data = np.load(cache_file(LOCAL_CACHE_DATA_FILE, cache_dir=cache_dir), allow_pickle=False)

    original_labels = data[:, -1]

    site_scores = {}
    for site_name, centroids in global_result.site_centroids.items():
        if site_name == my_name:
            site_scores[site_name] = original_labels
            continue

        selected_features = np.asarray(centroids.selected_features)
        ind_selected_data = data[:, selected_features]
        dist_sz = cdist(ind_selected_data, np.asarray(centroids.center_sz))
        dist_hc = cdist(ind_selected_data, np.asarray(centroids.center_hc))

        distance_typical_group_sz = dist_sz.mean(axis=1)
        distance_typical_group_hc = dist_hc.mean(axis=1)
        total_distance = distance_typical_group_sz + distance_typical_group_hc

        A = distance_typical_group_sz / total_distance
        B = distance_typical_group_hc / total_distance
        site_scores[site_name] = np.tan((A - B) * np.pi / 2)

    other_site_scores = [
        scores for name, scores in site_scores.items() if name != my_name
    ]
    stacked = np.where(np.array(other_site_scores) == 0, np.nan, other_site_scores)
    with np.errstate(invalid="ignore"):
        average = np.nanmean(stacked, axis=0)
    average = np.nan_to_num(average, nan=0.0)

    next_state = dict(state)
    next_state["my_name"] = my_name
    next_state["token_to_name"] = global_result.token_to_name
    next_state["site_scores"] = site_scores
    next_state["average_score"] = average
    return with_state(LocalScoreResult(scores=average), next_state)


def relabel_subjects(threshold, *, state, parameters, logger, output_dir, cache_dir):
    """Apply the adaptive threshold, then recompute FNC averages on relabeled subjects."""
    average_score = state["average_score"]

    data = np.load(cache_file(LOCAL_CACHE_DATA_FILE, cache_dir=cache_dir), allow_pickle=False)

    patient_label = parameters[constants.PARAM_KEY_PATIENT_LABEL] #label1
    control_label = parameters[constants.PARAM_KEY_CONTROL_LABEL]
    label_def = constants.get_label_definitions(patient_label=patient_label, control_label=control_label)

    total_subjects = average_score.shape[0]
    re_labels = np.full(total_subjects, constants.BOUNDARY_SUBJECT_LABEL_VALUE, dtype=float)
    mask_sz = average_score < -threshold
    mask_hc = average_score > threshold
    re_labels[mask_sz] = constants.PATIENT_LABEL_VALUE
    re_labels[mask_hc] = constants.CONTROL_LABEL_VALUE

    logger.info(f"Site metrices after relabeling: \n Patient count: {np.sum(mask_sz)}; \Control count: {np.sum(mask_hc)};"
                f" \Boundary count:{total_subjects - np.sum(mask_sz) - np.sum(mask_hc)} \n\n")

    scores_df = _build_scores_dataframe(state["site_scores"], average_score, re_labels)
    scores_df.to_csv(f"{output_dir}/relabeled.csv")

    X = get_complete_FNC_matrix_data(data[:, :-1])

    domain_names = parameters.get("FNCDomainNames", constants.DEFAULT_FNCDomainNames)

    t_values, p_values = compute_two_sample_ttest(X, re_labels)
    corrected_t_values = upper_triangle_bonferroni(t_values, p_values)
    render_fnc_heatmap(
        corrected_t_values,
        output_dir,
        "re_labeled_ttest.png",
        title=f"Relabeled {control_label} Vs {patient_label} T-test values",
        colorbar_name="T Values",
        domain_names=domain_names,
    )

    unique_labels = np.unique(re_labels).astype(int)
    unique_labels = unique_labels[unique_labels != constants.BOUNDARY_SUBJECT_LABEL_VALUE]
    avg_fnc, local_sum_count = average_fnc_by_label(X, re_labels, unique_labels)
    for label in unique_labels:
        label_name = label_def[str(label)]["name"]
        render_fnc_heatmap(
            avg_fnc[label],
            output_dir,
            f"local_relabeled_avg_fnc_{label_name}.png",
            title=f"Average FNC of Relabeled {label_name} Subjects",
            colorbar_name="Avg FNC Values",
            domain_names=domain_names,
        )

    next_state = dict(state)
    next_state["scores_df"] = scores_df
    payload = LocalRelabelResult(
        fnc_sum={str(label): local_sum_count[label]["sum"] for label in unique_labels},
        fnc_count={
            str(label): local_sum_count[label]["count"] for label in unique_labels
        },
    )
    return with_state(payload, next_state)


def _build_scores_dataframe(site_scores, average_score, re_labels):
    df = pd.DataFrame(site_scores)
    df["average"] = average_score
    df["re_labeled"] = re_labels
    return df


def split_dataset(dataset, labels):
    """Split a stack of FNC matrices into SZ/HC groups by label."""
    labels12 = np.asarray(labels).reshape(-1)
    group_sz = dataset[labels12 == constants.PATIENT_LABEL_VALUE]
    group_hc = dataset[labels12 == constants.CONTROL_LABEL_VALUE]
    return group_sz, group_hc


def compute_two_sample_ttest(dataset, labels12=None):
    """Welch's t-test per FNC cell, comparing HC vs. SZ subjects."""
    group_sz, group_hc = split_dataset(dataset, labels12)
    if group_sz.size == 0 or group_hc.size == 0:
        raise ValueError(f"Both groups need at least one subject. \nPatient group size: {group_sz.size}; Control group size: {group_hc.size}")
    return ttest_ind(group_hc, group_sz, axis=0, equal_var=False, nan_policy="omit")


def upper_triangle_bonferroni(t_values, p_values, alpha=0.01):
    """Mask FNC t-values that are not Bonferroni-significant in the upper triangle."""
    n = t_values.shape[0]
    upper_triangle_indexes = np.triu_indices(n, k=1)
    m_tests = len(upper_triangle_indexes[0])
    bonferroni_correction_value = alpha / m_tests

    t_matrix = np.array(t_values, copy=True)
    np.fill_diagonal(t_matrix, 0.0)

    sig_u = p_values[upper_triangle_indexes] < bonferroni_correction_value
    t_matrix[(upper_triangle_indexes[0][~sig_u], upper_triangle_indexes[1][~sig_u])] = (
        np.nan
    )
    return t_matrix


def find_sum_count_fnc(X, clip=1 - 1e-7):
    """Fisher-z sum/count of one group's FNC matrices, plus their back-transformed average."""
    p = X.shape[1]
    clipped_data = np.clip(X, -clip, clip)
    z_transformed = np.arctanh(clipped_data)

    diag = np.arange(p)
    z_transformed[:, diag, diag] = 0.0

    aggregated_sum_matrix = np.nansum(z_transformed, axis=0)
    total_count = np.sum(~np.isnan(z_transformed), axis=0)

    z_transformed_mean = np.nanmean(z_transformed, axis=0)
    inv_z_transformed = np.tanh(z_transformed_mean)
    inv_z_transformed = (inv_z_transformed + inv_z_transformed.T) / 2.0
    np.fill_diagonal(inv_z_transformed, 1.0)
    return {
        "sum": aggregated_sum_matrix,
        "count": total_count,
        "avg_fnc": inv_z_transformed,
    }


def average_fnc_by_label(X, y, labels):
    """Per-label average FNC matrix, plus the Fisher-z sum/count used to combine sites."""
    avg_fnc = {}
    local_sum_count = {}
    for lbl in labels:
        idx = y == lbl
        if not np.any(idx):
            raise ValueError(f"No subjects with label {lbl}")
        local_mean_data = find_sum_count_fnc(X[idx])
        avg_fnc[lbl] = local_mean_data["avg_fnc"]
        local_sum_count[lbl] = {
            "sum": local_mean_data["sum"],
            "count": local_mean_data["count"],
        }
    return avg_fnc, local_sum_count


def find_typical_subjects(original_labels, label_count, typical_threshold):
    """Split subjects into typical SZ/HC groups using their CRF non-noise ratio."""
    label_count[:, 3] = np.round(label_count[:, 3], 1)
    typical_indexes = np.where(label_count[:, 3] >= typical_threshold)[0]
    typical_labels = original_labels[typical_indexes]

    typical_sz = typical_indexes[typical_labels == constants.PATIENT_LABEL_VALUE]
    typical_hc = typical_indexes[typical_labels == constants.CONTROL_LABEL_VALUE]
    return np.array(typical_hc), np.array(typical_sz)


def cumulative_features_selection(p_values, p_value_threshold):
    """Bonferroni-style cumulative feature selection by ascending p-value."""
    feature_indexes = np.argsort(p_values)
    sorted_p_values = p_values[feature_indexes]

    above_threshold = np.where(sorted_p_values > p_value_threshold)[0]
    cutoff = above_threshold[0] if above_threshold.size > 0 else len(sorted_p_values)
    return feature_indexes[:cutoff]


def get_centroids(subject_data, subject_label_count, typical_threshold):
    """Compute typical-subject centroids and the features that define them."""
    col = subject_data.shape[1]
    typ_hc, typ_sz = find_typical_subjects(
        subject_data[:, -1], subject_label_count, typical_threshold
    )

    typical_sz_data = subject_data[typ_sz, :-1]
    typical_hc_data = subject_data[typ_hc, :-1]

    _, p_values = ttest_ind(typical_sz_data, typical_hc_data, axis=0, equal_var=True)
    significant_threshold = 0.01 / (col - 1)
    selected_features = cumulative_features_selection(p_values, significant_threshold)

    center_sz = np.mean(typical_sz_data[:, selected_features], axis=0).reshape(1, -1)
    center_hc = np.mean(typical_hc_data[:, selected_features], axis=0).reshape(1, -1)

    return {
        "center_sz": center_sz,
        "center_hc": center_hc,
        "selected_features": selected_features,
    }

