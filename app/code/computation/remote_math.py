"""Server-side aggregation: centroid pooling, adaptive threshold, relabeled metrics."""

import os
from typing import Dict

import numpy as np
from framework import with_state

from . import constants
from .reports import render_fnc_heatmap
from .types import (
    CrfLocalResult,
    GlobalCentroidResult,
    GlobalReportData,
    LocalRelabelResult,
    LocalScoreResult,
)


def aggregate_centroids(
    site_results: Dict[str, CrfLocalResult], *, parameters, output_dir
):
    """Pool every site's centroids and compute the federated original-label FNC average."""
    sorted_names = sorted(site_results.keys())
    check_matching_fnc_shapes(site_results)

    site_centroids = {name: site_results[name].centroids for name in sorted_names}
    token_to_name = {site_results[name].self_token: name for name in sorted_names}

    fnc_sums = {name: site_results[name].fnc_sum for name in sorted_names}
    fnc_counts = {name: site_results[name].fnc_count for name in sorted_names}
    global_avg_fnc_original = global_mean_from_sites(fnc_sums, fnc_counts)

    domain_names = parameters.get(
        constants.PARAM_KEY_FNC_DOMAIN_NAMES, constants.DEFAULT_FNCDomainNames
    )
    global_result_dir = os.path.join(output_dir, "global_results")
    os.makedirs(global_result_dir, exist_ok=True)

    patient_label = parameters[constants.PARAM_KEY_PATIENT_LABEL]  # label1
    control_label = parameters[constants.PARAM_KEY_CONTROL_LABEL]
    label_def = constants.get_label_definitions(
        patient_label=patient_label, control_label=control_label
    )

    for label, matrix in global_avg_fnc_original.items():
        label_name = label_def[label]["name"]
        render_fnc_heatmap(
            matrix,
            global_result_dir,
            f"global_original_avg_fnc_{label_name}.png",
            title=f"Average FNC of Original {label_name} Subjects",
            colorbar_name="Avg FNC Values",
            domain_names=domain_names,
        )

    payload = GlobalCentroidResult(
        site_centroids=site_centroids, token_to_name=token_to_name
    )
    state = {"global_avg_fnc_original": global_avg_fnc_original}
    return with_state(payload, state)


def compute_adaptive_threshold(
    site_results: Dict[str, LocalScoreResult], *, state, parameters
):
    """Combine every site's dimensional scores into one global adaptive threshold."""
    sorted_names = sorted(site_results.keys())
    global_scores = np.concatenate([site_results[name].scores for name in sorted_names])

    valid_mask = ~np.isnan(global_scores)
    s_valid = global_scores[valid_mask]
    order = np.argsort(s_valid, kind="mergesort")
    sorted_scores = s_valid[order]

    neg_mask_sorted = sorted_scores < 0
    pos_mask_sorted = sorted_scores > 0
    n_neg = int(np.sum(neg_mask_sorted))
    n_pos = int(np.sum(pos_mask_sorted))

    def round_clamp(x, lo, hi):
        idx = int(np.floor(x + 0.5))
        return max(lo, min(hi, idx))

    truncation = parameters.get("TruncationParameter")
    idx_neg0 = -1
    idx_pos0 = -1
    if n_neg > 0:
        idx_neg_1b = round_clamp(n_neg * (1.0 - truncation), 1, n_neg)
        idx_neg0 = idx_neg_1b - 1
    if n_pos > 0:
        idx_pos_1b = n_neg + round_clamp(n_pos * truncation, 1, n_pos)
        idx_pos0 = idx_pos_1b - 1

    if idx_neg0 > 0 and idx_pos0 > 0:
        threshold = abs(sorted_scores[idx_neg0] + sorted_scores[idx_pos0]) / 2.0
    elif idx_neg0 > 0:
        threshold = abs(sorted_scores[idx_neg0])
    elif idx_pos0 > 0:
        threshold = abs(sorted_scores[idx_pos0])
    else:
        threshold = 0.0

    if not np.isfinite(threshold):
        threshold = 0.0
    threshold = float(max(0.0, threshold))

    next_state = dict(state)
    next_state["adaptive_threshold"] = threshold
    return with_state(threshold, next_state)


def aggregate_relabeled_metrics(
    site_results: Dict[str, LocalRelabelResult], *, state, parameters, output_dir
):
    """Combine relabeled FNC averages across sites into the final report payload."""
    sorted_names = sorted(site_results.keys())
    fnc_sums = {name: site_results[name].fnc_sum for name in sorted_names}
    fnc_counts = {name: site_results[name].fnc_count for name in sorted_names}
    global_avg_fnc_relabeled = global_mean_from_sites(fnc_sums, fnc_counts)

    patient_label = parameters[constants.PARAM_KEY_PATIENT_LABEL]  # label1
    control_label = parameters[constants.PARAM_KEY_CONTROL_LABEL]
    label_def = constants.get_label_definitions(
        patient_label=patient_label, control_label=control_label
    )

    domain_names = parameters.get(
        constants.PARAM_KEY_FNC_DOMAIN_NAMES, constants.DEFAULT_FNCDomainNames
    )
    global_result_dir = os.path.join(output_dir, "global_results")
    os.makedirs(global_result_dir, exist_ok=True)
    for label, matrix in global_avg_fnc_relabeled.items():
        label_name = label_def[label]["name"]
        render_fnc_heatmap(
            matrix,
            global_result_dir,
            f"global_relabeled_avg_fnc_{label_name}.png",
            title=f"Average FNC of Relabeled {label_name} Subjects",
            colorbar_name="Avg FNC Values",
            domain_names=domain_names,
        )

    return GlobalReportData(
        global_avg_fnc_original=state["global_avg_fnc_original"],
        global_avg_fnc_relabeled=global_avg_fnc_relabeled,
        adaptive_threshold=state["adaptive_threshold"],
    )


def check_matching_fnc_shapes(site_results: Dict[str, CrfLocalResult]):
    """Fail clearly when sites send FNC matrices of different sizes (different atlases)."""
    shapes = {}
    for name in sorted(site_results):
        sums = site_results[name].fnc_sum
        if sums:
            shapes[name] = np.asarray(next(iter(sums.values()))).shape
    if len(set(shapes.values())) > 1:
        detail = ", ".join(
            f"{name}: {'x'.join(map(str, shape))}" for name, shape in shapes.items()
        )
        raise ValueError(
            "Sites sent FNC matrices of different sizes, so they did not use the same atlas "
            f"({detail}). Every site must use the atlas set in the computation parameters."
        )


def global_mean_from_sites(fnc_sums, fnc_counts):
    """Combine per-site Fisher-z sum/count FNC packets into one federated average, per label."""
    site_names = list(fnc_sums.keys())
    labels = list(fnc_sums[site_names[0]].keys()) if site_names else []

    result = {}
    for label in labels:
        aggregated_sum = sum(fnc_sums[name][label] for name in site_names)
        aggregated_total = sum(fnc_counts[name][label] for name in site_names)

        avg_fnc = np.divide(
            aggregated_sum,
            aggregated_total,
            out=np.zeros_like(np.asarray(aggregated_sum, dtype=float)),
            where=np.asarray(aggregated_total) > 0,
        )
        inverse_avg_fnc = np.tanh(avg_fnc)
        inverse_avg_fnc = (inverse_avg_fnc + inverse_avg_fnc.T) / 2.0
        np.fill_diagonal(inverse_avg_fnc, 1.0)
        result[label] = inverse_avg_fnc

    return result
