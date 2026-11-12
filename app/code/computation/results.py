"""Build the final per-site HTML report (the site_output_step)."""

import os
import shutil

from framework import cache_file

from . import constants
from .halfpipe_inputs import STAGED_DIR_NAME
from .local_math import LOCAL_CACHE_DATA_FILE
from .reports import generate_report_html, render_fnc_heatmap
from .types import GlobalReportData


def remove_cached_site_data(cache_dir):
    """Delete this site's cached subject data once the last step no longer needs it.

    The framework also clears ``cache_dir`` when the run ends, but it skips that
    if a transfer is still open, and ``cache_dir`` sits in the site's mounted
    output folder, so the computation removes its own copies explicitly.
    """
    data_path = cache_file(LOCAL_CACHE_DATA_FILE, cache_dir=cache_dir)
    if os.path.isfile(data_path):
        os.remove(data_path)
    shutil.rmtree(os.path.join(cache_dir, STAGED_DIR_NAME), ignore_errors=True)


def build_site_report(
    global_report: GlobalReportData, *, state, parameters, output_dir, cache_dir
):
    """Render the global heatmaps and assemble this site's self-contained report."""
    domain_names = parameters.get("FNCDomainNames", constants.DEFAULT_FNCDomainNames)
    # validate labels
    patient_label = parameters.get(constants.PARAM_KEY_PATIENT_LABEL, None)
    control_label = parameters.get(constants.PARAM_KEY_CONTROL_LABEL, None)
    label_def = constants.get_label_definitions(
        patient_label=patient_label, control_label=control_label
    )

    for label, matrix in global_report.global_avg_fnc_original.items():
        label_name = label_def.get(label, {}).get("name", label)
        render_fnc_heatmap(
            matrix,
            output_dir,
            f"global_original_avg_fnc_{label_name}.png",
            title=f"Global Average FNC of Original {label_name} Subjects",
            colorbar_name="Avg FNC Values",
            domain_names=domain_names,
        )

    for label, matrix in global_report.global_avg_fnc_relabeled.items():
        label_name = label_def.get(label, {}).get("name", label)
        render_fnc_heatmap(
            matrix,
            output_dir,
            f"global_relabeled_avg_fnc_{label_name}.png",
            title=f"Global Average FNC of Relabeled {label_name} Subjects",
            colorbar_name="Avg FNC Values",
            domain_names=domain_names,
        )

    html = generate_report_html(
        state["my_name"],
        output_dir,
        state.get("scores_df"),
        parameters,
        global_report.adaptive_threshold,
    )
    remove_cached_site_data(cache_dir)
    return {"index.html": html}
