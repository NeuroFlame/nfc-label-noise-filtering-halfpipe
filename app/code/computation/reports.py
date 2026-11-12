"""FNC heatmap rendering and the per-site HTML report."""

import os
import re as _re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import constants


def render_fnc_heatmap(
    fnc_matrix,
    output_dir,
    name,
    *,
    title="Heat Map",
    colorbar_name="",
    domain_names=None,
):
    """Render one symmetric FNC matrix as a PNG heatmap with a symmetric color range."""
    domain_breaks = domain_names or []

    max_value = np.nanmax(np.abs(fnc_matrix))
    max_value = max_value if np.isfinite(max_value) and max_value > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    im = ax.imshow(
        fnc_matrix, cmap="RdYlBu_r", vmin=-max_value, vmax=max_value, origin="upper"
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_name, rotation=270, labelpad=10)

    for g in domain_breaks:
        ax.axhline(g - 0.5, color="k", lw=0.6)
        ax.axvline(g - 0.5, color="k", lw=0.6)

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _img_tag(path, alt=""):
    if not os.path.exists(path):
        return (
            f'<div style="color:var(--text3);font-size:.8rem;padding:1rem;'
            f'border:1px dashed var(--border);border-radius:8px">{alt} (not generated)</div>'
        )
    filename = os.path.basename(path)
    return (
        f'<img src="{filename}" alt="{alt}" '
        f'style="max-width:100%;border-radius:8px;border:1px solid var(--border)"/>'
    )


def generate_report_html(
    site_name, output_dir, scores_df, parameters, adaptive_threshold
):
    """Build the self-contained per-site HTML report."""
    patient_label = parameters[constants.PARAM_KEY_PATIENT_LABEL]  # label1
    control_label = parameters[constants.PARAM_KEY_CONTROL_LABEL]
    label_def = constants.get_label_definitions(
        patient_label=patient_label, control_label=control_label
    )

    label_names = {str(k): v.get("name", str(k)) for k, v in label_def.items()}

    body = (
        _build_header(site_name, parameters)
        + _build_distribution_section(scores_df, label_def, adaptive_threshold)
        + _build_visualizations_section(output_dir, label_names, parameters)
        + _build_global_fnc_section(output_dir, label_names)
        + _build_scores_table_section(scores_df)
    )
    return _wrap_page(
        body,
        [
            "Site Overview",
            "Label Distribution",
            "Site Visualizations",
            "Global FNC Heatmaps",
            "Subject Scores",
        ],
        site_name,
    )


def _build_header(site_name, parameters):
    patient_label = parameters[constants.PARAM_KEY_PATIENT_LABEL]  # label1
    control_label = parameters[constants.PARAM_KEY_CONTROL_LABEL]
    label_def = constants.get_label_definitions(
        patient_label=patient_label, control_label=control_label
    )

    n_labels = len(label_def)
    n_iter = parameters.get("Iteration", "—")
    truncation = parameters.get("TruncationParameter", "—")
    label_pills = "".join(
        f'<span style="background:var(--chip-bg);border:1px solid var(--border);border-radius:999px;'
        f'padding:.2rem .65rem;font-size:.76rem;color:var(--chip-color)">{v.get("name", k)}</span>'
        for k, v in sorted(label_def.items())
    )
    return f"""<div class="page-header">
  <h1>Label Noise Filtering — {site_name}</h1>
  <p>Per-site results for federated CRF-based label noise filtering with adaptive thresholding</p>
  <div class="chips">
    <div class="chip">Site <b>{site_name}</b></div>
    <div class="chip">Labels <b>{n_labels}</b></div>
    <div class="chip">CRF Iterations <b>{n_iter}</b></div>
    <div class="chip">Truncation <b>{truncation}</b></div>
  </div>
  <div style="margin-top:.9rem;display:flex;flex-wrap:wrap;gap:.4rem">{label_pills}</div>
</div>"""


def _build_distribution_section(scores_df, label_def, adaptive_threshold):
    re_labels = (
        scores_df.get("re_labeled", pd.Series(dtype=float))
        if scores_df is not None
        else pd.Series(dtype=float)
    )
    total = len(re_labels)
    uncertain = (
        int((re_labels == constants.BOUNDARY_SUBJECT_LABEL_VALUE).sum())
        if total > 0
        else 0
    )

    rows = ""
    for k, v in sorted(label_def.items()):
        lbl_val = v.get("value")
        lbl_name = v.get("name", k)
        count = int((re_labels.round() == lbl_val).sum()) if total > 0 else 0
        pct = f"{100 * count / total:.1f}%" if total > 0 else "—"
        bar_w = f"{100 * count / total:.1f}" if total > 0 else "0"
        rows += (
            f'<tr><td style="font-weight:600">{lbl_name}</td>'
            f'<td style="font-family:monospace">{lbl_val}</td>'
            f'<td style="font-family:monospace">{count}</td>'
            f'<td style="font-family:monospace">{pct}</td>'
            f'<td style="padding:.42rem .85rem;min-width:120px">'
            f'<div style="background:var(--bg3);border-radius:999px;height:8px">'
            f'<div style="width:{bar_w}%;background:#6366f1;height:8px;border-radius:999px"></div>'
            f"</div></td></tr>"
        )
    unc_pct = f"{100 * uncertain / total:.1f}%" if total > 0 else "—"
    rows += (
        f'<tr style="color:var(--text3)"><td style="font-weight:600">Uncertain</td>'
        f'<td style="font-family:monospace">−1</td>'
        f'<td style="font-family:monospace">{uncertain}</td>'
        f'<td style="font-family:monospace">{unc_pct}</td><td></td></tr>'
    )

    thresh_fmt = (
        f"{adaptive_threshold:.4f}"
        if isinstance(adaptive_threshold, float)
        else str(adaptive_threshold)
    )

    content = f"""
<div style="display:flex;gap:.75rem;margin-bottom:1.2rem;flex-wrap:wrap">
  <div class="kpi-card">
    <div class="kpi-label">Total Subjects</div>
    <div class="kpi-value">{total}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Adaptive Threshold (&plusmn;t)</div>
    <div class="kpi-value" style="color:#6366f1">{thresh_fmt}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Uncertain / Unclassified</div>
    <div class="kpi-value" style="color:#d97706">{uncertain}</div>
  </div>
</div>
<div class="stat-card">
  <div class="stat-card-header"><div class="stat-card-title">Relabeled Subject Distribution</div></div>
  <div class="stat-card-scroll">
    <table class="stat-table">
      <thead><tr>
        <th style="text-align:left">Label</th><th>Value</th><th>Count</th><th>%</th>
        <th style="min-width:120px">Distribution</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""
    return _section("Label Distribution", content, "label-distribution")


def _build_global_fnc_section(output_dir, label_names):
    orig_imgs = "".join(
        f'<div><div style="font-size:.8rem;font-weight:600;margin-bottom:.5rem;color:var(--text2)">{name}</div>'
        f"{_img_tag(os.path.join(output_dir, f'global_original_avg_fnc_{name}.png'), f'Global original avg FNC – {name}')}</div>"
        for name in label_names.values()
    )
    relab_imgs = "".join(
        f'<div><div style="font-size:.8rem;font-weight:600;margin-bottom:.5rem;color:var(--text2)">{name}</div>'
        f"{_img_tag(os.path.join(output_dir, f'global_relabeled_avg_fnc_{name}.png'), f'Global relabeled avg FNC – {name}')}</div>"
        for name in label_names.values()
    )
    content = f"""
<div style="margin-bottom:1.5rem">
  <div class="subsection-title">Original Labels — Federated Average FNC</div>
  <div class="img-grid">{orig_imgs}</div>
</div>
<div>
  <div class="subsection-title">After Relabeling — Federated Average FNC</div>
  <div class="img-grid">{relab_imgs}</div>
</div>"""
    return _section("Global FNC Heatmaps", content, "global-fnc-heatmaps")


def _build_visualizations_section(output_dir, label_names, params):
    orig_ttest = _img_tag(
        os.path.join(output_dir, "original_labels_ttest.png"), "Original labels T-test"
    )
    relab_ttest = _img_tag(
        os.path.join(output_dir, "re_labeled_ttest.png"), "Relabeled T-test"
    )

    orig_fnc = "".join(
        f'<div><div style="font-size:.8rem;font-weight:600;margin-bottom:.5rem;color:var(--text2)">{name}</div>'
        f"{_img_tag(os.path.join(output_dir, f'local_original_avg_fnc_{name}.png'), f'Original avg FNC – {name}')}</div>"
        for name in label_names.values()
    )
    relab_fnc = "".join(
        f'<div><div style="font-size:.8rem;font-weight:600;margin-bottom:.5rem;color:var(--text2)">{name}</div>'
        f"{_img_tag(os.path.join(output_dir, f'local_relabeled_avg_fnc_{name}.png'), f'Relabeled avg FNC – {name}')}</div>"
        for name in label_names.values()
    )

    content = f"""
<div style="margin-bottom:1.5rem">
  <div class="subsection-title">T-test Heatmaps (Bonferroni-corrected)</div>
  <div class="img-grid">
    <div><div style="font-size:.8rem;font-weight:600;margin-bottom:.5rem;color:var(--text2)">Original Labels</div>{orig_ttest}</div>
    <div><div style="font-size:.8rem;font-weight:600;margin-bottom:.5rem;color:var(--text2)">After Relabeling</div>{relab_ttest}</div>
  </div>
</div>
<div style="margin-bottom:1.5rem">
  <div class="subsection-title">Average FNC — Original Labels</div>
  <div class="img-grid">{orig_fnc}</div>
</div>
<div>
  <div class="subsection-title">Average FNC — After Relabeling</div>
  <div class="img-grid">{relab_fnc}</div>
</div>"""
    return _section("Site Visualizations", content, "site-visualizations")


def _build_scores_table_section(scores_df):
    if scores_df is None or scores_df.empty:
        return _section(
            "Subject Scores",
            '<p style="color:var(--text3)">No scores available.</p>',
            "subject-scores",
        )

    preview = scores_df.head(100)
    cols = list(preview.columns)
    header = "".join(f"<th>{c}</th>" for c in cols)

    rows = ""
    for _, row in preview.iterrows():
        cells = ""
        for c in cols:
            val = row[c]
            if c == "re_labeled" and isinstance(val, float):
                style = (
                    "color:#d97706"
                    if val == constants.BOUNDARY_SUBJECT_LABEL_VALUE
                    else ""
                )
                display = (
                    "?"
                    if val == constants.BOUNDARY_SUBJECT_LABEL_VALUE
                    else str(int(val))
                )
                cells += f'<td style="font-family:monospace;{style}">{display}</td>'
            elif isinstance(val, float):
                cells += f'<td style="font-family:monospace">{val:.4f}</td>'
            else:
                cells += f'<td style="font-family:monospace">{val}</td>'
        rows += f"<tr>{cells}</tr>"

    note = (
        f'<div style="padding:.5rem 1rem;font-size:.72rem;color:var(--text3)">'
        f"Showing first {min(100, len(scores_df))} of {len(scores_df)} subjects. "
        f"re_labeled = ? means score fell within the uncertain zone (&plusmn;t).</div>"
        if len(scores_df) > 100
        else ""
    )

    content = f"""
<div class="stat-card">
  <div class="stat-card-header">
    <div class="stat-card-title">Dimensional Scores &amp; Re-labels</div>
  </div>
  <div class="stat-card-scroll">
    <table class="stat-table">
      <thead><tr>{header}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  {note}
</div>"""
    return _section("Subject Scores", content, "subject-scores")


def _section(title, content, slug=None):
    if slug is None:
        slug = _re.sub(r"-+", "-", _re.sub(r"[^a-z0-9]+", "-", title.lower())).strip(
            "-"
        )
    return (
        f'<div class="container"><div class="section" id="sec-{slug}">'
        f'<div class="section-title">{title}</div>{content}</div></div>'
    )


def _wrap_page(body, nav_titles, site_name=""):
    def _slug(t):
        return _re.sub(r"-+", "-", _re.sub(r"[^a-z0-9]+", "-", t.lower())).strip("-")

    nav_items = "\n".join(
        f'<button class="nav-item" data-sec="sec-{_slug(t)}" '
        f"onclick=\"document.getElementById('sec-{_slug(t)}').scrollIntoView({{behavior:'smooth',block:'start'}})\">{t}</button>"
        for t in nav_titles
    )
    sidebar_and_body = (
        '<div class="layout">'
        '<aside class="sidebar" id="sidebar"><div class="sidebar-inner">'
        '<div class="sidebar-header">'
        '<span class="sidebar-label">Sections</span>'
        '<button class="sidebar-toggle" onclick="toggleSidebar()">&#x2715;</button>'
        "</div>" + nav_items + "</div></aside>"
        '<button class="sidebar-peek" id="sidebarPeek" onclick="toggleSidebar()">Sections</button>'
        '<div class="main-content">' + body + "</div></div>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Label Noise Filtering — {site_name}</title>
<style>
:root {{
  --bg:#ffffff;--bg3:#f1f5f9;--border:#e2e8f0;
  --text:#0f172a;--text2:#334155;--text3:#64748b;
  --header-bg:linear-gradient(135deg,#e0e9ff 0%,#f8fafc 100%);
  --header-border:#e2e8f0;
  --chip-bg:#f1f5f9;--chip-color:#475569;--chip-b:#6366f1;
  --card-bg:#ffffff;--th-bg:#f8fafc;--td-mono:#1e293b;
}}
[data-theme="dark"] {{
  --bg:#0f172a;--bg3:#0f172a;--border:#334155;
  --text:#e2e8f0;--text2:#cbd5e1;--text3:#64748b;
  --header-bg:linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%);
  --header-border:#334155;
  --chip-bg:#1e293b;--chip-color:#94a3b8;--chip-b:#a5b4fc;
  --card-bg:#1e293b;--th-bg:#161f30;--td-mono:#cbd5e1;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;transition:background .2s,color .2s;margin:1rem 0}}
button{{font-family:inherit}}
.theme-toggle{{position:fixed;top:2rem;right:1.25rem;z-index:999;background:var(--card-bg);border:1px solid var(--border);border-radius:999px;padding:.35rem .85rem;font-size:.8rem;font-weight:600;color:var(--text2);cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.theme-toggle:hover{{background:var(--bg3)}}
.page-header{{background:var(--header-bg);border-bottom:1px solid var(--header-border);padding:2rem 2.5rem;padding-right:8rem}}
.page-header h1{{font-size:1.7rem;font-weight:700;color:var(--text);letter-spacing:-.02em}}
.page-header p{{color:var(--text3);margin-top:.35rem;font-size:.93rem}}
.chips{{display:flex;gap:.6rem;margin-top:1rem;flex-wrap:wrap}}
.chip{{background:var(--chip-bg);border:1px solid var(--border);border-radius:999px;padding:.25rem .75rem;font-size:.78rem;color:var(--chip-color)}}
.chip b{{color:var(--chip-b)}}
.container{{max-width:1400px;margin:0 auto;padding:2rem 0}}
.section{{margin-bottom:3rem}}
.section-title{{font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:1.1rem;padding-bottom:.5rem;border-bottom:1px solid var(--border)}}
.subsection-title{{font-size:.85rem;font-weight:700;color:var(--text2);margin-bottom:.75rem}}
.img-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1rem}}
.kpi-card{{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:.9rem 1.2rem;flex:1;min-width:140px}}
.kpi-label{{font-size:.72rem;color:var(--text3);text-transform:uppercase;letter-spacing:.07em}}
.kpi-value{{font-size:1.6rem;font-weight:700;color:var(--text);font-family:monospace}}
.stat-card{{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;overflow:hidden}}
.stat-card-scroll{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
.stat-card-header{{padding:.75rem 1rem;border-bottom:1px solid var(--border);display:flex;align-items:center}}
.stat-card-title{{font-weight:700;color:var(--text);font-size:.9rem;flex:1}}
table.stat-table{{width:100%;border-collapse:collapse;font-size:.8rem}}
table.stat-table th{{color:var(--text3);font-weight:600;padding:.45rem .85rem;text-align:right;background:var(--th-bg)}}
table.stat-table th:first-child{{text-align:left}}
table.stat-table td{{padding:.42rem .85rem;border-top:1px solid var(--border);text-align:right}}
table.stat-table td:first-child{{text-align:left}}
.layout{{display:flex;align-items:flex-start;padding:1.5rem 2rem 0}}
.sidebar{{width:190px;flex-shrink:0;position:sticky;top:1.5rem;max-height:calc(100vh - 3rem);overflow-y:auto;margin-right:1.5rem;transition:width .2s,opacity .2s,margin .2s}}
.sidebar.hidden{{width:0;opacity:0;overflow:hidden;margin-right:0;pointer-events:none}}
.sidebar-inner{{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:.6rem .5rem}}
.sidebar-header{{display:flex;align-items:center;justify-content:space-between;padding:.2rem .3rem .45rem;border-bottom:1px solid var(--border);margin-bottom:.4rem}}
.sidebar-label{{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text3)}}
.sidebar-toggle{{background:none;border:none;cursor:pointer;font-size:.75rem;color:var(--text3);padding:.15rem .35rem;border-radius:5px}}
.sidebar-toggle:hover{{background:var(--bg3);color:var(--text)}}
.nav-item{{display:block;width:100%;padding:.4rem .65rem;border-radius:7px;font-size:.78rem;color:var(--text3);cursor:pointer;border:none;background:none;text-align:left;transition:background .12s,color .12s}}
.nav-item:hover{{background:var(--bg3);color:var(--text)}}
.nav-item.active{{background:rgba(99,102,241,.13);color:#6366f1;font-weight:600}}
[data-theme="dark"] .nav-item.active{{background:rgba(165,180,252,.1);color:#a5b4fc}}
.main-content{{flex:1;min-width:0}}
.sidebar-peek{{position:fixed;left:0;top:50%;transform:translateY(-50%);background:var(--card-bg);border:1px solid var(--border);border-left:none;border-radius:0 8px 8px 0;padding:.55rem .35rem;cursor:pointer;font-size:.72rem;color:var(--text3);display:none;z-index:200;writing-mode:vertical-rl}}
.sidebar-peek.visible{{display:block}}
</style>
</head>
<body>
<button class="theme-toggle" onclick="toggleTheme()" id="themeBtn">&#127769; Dark mode</button>
{sidebar_and_body}
<script>
function getTheme(){{try{{return localStorage.getItem('theme')||'light'}}catch(e){{return'light'}}}}
function applyTheme(t){{
  document.documentElement.setAttribute('data-theme',t==='dark'?'dark':'');
  document.getElementById('themeBtn').textContent=t==='dark'?'☀️ Light mode':'🌙 Dark mode';
}}
function toggleTheme(){{var n=getTheme()==='dark'?'light':'dark';try{{localStorage.setItem('theme',n)}}catch(e){{}}applyTheme(n);}}
function toggleSidebar(){{
  var sb=document.getElementById('sidebar'),pk=document.getElementById('sidebarPeek');
  var h=sb.classList.toggle('hidden');
  try{{localStorage.setItem('sidebarHidden',h?'1':'0')}}catch(e){{}}
  if(pk)pk.classList.toggle('visible',h);
}}
(function(){{try{{
  if(localStorage.getItem('sidebarHidden')==='1'){{
    var sb=document.getElementById('sidebar'),pk=document.getElementById('sidebarPeek');
    if(sb)sb.classList.add('hidden');if(pk)pk.classList.add('visible');
  }}
}}catch(e){{}}}}());
document.addEventListener('DOMContentLoaded',function(){{
  applyTheme(getTheme());
  var items=document.querySelectorAll('.nav-item[data-sec]');
  if(!items.length)return;
  var secs=Array.from(items).map(function(el){{return document.getElementById(el.getAttribute('data-sec'));}}).filter(Boolean);
  function onScroll(){{
    var y=window.scrollY+140,active=secs[0];
    secs.forEach(function(s){{if(s.offsetTop<=y)active=s;}});
    items.forEach(function(el){{el.classList.toggle('active',active&&el.getAttribute('data-sec')===active.id);}});
  }}
  window.addEventListener('scroll',onScroll,{{passive:true}});onScroll();
}});
</script>
</body>
</html>"""
