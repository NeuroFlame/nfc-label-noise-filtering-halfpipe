# Migration to computation-nvflare-boilerplate

This computation was migrated from the old, hand-written NVFlare
`Controller`/`Executor`/`Aggregator` architecture to the current
[`computation-nvflare-boilerplate`](https://github.com/NeuroFlame/computation-nvflare-boilerplate)
contract, where authors write only `app/code/computation/` and the
boilerplate's `framework/`/`runtime/` own all NVFlare integration. See
[computation_development.md's migration
guide](https://github.com/NeuroFlame/computation-nvflare-boilerplate/blob/main/docs/computation_development/migrating_computations.md)
for the general migration process, and
[nfc-multi-round-regression-freesurfer](https://github.com/NeuroFlame/nfc-multi-round-regression-freesurfer)
for a sibling computation migrated the same way.

## Fork: reading HALFpipe derivatives

This repository (`nfc-label-noise-filtering-halfpipe`) was forked from
[`nfc-label-noise-filtering`](https://github.com/NeuroFlame/nfc-label-noise-filtering)
at `78191cd`, keeping its history. The only functional addition is input
handling: when a site's folder holds nested HALFpipe derivatives, `load_inputs`
first converts them into the flat per-subject layout the original reader uses
(`app/code/computation/halfpipe_inputs.py`), and the rest of the workflow is
unchanged. The design, the decisions behind it and the remaining open questions
are in [docs/PLAN_nested_halfpipe_inputs.md](docs/PLAN_nested_halfpipe_inputs.md).

## Why `stepped_workflow`, not `iterative_workflow`

The old controller ran a fixed 4-round sequence with no convergence loop:
local CRF → aggregate centroids → dimensional scoring → adaptive threshold →
relabeling → aggregate relabeled metrics → local report. That maps directly
onto three `local_step`/`remote_step` pairs plus one final
`site_output_step`, so this computation uses `stepped_workflow`, unlike
`nfc-multi-round-regression-freesurfer`'s single phase-tagged
`iterative_workflow` (which that computation needed because one of its
phases is an open-ended gradient-descent loop).

## Site identity without `FLContextKey.CLIENT_NAME`

Two things needed this site's own display name, which the new framework
doesn't expose to author code:

1. **Dimensional scoring** (`local_math.compute_dimensional_scores`) needs
   to recognize which entry in the aggregator's site-keyed centroid map is
   its own, so it can use its own original labels there instead of scoring
   itself against its own centroids.
2. **The HTML report's title/header** names the site.

Same fix as the freesurfer migration: `load_inputs` mints a random
`self_token`, kept in local state; `aggregate_centroids` returns a
`token_to_name` map built from its already-display-name-keyed
`site_results`; each site resolves `my_name = token_to_name[self_token]`
once (in `compute_dimensional_scores`) and keeps it in state for later
steps, including the final report.

## State and the run cache replace the old disk round-trips

The old code round-tripped intermediate results through disk between tasks
(`{site}_orig.mat` written then read back via `data_loaders.load_result_matfile`,
`{site}_relabeled.csv` written then re-read). `data_loaders.py` (whose only
callers were those two re-reads) wasn't ported.

Small values (the site's `self_token`, its resolved name, its scores) live in
the framework's local state. The validated subject-by-feature data array is
cached as a file instead, using the run-scoped `cache_dir` the framework
injects (computation-nvflare-boilerplate PR #13): `compute_local_crf` saves it
with `cache_file("local_cache_data_file.npy", cache_dir=cache_dir)`, and
`compute_dimensional_scores` and `relabel_subjects` load it from the same
path. `cache_dir` is separate for each site and each run
(`<output_dir>/.artifacts/cache`) and is cleared when the run ends.

`build_site_report`, the last site step, also deletes the cached data file and
the converted HALFpipe inputs itself (`results.remove_cached_site_data`). The
framework's end-of-run cleanup is skipped when an artifact transfer is still
open, and `cache_dir` sits in the site's mounted output folder, so the
computation does not rely on it alone.


## Output filenames no longer include the site name

`{site}_orig.mat`, `{site}_CRF.mat`, and `{site}_relabeled.csv` are now
`orig.mat`, `crf.mat`, and `relabeled.csv`. Each site already gets its own
private output directory from the runtime, so the prefix was always
redundant — `centers.npz` never had one. This also follows from the site
identity change above: the very first local step (`compute_local_crf`) runs
before any remote round-trip, so it has no site name available yet to prefix
with.

## Other intentional differences

- **`FNCDomainNames` heatmap gridlines**: the old `relabel_data`/
  `print_relabeled_metrics` heatmap calls hardcoded
  `domain_names=[0, 5, 7, 16, 25, 42, 49, 53]` instead of reading the
  `FNCDomainNames` computation parameter like every other heatmap call site
  did. Fixed to read the parameter consistently. Invisible on this
  computation's test data, whose `FNCDomainNames` value is exactly that
  list — see Verification below.
- Dropped dead code identified while porting: `utils/html_templates.py`
  (templates for an unrelated computation), `Models/RelableModel.py`,
  `Models/MatfileModel.py`, `utils/exceptions.py` (`ValidationException`,
  never raised), `get_label_map`/the cached `label_map` value in input
  validation (computed but never read downstream), and the commented-out
  `.mat`-loading code paths in the old `perform_local_crf`.
- The old per-task `NvFlareLogger`/`CacheSerialStore` plumbing is gone; the
  framework now injects a ready-to-use logger and owns its lifecycle
  entirely.
- `nvflare` bumped 2.4.0 → 2.8.0, base Python image 3.9 → 3.11, per the
  boilerplate's current pins. Added an explicit `joblib` requirement (the
  CRF forest construction imports it directly; it was previously only an
  incidental transitive dependency of `scikit-learn`).

## Verification performed

- **Numeric parity**: captured a baseline from the pre-migration
  implementation (2-site NVFlare simulation, `test_data/site1`+`site2`,
  default parameters) and compared it against the migrated implementation's
  output.
  - `{site}_relabeled.csv` vs. `relabeled.csv`: every score and re-assigned
    label cell matches exactly (0 of 628 / 1244 cells differ across the two
    sites), including the adaptive threshold's effect on relabeling.
  - The adaptive threshold displayed in `index.html` matches exactly
    (`0.0182`).
  - All local/global average-FNC heatmaps (16 of 20 images) are
    byte-identical. The two t-test heatmaps are byte-identical for one site
    and pixel-identical but for anti-aliasing-level differences on the
    other (max channel delta of 2/255 on 1.1% of pixels) — consistent with
    the matplotlib version bump (3.9.4 → 3.10.9) from the boilerplate's
    Python 3.11 base image, not a computation difference.
  - Site names resolved correctly in each site's own report
    (`site1`/`site2`).
- **`make check`**: lint, format, compile, and unit tests all pass.
- **Image validation**: `./dockerPush.sh --no-push` confirms the production
  image builds with all required OCI/NeuroFLAME labels.
- **`migrate_computation.py --check`**: reports 0 managed paths differing
  from the checked-out boilerplate.

**Not directly measured**: an actual local NeuroFLAME platform stack
(central API + edge sites), as opposed to the NVFlare simulator directly —
only the simulator flow was exercised.
