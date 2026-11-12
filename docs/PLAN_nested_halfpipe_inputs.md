# Plan: Reading nested HALFpipe derivatives in Fed LAMP (nfc-label-noise-filtering)

**Status:** implemented in `nfc-label-noise-filtering-halfpipe` (forked from `nfc-label-noise-filtering` at `4b53d0b`)

> **Decision (2026-11-12), resolving Q8:** a hybrid of §5 and §11. At the start of each run the computation converts the nested derivatives into the existing flat per-subject layout, inside the run's cache directory, and then runs the unchanged pipeline (`app/code/computation/halfpipe_inputs.py`). Sites provide `participants.csv` or `participants.tsv`. `qc.csv` is used when present; **without it, every run found is used** (this replaces the "error for multi-session subjects" fallback in §5.6). Not yet built: the QC template helper and the `exclude*.json` pre-fill (§5.5).

## 1. Summary

Sites will provide connectivity matrices straight from their own stand-alone HALFpipe runs: nested folders with multiple sessions and runs per subject. The computation currently only reads a flat folder with one file per subject.

The plan:

- **Search the whole site folder** for HALFpipe correlation matrices, whatever the top folder is called and however deeply they're nested.
- **Add a QC file** at each site (`qc.csv`) that says which subject, session and run combinations may be used. A helper generates it, pre-filled from HALFpipe's own QC ratings where they exist, so sites only make the decisions and don't type the list.
- **Combine data with one standard rule:** for each subject, Fisher-z average every run the QC file includes, across every session it includes.
- **Check strictly** so mistakes (unreviewed runs, mistyped IDs, missing files) stop the run with a clear list to fix, rather than being silently included or dropped.

Labels stay one row per subject. No per-site (local) parameters are needed, and existing flat inputs keep working.

> **Alternative worth deciding on first:** keep the computation reading a flat dataset and do the folder search, QC filtering and averaging in a separate export step at each site. §11 makes the case for it; Q8 asks the team to choose.

## 2. Background

### Current behaviour

`app/code/computation/validation.py` (`load_subject_fnc_files`) builds one exact filename per subject and looks for it only at the top level of the site folder:

```
<subject_id>_task-rest_feature-<corr>_atlas-<atlas>CombinedDseg_desc-correlation_matrix.tsv
```

So it can't read stand-alone HALFpipe output, which:

- is nested (`sub-*/[ses-*/]func/task-*/…`)
- contains `ses-` and `run-` entities in filenames
- uses atlas names without a `CombinedDseg` suffix, depending on the atlas
- sits under a top folder whose name varies (data is moved in and out of scratch space)

### Expected site layout

```
<site data folder>/
├── participants.csv                    subject_id, sex, age, diagnosis
└── halfpipe_derivatives/               (name may vary: data moves in and out of scratch space)
    └── sub-<ID>/                       <ID> = 01, 02, PHCPP1011968, …
        └── ses-<S>/                    optional; session names vary but always start with ses-
            └── func/
                ├── sub-<ID>_ses-<S>_task-rest_run-1_stat-tsnr_boldmap.nii.gz
                ├── sub-<ID>_ses-<S>_task-rest_run-2_stat-tsnr_boldmap.nii.gz
                └── task-rest/
                    ├── sub-<ID>_ses-<S>_task-rest_run-1_feature-connectivity_atlas-Schaefer200_desc-correlation_matrix.tsv
                    ├── sub-<ID>_ses-<S>_task-rest_run-1_feature-neuromark_atlas-NeuroMark1_desc-correlation_matrix.tsv
                    ├── sub-<ID>_ses-<S>_task-rest_run-2_feature-…_desc-correlation_matrix.tsv
                    └── … (covariance, timeseries, falff, reho: ignored)
```

Without sessions, the `ses-<S>/` level and the `_ses-<S>` filename entity are simply absent. Run numbers may be unpadded (`run-1`) or padded (`run-01`).

Only `*_desc-correlation_matrix.tsv` files are read. The atlas is chosen once for the whole analysis in NeuroFLAME's global (study) parameters (§5.7). The HALFpipe feature name (`connectivity`, `neuromark`, …) is chosen by each site when it sets up HALFpipe, so matching is by atlas.

## 3. Requirements (from the discussion)

| # | Requirement | Source |
|---|---|---|
| R1 | Handle nested folders, any top-folder name, multiple sessions and runs | Data owners |
| R2 | One atlas and feature for the whole analysis, the same at every site | Data owners |
| R3 | Use all usable runs per subject per visit; modern protocols split rest into several AP/PA runs and all are needed for reliable connectivity | Data owners |
| R4 | Exclude runs that fail QC; QC is manual and happens between HALFpipe and this analysis | Data owners |
| R5 | Never silently average visits collected far apart (longitudinal data) | Data owners |
| R6 | Average across sessions when a subject legitimately has more than one (for example P-HCP V1A/V1B) | Sandeep |
| R7 | One standard, simple rule all sites accept; the option with the least chance of unintended mistakes | Sandeep |
| R8 | Sites use stand-alone HALFpipe output, not nfc-halfpipe | Javier |

## 4. Decision: QC file, not removing failed runs

Two options were considered for marking which runs to use:

| Possible mistake | A. Sites remove failed runs from the folder | B. Sites provide a QC file (chosen) |
|---|---|---|
| Site forgets to do QC | Every run is used, nothing flags it | Unreviewed runs stop the run with a list |
| A failed run is left in by mistake | Silently included | Only included if explicitly marked `yes` |
| Derivatives re-copied from scratch, or HALFpipe rerun | Removed runs can come back without anyone noticing | The QC file stays the same |
| "What was excluded and why?" | No record; files may be deleted | The QC file is the record |
| QC decision changes | Restore and delete files again | Edit one cell |
| Mistyped subject ID | n/a | Reported, with the ID |

Option A is simpler to code but makes the folder contents the QC decision, and a folder can't show that a run was reviewed. Option B costs sites one file, and the helper in §5.5 reduces that to filling in one column.

## 5. Design

### 5.1 Site inputs

```
<site data folder>/
├── participants.csv    one row per subject: subject_id, sex, age, diagnosis (set as LabelFile)
├── qc.csv              NEW: one row per run, include yes/no (provided by each site)
└── <HALFpipe derivatives, any folder name, any depth>
```

`participants.csv` replaces `labels.csv` by configuration only (`"LabelFile": "participants.csv"`). Extra columns such as `sex` and `age` are ignored.

`subject_id` must be read as text, not as a number. The current code lets pandas read `01` as the integer `1`, which would then fail to match the `sub-01` folder. Subject IDs match with or without the `sub-` prefix, so `01` and `sub-01` both work.

### 5.2 QC file format

```csv
subject_id,session,run,include
sub-PHCPP1011968,ses-V1A,01,yes
sub-PHCPP1011968,ses-V1A,02,no
sub-PHCPP1011968,ses-V1B,01,yes
sub-PHCPP1011969,ses-V1A,01,yes
```

- `include` accepts `yes`/`no` (also `1`/`0`, `true`/`false`; case-insensitive).
- `session` and `run` may be blank for datasets whose filenames don't have those entities.
- Matching rules:
  - Subject IDs match case-insensitively, with or without the `sub-` prefix.
  - Sessions match with or without `ses-`.
  - Runs match as numbers, so `01`, `1` and `run-01` are the same run.
- The file is listed by name in the global parameters (`QCFile`), like `LabelFile`.

### 5.3 File discovery

- Search the site folder recursively for `*_desc-correlation_matrix.tsv`.
- Parse the BIDS entities from each filename: `sub`, `ses`, `task`, `run`, `feature`, `atlas`.
- Keep only files whose `atlas` (and `task`) match the global parameters exactly (case-insensitive). If the optional `feature` parameter is set, it must match too.
- If no `feature` is set and a run has correlation matrices for the chosen atlas under more than one feature name, the site stops with an error listing them. The study lead then sets `feature` to choose one. This keeps two different matrices from being mixed silently.
- Everything else is ignored: other atlases, timeseries, covariance, fMRIPrep outputs, reports, and stray copies such as `… copy.tsv`.

### 5.4 Combining runs and sessions (the standard rule)

For each subject in `labels.csv`:

1. Collect every matrix the QC file marks `include = yes`, across all included sessions.
2. Fisher-z average them: `tanh(mean(arctanh(r)))`, with `r` clipped to ±0.999999 before `arctanh` and the diagonal set to 1.
3. Average each ROI pair over the runs where it's present. A pair missing in every run stays missing, and the existing `IgnoreSubjectsWithMissingData` setting decides whether that subject is dropped or the run fails.
4. Use the upper triangle as the subject's feature vector, as today.

Each site decides which visits count by what it marks in `qc.csv`:

- To average a split visit (for example P-HCP V1A and V1B), mark both sessions `yes`.
- For longitudinal data, mark only the visit you want.

The site log lists every subject that ended up with more than one session, so averaging across visits is always visible (R5, R6).

### 5.5 QC template helper (generating `qc.csv`)

A script shipped in the computation's Docker image, so sites need nothing extra:

```bash
docker run --rm -v /path/to/site/data:/data coinstacteam/nfc-label-noise-filtering \
  python /workspace/scripts/make_qc_template.py /data
```

- It writes `qc.csv` with one row for every subject, session and run found, using the same discovery rules as §5.3. `include` is left blank.
- If HALFpipe QC-viewer ratings (`exclude*.json`) are present, it pre-fills `include`:

  | HALFpipe rating | `include` |
  |---|---|
  | bad | `no` |
  | good | `yes` |
  | uncertain or unrated | blank, for a person to decide |

- **Backup:** if a run starts with no QC file, the computation stops and writes the same template into that site's output folder, with instructions.

### 5.6 Validation (stops before any federated step)

With a QC file present:

| Check | Result |
|---|---|
| A run on disk for a labeled subject has no QC row, or `include` is blank or invalid | **Error**, listing each run |
| A QC row marked `yes` has no matching file | **Error**, listing each row (catches typos and incomplete copies) |
| Duplicate QC rows for the same subject/session/run | **Error** |
| A labeled subject ends up with zero included runs | Missing subject; handled by `IgnoreSubjectsWithMissingData` |
| A QC subject not in `labels.csv` | Warning only |
| Matrices within a site have different sizes | **Error** |
| Sites send different feature counts (different atlas) | **Error** at the aggregator, naming the sites (R2) |

Without a QC file (backward compatibility, §5.8):

- Subjects with one session use all their runs, with a warning.
- Subjects with several sessions cause an error asking for a QC file.

Each site writes a local `qc_summary.csv`: per subject, the sessions and runs used and excluded, and the matrix size. It stays at the site; only the existing aggregate results go to the server.

### 5.7 Parameters

Global parameters, set once by the study lead in NeuroFLAME and the same for every site:

```json
{
  "LabelFile": "participants.csv",
  "QCFile": "qc.csv",
  "ReadDataFileFromHalfpipeOutputs": {
    "atlas": "NeuroMark1",
    "task": "rest"
  },
  "ControlLabel": 2,
  "PatientLabel": 1,
  "IgnoreSubjectsWithMissingData": true
}
```

The other algorithm parameters are unchanged:

- `atlas` is required. It's the HALFpipe atlas name exactly as it appears in filenames (`NeuroMark1`, `Schaefer200`, …).
- `task` defaults to `rest`.
- `feature` is optional and only needed if a site has the same atlas under two feature names (§5.3). `corr` is still accepted as an alias for it.
- `atlas` matches the atlas name exactly or with `CombinedDseg` appended, so current settings like `"atlas": "schaefer2011"` keep working.
- `FNCDomainNames` must match the chosen atlas, for the heatmaps (see §9).
- No local parameters: the computation record keeps `hasLocalParameters: false`.

### 5.8 Backward compatibility

These keep working unchanged:

- `ExistingDataFile` (`data.csv`) inputs.
- Flat folders of TSVs (the existing `test_data_halfpipe_atlas`), since the recursive search includes the top level.

## 6. Implementation steps

| # | Step | Files (indicative) |
|---|---|---|
| 1 | Branch off `main` (`feature/nested-halfpipe-inputs`) | — |
| 2 | BIDS filename parser and recursive discovery | `app/code/computation/halfpipe_inputs.py` (new) |
| 3 | QC file loading, normalisation and the §5.6 checks | same |
| 4 | Fisher-z run/session averaging with NaN handling | same |
| 5 | Wire into validation; read `subject_id` as text so `01` stays `01`; keep the `data.csv` path; fix the group-count check, which currently never fires | `validation.py` |
| 6 | Cross-site feature-count check | `remote_math.py` |
| 7 | `qc_summary.csv` and the missing-QC template in the site output | `results.py` / `inputs.py` |
| 8 | QC template helper, including `exclude*.json` pre-fill | `scripts/make_qc_template.py`, `Dockerfile-prod` |
| 9 | Unit tests (§7) | `tests/test_halfpipe_inputs.py` |
| 10 | Nested test datasets with QC files | `test_data_halfpipe_nested_*` |
| 11 | Docs: README and display notes (inputs, QC file, helper, parameters) | `README.md`, `display_notes.md` |
| 12 | Version bump to `0.2.0` (new parameters) | `.neuroflame.json` |
| 13 | Review with Sunitha, then merge | PR |
| 14 | After approval: build and push the image, update the computation record's notes | `dockerPush.sh`, NeuroFLAME CLI |

## 7. Testing

**Unit tests** on generated folder trees:

- P-HCP shape: two sessions, `run-01`/`run-02`, a renamed top folder.
- No `ses-` level; no `run-` entity.
- QC excludes some runs; QC excludes a whole session.
- Unreviewed run; mistyped subject ID; QC row with no file; duplicate rows.
- A run with NaN ROIs covered by another run; an ROI missing in every run.
- Distractor files (other features and atlases, ` copy.tsv`, timeseries, covariance).
- Fisher-z average checked against a hand-computed value.
- The flat layout and `data.csv` still produce identical results to today.

**Integration:**

- A 3-site local simulation (`run_local_simulation.sh`) on the nested test datasets.
- Compare relabeled subjects against the datasets' `ground_truth.csv`.

**Real-data check:**

- A directory listing (filenames only, no data) from one real site, ideally P-HCP.
- Confirm discovery, the QC template and the subject-ID matching against it before release.

## 8. Rollout

1. Merge after review. Publish image `0.2.0` (also tagged `latest`) and the updated display notes.
2. Send sites a short guide:
   1. Point NeuroFLAME at the folder that contains `labels.csv`, `qc.csv` and the HALFpipe derivatives.
   2. Run the template helper.
   3. Fill in `include`.
   4. Start the run.
3. First real run: one site, then all sites.

## 9. Open questions

| # | Question | Who |
|---|---|---|
| Q1 | ~~Do real filenames start with `sub-`?~~ **Resolved:** the expected format uses `sub-<ID>` for both folders and filenames. | — |
| Q2 | **Decided:** the atlas is set in NeuroFLAME's global parameters. **Still open:** which atlas the first analysis uses (for example `NeuroMark1`, 53×53, or `Schaefer200`, 200×200), and the matching `FNCDomainNames`. The current `[0, 5, 7, 16, 25, 42, 49, 53]` matches NeuroMark1. Schaefer200 would need its own domain boundaries, and in earlier test output it had empty parcels (all-NaN ROIs) for every subject. | Study lead |
| Q7 | ~~QC file or folder-as-approved-set?~~ **Resolved:** each site provides a separate `qc.csv` next to `participants.csv`. | — |
| Q3 | Which sites rated their data in HALFpipe's QC viewer and saved `exclude*.json`? We use it to pre-fill `qc.csv` where it exists but don't assume it: HALFpipe only writes it when someone rates in the viewer, its derivatives are produced for every run regardless of ratings, and some sites use other QC (motion thresholds, MRIQC, visual checks). | Sites |
| Q4 | Are all sites OK with "average all included runs and sessions per subject" as the single standard rule? | Sandeep, Jessica |
| Q5 | Longitudinal labels: `labels.csv` stays one diagnosis per subject. Is a per-visit label ever needed? (Out of scope for this change.) | Data owners |
| Q6 | Sunitha wrote the current reader on Oct 27: review this change, or own parts of it? | Sunitha |
| Q8 | Read nested derivatives inside the computation (§5), or have sites export a flat dataset first (§11)? | Sandeep, Jessica, Sunitha |

## 10. Risks

| Risk | Mitigation |
|---|---|
| Sites find the QC file a burden | Template helper; pre-fill from HALFpipe ratings |
| Subject ID mismatches between `labels.csv`, `qc.csv` and filenames | Lenient matching (case, `sub-`/`ses-` prefixes) plus explicit error lists |
| Different atlases across sites | Single global parameter plus the aggregator size check |
| Very large atlases (for example 434 ROIs, about 94k features) slow the forest step at large sites | Measure in the integration test; document expected run times |
| Changes to the existing reader conflict with ongoing work | Review with Sunitha before merging; keep backward compatibility |

## 11. Alternative: export a flat dataset at each site before the run

Everything in §5 makes the federated computation responsible for understanding HALFpipe's folder layout, QC decisions and run averaging. An alternative is to keep the computation reading a **flat dataset**, which it already supports, and move that work into a separate **export step** that each site runs once QC is finished.

### 11.1 What it looks like

```
HALFpipe derivatives + participants.csv + qc.csv
        │
        ▼   export step (script shipped in the computation image; same rules as §5.3–5.6)
        │
<site data folder for NeuroFLAME>/
├── labels.csv        subject_id, diagnosis                    (one row per included subject)
├── data.csv          one row per subject: averaged upper-triangle FNC, same order as labels.csv
└── manifest.json     atlas, feature, task, export-script version, QC file checksum,
                      and per subject the exact source files and runs that were averaged
```

```bash
docker run --rm -v /path/to/site:/data coinstacteam/nfc-label-noise-filtering \
  python /workspace/scripts/export_flat_dataset.py /data/halfpipe_derivatives \
    --participants /data/participants.csv --qc /data/qc.csv --atlas NeuroMark1 \
    --out /data/fedlamp_input
```

The standard rule (Fisher-z average of every included run and session) and the strict QC checks are exactly the same as in §5. They just run before the federated analysis instead of inside it. NeuroFLAME is then pointed at `fedlamp_input/`.

### 11.2 Why this is attractive

| | Nested reading inside the computation (§5) | Flat export first (§11) |
|---|---|---|
| **Changes to the federated computation** | New reader, QC logic, averaging, and a new version for sites to pull | Almost none: `data.csv` + `labels.csv` already work in the published `0.1.0` image. Only an optional manifest check is new (§11.3) |
| **When mistakes surface** | During a consortium run, after every site has joined and started | At the site, on its own, before anyone schedules a run. A failed export costs nobody else time |
| **Seeing exactly what goes in** | Has to be reconstructed from the site log and `qc_summary.csv` | The site can open `data.csv`/`labels.csv` and the manifest before the run |
| **Data the container sees** | The whole derivatives tree (NIfTIs, timeseries, reports), often tens to hundreds of GB | Only the averaged matrices and labels: a few MB for NeuroMark1 |
| **Changes to HALFpipe's layout or naming** | Every change needs a new computation release | Only the export script changes; the federated computation is untouched |
| **Reuse by other computations** | Each FNC computation (for example nfc-dpsvm, brain-age FNC) would need its own nested reader, QC and averaging, which could drift apart | One export, one rule, one QC record, usable by any computation that reads per-subject FNC vectors |
| **Testing** | Must be tested through federated simulations | A plain script, testable on its own with fixtures and real directory listings |
| **Code ownership** | Rewrites the reader Sunitha added on Oct 27 | Leaves her reader in place |

The main argument is that **the hard part here is not federated**. Finding files, applying QC and averaging runs all depend only on one site's own data. Keeping that out of the federated workflow makes both pieces simpler. The computation does the federated math on a clean, standard input, and the site-specific mess is handled where it can be seen and fixed.

### 11.3 Costs and how to handle them

| Cost | Mitigation |
|---|---|
| One more step for sites | QC is already a manual step between HALFpipe and the analysis. The export runs right after it, with the same template helper (§5.5) feeding `qc.csv` |
| Exports can go stale (QC edited, export not rerun) | `manifest.json` records the QC file checksum. The export can be rerun at any time, and it overwrites its own output folder |
| Sites exporting different atlases, or with different versions of the script | The computation reads `manifest.json` and stops if the atlas does not match the global parameter, or if the export-script version is older than required. This is the only computation change needed |
| `data.csv` relies on row order to match `labels.csv` | The export writes both files together from the same list, and the manifest maps each row to its subject ID |
| Two things to version (export script and computation) | Ship both in the same image and version them together |

### 11.4 Recommendation

Use the flat export, unless the team specifically wants a single-step workflow inside NeuroFLAME. It gives the same standard rule and the same strict QC. It's less code in the federated computation, puts errors in front of the site that can fix them, and can be reused by other FNC computations.

If it's chosen, §6 shrinks to:

1. The export script, built from §5.3–5.6.
2. The template helper.
3. The manifest check in the computation.
4. Tests and docs.

The nested reader inside the computation (steps 2–7 of §6) is no longer needed.
