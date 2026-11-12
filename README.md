### Overview

This paper presents Fed LAMP, a label noise filtering based dimensional prediction method called LAMP to improve biomarker discovery and prediction for mental disorders using fMRI data. The method uses a complete random forest model to identify typical subjects whose clinical labels are consistent with brain functional connectivity patterns. These reliable subjects are then used to build a dimensional model that assigns continuous scores reflecting disease severity to unseen subjects. Experiments on multi site schizophrenia and autism datasets show improved group separability, more stable biomarkers, and better generalization than traditional label based approaches.

### Example Settings

```json
{
  "SamplingThreshold": 0.7,
  "Iteration": 101,
  "NTree": 201,
  "LabelThreshold": 2,
  "TypicalThreshold": 0.8,
  "TruncationParameter": 0.2,
  "ControlLabel": 2,
  "PatientLabel": 1,
  "LabelFile": "participants.csv",
  "QCFile": "qc.csv",
  "ReadDataFileFromHalfpipeOutputs": {
    "atlas": "NeuroMark1",
    "task": "rest"
  },
  "FNCDomainNames": [0, 5, 7, 16, 25, 42, 49, 53],
  "IgnoreSubjectsWithMissingData": true,
  "LogLevel": "info"
}
```

### Settings Specification

| Variable Name         | Type         | Description                                                                                                                                                                                                                                              | Allowed Options   | Default | Required |
|-----------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|---------|----------|
| `SamplingThreshold`   | `float`      | To decide how often a sample must look non noisy across repeated random sampling.                                                                                                                                                                        | float             | 0.7     | ✅ true   |
| `Iteration`           | `int`        | Controls how many times the full CRF process is repeated.                                                                                                                                                                                                | integer           | 101     | ✅ true   |
| `NTree`               | `int`        | Number of decision trees in each complete random forest.                                                                                                                                                                                                 | integer           | 201     | ✅ true   |
| `LabelThreshold`      | `int`        | Number of change sequence happens in leaf nodes of CRF to detect tree label                                                                                                                                                                              | integer           | 2       | ✅ true   |
| `TypicalThreshold`    | `float`      | To decide if a subject is typical or noisy. If score ≥ this threshold, subject is kept as typical subject                                                                                                                                                | float             | 0.8     | ✅ true   |
| `TruncationParameter` | `float`      | It trims the extreme x% from both sides of the negative and positive score distributions before computing the boundary threshold.                                                                                                                        | float             | 0.2     | ✅ true   |
| `ControlLabel`        | `str or int` | Label used to indicate healthy controls in the dataset| String or integer | -       | ✅ true   |
| `PatientLabel`        | `str or int` | Label used to indicate patients or unhealthy subjects| integer           | -       | ✅ true   |
| `LabelFile` | `str` | Participants file in the site folder, with a `subject_id` (or `participant_id`) column and a `diagnosis` column; other columns are ignored. For HALFpipe derivatives, `participants.csv` and then `participants.tsv` are used if this file is not found. | string | - | ✅ true |
| `QCFile` | `str` | HALFpipe derivatives only: per-run QC file in the site folder with columns `subject_id`, `session`, `run`, `include` (yes/no). If it is missing, every run found is used. | string | `qc.csv` | ❌ false |
| `ExistingDataFile`        | `str`        | Name of the csv file containing data of all the subjects (subjects * data). For FNC, the data is the linearized upper traingular data of the FNC matrix. Either this or the Halfpipe outputs should be provided for data. Should not contain any header. | string            | -       | ✅ true (if halfpipe outputs are not provided) ❌ false (otherwise)   |
| `ReadDataFileFromHalfpipeOutputs` | `dict` | Read HALFpipe correlation matrices. `atlas` (required) is the atlas name as it appears in filenames, the same at every site (for example `NeuroMark1`). `task` defaults to `rest`. `feature` (alias `corr`) is only needed if a site has the same atlas under two feature names. Either this or `ExistingDataFile` must be provided. | dict | - | ✅ true (if data file is not provided) ❌ false (otherwise) |
| `IgnoreSubjectsWithMissingData`    | bool        | Ignore or fail the run if any of the subjects with empty or missing values       | bool              | false   | ✅ true  |
| `FNCDomainNames`    | `list`        | Domain names of FNC data used for displaying while rendering FNC data  | string            | []     | ✅ true   |
| `LogLevel`    | `str`        | Logs the errors. Value either "info" or "error"       | string            | "info"   | ❌false   |


### Input Description

Each site provides one of the following.

**Format 1: HALFpipe derivatives (nested, as HALFpipe writes them)**

```
<site data folder>/
├── participants.csv | participants.tsv   subject_id (or participant_id), …, diagnosis
├── qc.csv                                 optional: subject_id, session, run, include
└── <any folder name>/                      HALFpipe derivatives, any depth
    └── sub-<ID>/
        └── ses-<S>/                        optional
            └── func/task-rest/
                ├── sub-<ID>_ses-<S>_task-rest_run-1_feature-<f>_atlas-<atlas>_desc-correlation_matrix.tsv
                └── sub-<ID>_ses-<S>_task-rest_run-2_feature-<f>_atlas-<atlas>_desc-correlation_matrix.tsv
```

At the start of the run, each subject's correlation matrices for the configured atlas are combined into one matrix and the analysis proceeds as for Format 2:

- **Which runs are used:** with `qc.csv`, only runs marked `include` = yes. Every run found on disk for a participant must have a row (yes or no), and every row marked yes must match a file; otherwise the site stops with a list to fix. Without `qc.csv`, every run found is used.
- **How runs are combined:** Fisher-z average of all included runs across all included sessions (`tanh(mean(arctanh(r)))`). An ROI pair missing in one run is averaged from the others. To keep longitudinal visits apart, include only the wanted session in `qc.csv`.
- **Matching:** subject IDs match with or without `sub-` and regardless of case (`01` matches `sub-01`); sessions match with or without `ses-`; runs match as numbers (`1`, `01`, `run-01`).
- **Ignored files:** other atlases, covariance and timeseries files, and stray copies (for example `… copy.tsv`).
- **Site output:** `qc_summary.csv` lists, for each participant, the sessions and runs used and excluded. It stays at the site.

```csv
subject_id,session,run,include
sub-PHCPP1011968,ses-V1A,1,yes
sub-PHCPP1011968,ses-V1A,2,no
sub-PHCPP1011968,ses-V1B,1,yes
```

Sample sites are in `test_data/site1/` (two sessions, `qc.csv`), `test_data/site2/` (no sessions, `participants.tsv`, no QC file) and `test_data/site3/` (one session, `qc.csv`, distractor files). They are generated by `scripts/generate_nested_test_data.py`.

**Format 2: flat folder of HALFpipe matrices**

- One `<subject_id>_task-rest_feature-<corr>_atlas-<atlas>CombinedDseg_desc-correlation_matrix.tsv` per subject at the top level of the site folder, with `ReadDataFileFromHalfpipeOutputs` naming `atlas` and `corr`.
- `labels.csv` (or the configured `LabelFile`) with headers `subject_id` and `diagnosis`; each `subject_id` is the prefix of its matrix filename.

**Format 3: data file**

- `data.csv` (`ExistingDataFile`): one row per subject with the vectorized upper-triangular FNC values, no header.
- `labels.csv` (`LabelFile`): headers `subject_id` and `diagnosis`, in the same subject order as `data.csv`.

### Algorithm Description

The key steps of the algorithm include:

1.  **Local Identification of Typical Subjects**:

    * Each site begins by filtering the subjects using complete random forest models. This step measures the stability of the original label across the labels predicted by these models.

    * This produces a reduced feature set that better reflects the group differences. Using only the typical subjects with their selected features, each site computes an average feature vector for both the healthy and the patient group. These local summaries that represent the site-specific information are then shared with the aggregator (remote) site

2.  **Forming Global Models**:

    * The remote aggregator receives two group averages sent by each of the participating sites. These are aggregated across sites to form a map of centriods for healthy and patient groups.

3. **Computing Dimensional Scores at Each Site**:

    * After receiving the global models, each site computes a dimensional score for every subject. This score quantifies how similar a subject is to the patient versus a healthy reference pattern.

    * This score places the subject along a continuous healthy–patient axis and reduces thousands of imaging features to a single interpretable value.

4. **Federated Adaptive Thresholding**:

    * Dimensional score ranges vary across sites and iterations. In the federated method, each site sends only its dimensional scores to a remote aggregator. The aggregator combines all scores from all sites and computes one global threshold. This global threshold is returned to all sites.

    * Each site then applies the same thresholding rule to determine whether a subject is strongly healthy, strongly patient, or falls in the boundary region.

5. **Producing Updated Labels**:

    * Once the new labels are assigned, each site updates its typical subjects and recomputes its group averages.

    * Over iterations, typical subjects become cleaner, intra-group compactness improves, and inter-group differences become more stable.

6. **Federated Report Generation**:

    * After relabeling, the aggregator combines the per-site relabeled FNC averages into global group averages (one per label) and broadcasts them back to all sites along with the adaptive threshold value.

    * Each site uses this global data together with its local results to generate a self-contained `index.html` report containing: label distribution KPIs, Bonferroni-corrected T-test heatmaps (original vs. relabeled), local and global average FNC heatmaps, and a per-subject dimensional score table with dark-mode support.

### Output Description

Each site produces the following files, in its own output directory:

| File | Description |
| --- | --- |
| `relabeled.csv` | Per-subject dimensional scores and re-assigned labels (−1 = uncertain) |
| `index.html` | Self-contained HTML report with KPIs, heatmaps, and scores table |
| `qc_summary.csv` | HALFpipe inputs only: for each participant, the sessions and runs used and excluded and the matrix size. Stays at the site |
| `original_labels_ttest.png` | Bonferroni-corrected T-test heatmap using original labels |
| `re_labeled_ttest.png` | Bonferroni-corrected T-test heatmap using relabeled subjects |
| `local_original_avg_fnc_{label}.png` | Local average FNC per group before relabeling |
| `local_relabeled_avg_fnc_{label}.png` | Local average FNC per group after relabeling |
| `global_original_avg_fnc_{label}.png` | Federated average FNC per group before relabeling |
| `global_relabeled_avg_fnc_{label}.png` | Federated average FNC per group after relabeling |
| `orig.mat`, `crf.mat`, `centers.npz` | Diagnostic intermediates (validated input data, CRF results, typical-subject centroids) |

### Running this computation

To locally run this computation, clone this repo and run:

```
./run_local_simulation.sh site1,site2
```

This builds the development image, creates the job folder, and runs the
NVFlare simulator in one step. Pass `--no-build` to skip rebuilding the
image on repeat runs (e.g. `./run_local_simulation.sh site1,site2 --no-build`).

Output for each site lands under `test_output/simulate_job/<site>/`.

See [MIGRATION_NOTES.md](MIGRATION_NOTES.md) for what changed when this
computation was migrated to the current `computation-nvflare-boilerplate`
architecture, and how that migration was verified against the prior
implementation.
