# Surrogate Assets

## Purpose

This note records the current surrogate-model assets used in the engineering
layout study. The project now contains several datasets and feature versions.
They are not interchangeable even when their directory names look similar.

The two targets should be tracked separately:

- `u_sigma` is the stress utilization ratio. The current best grouped-section
  model uses grouped geometry features and a reduced feature set.
- `u_delta` is the deflection utilization ratio. The best recorded model still
  uses the earlier 30-dimensional engineering feature vector. Grouped-feature
  delta experiments are recorded below because they are the relevant comparison
  for grouped-section optimization.

## Current Best Records

### Stress Ratio `u_sigma`

| Item | Current best recorded asset |
| --- | --- |
| Dataset | `results/surrogate_data/engineering_230k_group_v2_tail/dataset.pkl` |
| Dataset feature format | `group_v2`, 536 base columns |
| Training feature set | `reduced_v2`, 210 selected columns |
| Model | `results/surrogate_data/engineering_230k_group_v2_tail/sigma_reduced_v2.pkl` |
| Result record | `results/surrogate_data/engineering_230k_group_v2_tail/sigma_reduced_v2.json` |
| Test metrics | MAE `0.0563`, RMSE `0.0992`, R2 `0.9850` |

This is the current main stress surrogate for grouped-section study. It is a
single ResNet regression model trained by
`scripts/experiment/train_sigma_feature_experiment.py`.

The 5-fold grouped `group_v2` model on the original 200k dataset is almost tied:

| Item | Asset |
| --- | --- |
| Dataset | `results/surrogate_data/engineering_200k_group/dataset_group_v2.pkl` |
| Feature set | `base`, 536 columns |
| Model | `results/surrogate_data/engineering_200k_group/sigma_group_v2_kfold5.pkl` |
| Result record | `results/surrogate_data/engineering_200k_group/sigma_group_v2_kfold5.json` |
| Test metrics | MAE `0.0494`, RMSE `0.0910`, R2 `0.9849` |

The k-fold model has a slightly lower recorded R2 and uses the full 536-column
feature vector. Its lower MAE and RMSE come from a different train/test split
and dataset distribution, so it should not be treated as a strict win over the
230k `reduced_v2` model.

### Deflection Ratio `u_delta`

| Item | Best recorded asset |
| --- | --- |
| Dataset | `results/surrogate_data/engineering_200k/dataset.pkl` |
| Dataset feature format | legacy engineering features, 30 columns |
| Training feature set | `base` |
| Model | `results/surrogate_data/engineering_200k/delta_single_legacy30.pkl` |
| Result record | `results/surrogate_data/engineering_200k/delta_single_legacy30.json` |
| Test metrics | MAE `0.0152`, RMSE `0.0295`, R2 `0.9995` |

This is the strongest recorded delta regression result. It belongs to the
legacy engineering feature space and does not preserve grouped beam geometry or
per-group section information.

For the grouped-section feature space, the best recorded delta result is:

| Item | Grouped comparison asset |
| --- | --- |
| Dataset | `results/surrogate_data/engineering_200k_group/dataset.pkl` |
| Dataset feature format | `group_v1`, 236 columns |
| Training feature set | `base` |
| Model | `results/surrogate_data/engineering_200k_group/delta_single_group_v1.pkl` |
| Result record | `results/surrogate_data/engineering_200k_group/delta_single_group_v1.json` |
| Test metrics | MAE `0.0606`, RMSE `0.1245`, R2 `0.9898` |

Delta models trained on the newer 230k `group_v2` tail dataset performed worse
in the recorded experiments:

- `delta_base.json`: R2 `0.9210` with 536 `group_v2` base features.
- `delta_reduced_v2.json`: R2 `0.9353` with 210 `reduced_v2` features.

The current evidence therefore separates the two cases:

- For the best delta accuracy on the legacy engineering space, use the 30-column
  legacy model.
- For grouped-section work, do not assume the current best sigma feature path is
  also the best delta feature path.

## Dataset Lineage

| Dataset | Samples | Stored feature format | Base columns | Sample specs | Role |
| --- | ---: | --- | ---: | --- | --- |
| `results/surrogate_data/engineering_200k/dataset.pkl` | 200000 | legacy | 30 | no | Original engineering surrogate dataset. Strong delta baseline. |
| `results/surrogate_data/engineering_200k_group/dataset.pkl` | 200000 | `group_v1` | 236 | no `sample_specs` payload | First grouped dataset feature matrix. Used by reproduced grouped joint model and grouped delta comparison. |
| `results/surrogate_data/engineering_200k_group/dataset_group_v2.pkl` | 200000 | `group_v2` | 536 | yes | Rebuilt grouped dataset with geometry-aware group features. |
| `results/surrogate_data/engineering_sigma_tail_30k/dataset.pkl` | 30000 | `group_v2` | 536 | yes | Targeted extra samples for high-error sigma regions. |
| `results/surrogate_data/engineering_230k_group_v2_tail/dataset.pkl` | 230000 | `group_v2` | 536 | yes | Merge of the 200k `group_v2` dataset and the 30k sigma-tail dataset. Current grouped sigma main dataset. |

The merged 230k dataset records its sources in `source_datasets`:

1. `results/surrogate_data/engineering_200k_group/dataset_group_v2.pkl`
2. `results/surrogate_data/engineering_sigma_tail_30k/dataset.pkl`

The saved `sample_specs` in the `group_v2` datasets matter. They preserve the
design information needed to rebuild alternative feature matrices without
rerunning FEA.

## Feature Versions

### Legacy Engineering Features

Code entry:

- `src/optimization/design_space.py::design_to_feature_vector`

Stored example:

- `results/surrogate_data/engineering_200k/dataset.pkl`

The legacy vector has 30 columns. It stores global geometry, pressure, topology
features, and one aggregate section-property description for each beam type. It
works very well for the legacy delta model, but it collapses grouped section
assignments and therefore loses information needed by grouped stress prediction.

### Group V1

Code entries:

- `src/optimization/design_space.py::design_to_group_feature_vector`
- `src/optimization/design_space.py::group_feature_names`

Stored example:

- `results/surrogate_data/engineering_200k_group/dataset.pkl`

`group_v1` has 236 columns. It keeps fixed slots for up to three groups of each
beam type. It adds group counts, per-group section properties, per-type section
statistics, and section-demand interaction features.

This version explains the jump from the early grouped sigma result around R2
`0.91` to the later reproduction around R2 `0.97`: the grouped section
distribution is no longer reduced to only per-type averages.

### Group V2

Code entries:

- `src/optimization/design_space.py::design_to_group_feature_vector_v2`
- `src/optimization/design_space.py::group_feature_names_v2`

Stored examples:

- `results/surrogate_data/engineering_200k_group/dataset_group_v2.pkl`
- `results/surrogate_data/engineering_230k_group_v2_tail/dataset.pkl`

`group_v2` has 536 columns. It extends `group_v1` with geometry features for the
beam groups, including group member counts, normalized length summaries,
location statistics, edge and corner proximity, orientation fractions, face
fractions, and geometry-section demand interactions.

This is the base feature format for the best current grouped sigma model.

### Reduced V2

Code entry:

- `scripts/experiment/train_sigma_feature_experiment.py::build_reduced_v2_features`

`reduced_v2` is not a separate stored dataset format. It is a deterministic
feature subset built from `group_v2` columns at training time. The current
selection has 210 columns:

- global geometry, pressure, load, and topology descriptors;
- compact main-beam grouped section and geometry features;
- compact secondary-beam grouped section and geometry features;
- compact brace grouped slot and geometry features;
- compact frame summary features without full frame group geometry slots.

The reduced representation is the current preferred sigma feature set because it
slightly improves the best recorded R2 while cutting the 536-column `group_v2`
base vector to 210 columns.

## Feature And Model Comparison

### Stress Experiments

| Dataset | Feature set | Columns | Model/result | Test R2 | Note |
| --- | --- | ---: | --- | ---: | --- |
| 230k `group_v2` tail | `reduced_v2` | 210 | `sigma_reduced_v2` | `0.9850` | Current best recorded grouped sigma model. |
| 200k `group_v2` | `base`, 5-fold ensemble | 536 | `sigma_group_v2_kfold5` | `0.9849` | Nearly tied full-feature ensemble. |
| 230k `group_v2` tail | `reduced_v1` | 281 | `sigma_reduced_v1` | `0.9844` | Slightly worse than `reduced_v2`. |
| 230k `group_v2` tail | LightGBM top-150 ResNet input | 150 | `sigma_resnet_lgb_top150` | `0.9843` | Useful compact comparison. |
| 230k `group_v2` tail | `base` | 536 | `sigma_group_v2_tail` | `0.9839` | Full-feature single-model baseline. |
| 200k `group_v1` | `base` | 236 | `sigma_v2` | `0.9683` | Earlier grouped feature version. |

### Delta Experiments

| Dataset | Feature set | Columns | Model/result | Test R2 | Note |
| --- | --- | ---: | --- | ---: | --- |
| 200k legacy | `base` | 30 | `delta_single_legacy30` | `0.9995` | Best recorded delta model. |
| 200k `group_v1` | `base` | 236 | `delta_single_group_v1` | `0.9898` | Best recorded grouped delta comparison. |
| 230k `group_v2` tail | `reduced_v2` | 210 | `delta_reduced_v2` | `0.9353` | Sigma feature reduction does not transfer well to delta. |
| 230k `group_v2` tail | `base` | 536 | `delta_base` | `0.9210` | Grouped sigma-tail distribution is poor for recorded delta training. |

## Training Entry Points

The important scripts are:

| Script | Role |
| --- | --- |
| `src/analysis/surrogate_model.py` | Original joint `u_sigma` and `u_delta` K-fold ResNet trainer. |
| `scripts/experiment/train_sigma_feature_experiment.py` | Single-target ResNet experiments for `sigma` or `delta`, feature-set comparisons, optional feature selection by importance file. |
| `scripts/experiment/train_sigma_kfold.py` | K-fold sigma-only ResNet experiment on grouped feature sets. |
| `scripts/experiment/train_sigma_lightgbm.py` | LightGBM sigma regression and feature-importance experiments. |
| `scripts/experiment/rebuild_dataset_features.py` | Rebuilds feature matrices from saved `sample_specs`. |
| `scripts/experiment/merge_surrogate_datasets.py` | Builds merged datasets such as the 230k sigma-tail dataset. |

## Practical Rules

1. Always identify a surrogate asset by dataset path, feature format, feature
   set, and model path together.
2. Do not compare results from 30-column legacy features, 236-column `group_v1`,
   and 536-column `group_v2` as though they used the same input information.
3. Do not overwrite a dataset path with a different feature format without
   recording the change. The `engineering_200k_group` directory already contains
   assets from more than one feature generation stage.
4. For new grouped stress work, start from the 230k `group_v2` dataset and the
   210-column `reduced_v2` feature path unless a new experiment is explicitly
   testing a feature revision.
5. For a future production surrogate pair used by grouped optimization, decide
   whether `u_sigma` and `u_delta` must share one feature vector and one model
   container. The current best recorded target-specific models do not yet give a
   single best grouped pair.
