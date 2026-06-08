# Methodology

This repository implements a domain-aware causal feature-selection pipeline for actionable depression intervention analysis using NHANES survey data.

## 1. Data ingestion and person-level merge

- Raw NHANES XPT files are loaded from `data/raw/Dataset` using `src.data.loader.read_all_xpt`.
- Each module file is normalized and cast to appropriate numeric types to reduce memory usage.
- Long-form NHANES modules with repeated `SEQN` rows are aggregated to one row per person using median for numeric features and mode for categoricals.
- Modules are merged outer-join style at the person level via `src.data.loader.merge_modules_person_level`.
- Each feature is mapped to a domain label based on its NHANES module, such as `socio-demographic`, `behavioral-lifestyle`, `clinical-health`, or `mental-health-outcome`.

## 2. Outcome construction

- The depression outcome is built as `PHQ9_TOTAL` from DPQ item columns (`DPQ010` through `DPQ090`) using `src.utils.helpers.build_phq9`.
- This total score serves as the continuous target for the regression pipeline.

## 3. Preprocessing and feature filtering

- Rows with missing `PHQ9_TOTAL` are removed.
- Unwanted columns such as `SEQN`, `PHQ9_TOTAL`, and raw DPQ items are dropped before modeling.
- Features with more than 40% missing values are removed to improve data quality.
- Numeric features are imputed with medians and categorical features are imputed with mode values and one-hot encoded with frequency-based category filtering.

## 4. Base model and feature scoring

- A holdout split is used to separate training and test sets.
- A `HistGradientBoostingRegressor` is trained on the processed features to predict `PHQ9_TOTAL`.
- Feature importance is scored using SHAP values when available, with permutation importance fallback if SHAP fails.
- Importance is aggregated from processed feature columns back to base feature names.

## 5. Domain-aware pruning

- Feature importance is grouped by expert-defined domains.
- Within each domain, relative feature importance is computed.
- Features are pruned by keeping only those with at least 2% relative importance per domain and up to the top 10 features per domain.
- This yields `groupwise_pruned_features.csv` and a domain-aware ranked feature list.

## 6. Domain prioritization and selection

- A domain-level regression model is fit using the pruned feature set from selected domains.
- The pipeline optionally uses `ExplainableBoostingRegressor` from `interpret` if available, otherwise it falls back to `pygam.LinearGAM`.
- Domain importance is estimated through permutation importance on the domain model.
- A genetic algorithm (`pygad`) searches the domain selection space to maximize the model's holdout R2 while penalizing feature count and selected domain count.
- Final domain selection and importance outputs are saved as `domain_priority_final.csv`, `domain_importance_ebm.csv`, and `ga_domain_selection.json`.

## 7. Optional binary outcome evaluation

- When `--binary-outcome true`, the continuous PHQ-9 score is thresholded at 10 to produce a binary depression indicator.
- A comparative classification evaluation is performed between the domain-aware selected features and a proxy SGFR feature selection baseline.
- Multiple classifiers are calibrated and evaluated with AUC, macro F1, balanced accuracy, and Brier score.
- Comparative results include ROC grids, Wilcoxon signed-rank summary tables, and rejection maps.

## 8. Outputs

The pipeline writes these artifacts to `data/processed/mental_outputs` by default:

- `domain_mapping.json`
- `groupwise_ranked_features_full.csv`
- `groupwise_pruned_features.csv`
- `domain_importance_ebm.csv`
- `ga_domain_selection.json`
- `domain_priority_final.csv`
- `run_metrics.json`
- `viz_top_pruned_features.png`
- `viz_domain_priority_final.png`
- `cmp_panel_metrics.csv`
- `wilcoxon_positive_rank.csv`
- `wilcoxon_rejection_map.csv`
- `cmp_roc_grid.png`

## 9. Execution

Run the pipeline with:

```powershell
python -m src.models.train
```

Optional runtime arguments include input/output paths, sample sizes, GA settings, and whether to run the binary outcome comparison.
