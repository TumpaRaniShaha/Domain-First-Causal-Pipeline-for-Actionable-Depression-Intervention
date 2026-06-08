# Domain-Aware Causal Feature Selection for Depression

A research-oriented Python implementation of a domain-aware causal feature selection pipeline for actionable depression intervention analysis using NHANES survey data.

## Overview

This project demonstrates a reproducible workflow for selecting depression-related features with domain-awareness, combining:

- NHANES multi-module data ingestion and person-level merging
- PHQ-9 depression outcome construction
- tree-based regression with SHAP-driven feature scoring
- domain-aware pruning of candidate variables
- genetic algorithm selection of domain subsets
- optional binary classification comparison for depression screening

The implementation is designed to support both exploratory research and production-style reproducibility.

## Research Motivation

Depression is a complex outcome influenced by demographic, lifestyle, clinical, and behavioral factors.
This pipeline emphasizes:

- domain-aware grouping of features to preserve interpretability
- causal-minded feature selection through SHAP importance and domain-level evaluation
- a smaller, more actionable feature set that remains predictive across domains

## Repository Structure

- `configs/default.yaml` - default configuration and runtime parameters.
- `data/raw/Dataset/` - source NHANES `.xpt` survey files.
- `data/processed/mental_outputs/` - generated outputs, metrics, and visualizations.
- `notebooks/` - original exploratory research notebook.
- `src/data/loader.py` - NHANES ingestion, module inference, aggregation, and merge utilities.
- `src/causal/feature_selection.py` - end-to-end pipeline implementation, feature scoring, domain prioritization, and evaluation.
- `src/models/train.py` - CLI entrypoint for running the pipeline.
- `src/utils/helpers.py` - shared utilities for preprocessing, PHQ-9 construction, and output management.
- `docs/methodology.md` - detailed workflow and methodology documentation.

## Key Components

1. **Data ingestion**
   - Read all NHANES XPT files from `data/raw/Dataset`
   - Infer NHANES module labels and map features to expert domains
   - Aggregate repeated measurements to one person-level row

2. **Outcome construction**
   - Build `PHQ9_TOTAL` from DPQ items
   - Use PHQ-9 as the primary continuous depression target

3. **Feature engineering and filtering**
   - Impute numeric values with median and categorical values with mode
   - One-hot encode categorical features with frequency-based filtering
   - Remove features with excessive missingness

4. **Feature scoring and pruning**
   - Train a `HistGradientBoostingRegressor` model on holdout-split data
   - Compute SHAP-based feature importance; fallback to permutation importance if needed
   - Aggregate importance at the base feature level and prune by domain-specific relative contribution

5. **Domain prioritization**
   - Fit a domain-level model using selected pruned features
   - Estimate domain importance via permutation importance on the fitted model
   - Search for the optimal domain subset using a genetic algorithm

6. **Optional classification comparison**
   - Convert PHQ-9 to a binary depression outcome using threshold 10
   - Compare the proposed domain-aware feature set against a baseline proxy
   - Evaluate with calibrated classifiers, AUC, F1, balanced accuracy, and Wilcoxon tests

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Preparation

Place the raw NHANES `.xpt` files in:

```text
data/raw/Dataset
```

If you are using the original notebook data source, this repo-local path replaces the prior Colab-specific data locations.

## Running the Pipeline

Run the full pipeline with:

```powershell
python -m src.models.train
```

Override input/output paths or configure sampling and GA settings:

```powershell
python -m src.models.train --input-dir data/raw/Dataset --output-dir data/processed/mental_outputs --row-subsample 2000 --shap-sample 500 --ga-generations 2 --ga-population 6 --binary-outcome false
```

## Output Artifacts

Default outputs are saved to `data/processed/mental_outputs`:

- `domain_mapping.json`
- `groupwise_ranked_features_full.csv`
- `groupwise_pruned_features.csv`
- `domain_importance_ebm.csv`
- `domain_priority_final.csv`
- `ga_domain_selection.json`
- `run_metrics.json`
- `viz_top_pruned_features.png`
- `viz_domain_priority_final.png`

If binary evaluation is enabled:

- `cmp_panel_metrics.csv`
- `wilcoxon_positive_rank.csv`
- `wilcoxon_rejection_map.csv`
- `cmp_roc_grid.png`

## Notes for Researchers

- `docs/methodology.md` contains a detailed description of the pipeline and design decisions.
- The notebook in `notebooks/` contains the exploratory analysis and initial model development.
- The code is written to be extensible: adjust domain labels, pruning thresholds, or model choices in the pipeline source.

## Citation

If you reuse or extend this work, please cite the repository and acknowledge the domain-aware causal selection approach.
