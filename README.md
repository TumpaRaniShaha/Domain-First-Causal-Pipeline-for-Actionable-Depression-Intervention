# Severity Backtracking Depression Intervention through Domain-Aware Causally Consistent Grouped Feature Learning with Interpretable Additive Model

## Description
This repository contains a Domain-Aware Causally-Consisted Grouped Feature Selection (DCGFS) framework for identify the significant domain and feature to plan the severity backtract intervention. It prioritizes domain knowledge before model construction. Instead of treating features as independent variables, the framework organizes variables into clinically meaningful domains, evaluates their contribution using interpretable additive learner, and identifies the most influential domains and features through causal and optimization-driven analysis.

## Dataset Information
This study utilizes the publicly available National Health and Nutrition Examination Survey (NHANES) 2017–March 2020 Pre-pandemic dataset, provided by the National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC). The dataset contains demographic, socioeconomic, behavioral, and health-related variables along with the Patient Health Questionnaire-9 (PHQ-9), a clinically validated instrument for depression assessment. The PHQ-9 score, ranging from 0 to 27, was used to derive depression severity levels.

**Source** NHANES 2017–March 2020 Pre-pandemic Data, CDC/NCHS https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2017-2020

**Citation**
Centers for Disease Control and Prevention (CDC). National Health and Nutrition Examination Survey (NHANES), 2017–March 2020 Pre-pandemic Data Files. National Center for Health Statistics (NCHS), Hyattsville, MD, USA. Available at: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2017-2020. 

**Depression Severity Classes** Following the standard PHQ-9 interpretation, participants were categorized into five severity levels: Normal (0–4), Mild (5–9), Moderate (10–14), Moderately Severe (15–19), and Severe (20–27).

**Preprocessing Steps** The preprocessing pipeline included 
1. Merging relevant NHANES questionnaire and demographic files using the participant identifier (SEQN),
2. Removing duplicate records,
3. Handling missing values through data quality screening and imputation/removal where appropriate,
4. Encoding categorical variables into numerical representations,
5. Constructing the depression severity target variable from PHQ-9 scores,
6. Standardizing variable formats, and
7. Filtering incomplete samples to ensure analytical consistency.

The final processed dataset was used for domain-aware grouping, causal feature analysis, and intervention-oriented modeling.


# Overview of the process
The pipeline integrates:
Domain-aware feature grouping
Structured feature prioritization
Explainable machine learning
SHAP-based feature interpretation
Permutation importance analysis
Genetic Algorithm (GA) based domain optimization
Group-wise feature ranking and pruning
Comparative evaluation against conventional feature selection approaches

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

