"""Production pipeline for domain-aware causal feature selection."""

from __future__ import annotations

import gc
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer

from src.data.loader import DOMAIN_BY_MODULE, downcast_numeric, merge_modules_person_level, read_all_xpt
from src.utils.helpers import base_feature_name, build_phq9, ensure_dir, make_ohe, write_json


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    low_mem: bool = True
    row_subsample: int | None = 30000
    shap_sample: int = 2000
    pi_repeats: int = 2
    ohe_min_frequency: float = 0.02
    ohe_max_categories: int = 25
    ga_generations: int = 8
    ga_population: int = 12
    force_gam: bool = True
    eval_sample: int = 2000
    binary_outcome: bool = True
    phq9_binary_threshold: int = 10
    prune_relative_threshold: float = 0.02
    topk_per_group: int = 10
    random_state: int = 42


def _build_preprocessor(
    df: pd.DataFrame,
    min_frequency: float,
    max_categories: int,
) -> tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    cat_cols = [column for column in df.columns if column not in num_cols]
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", make_ohe(min_frequency, max_categories)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, num_cols, cat_cols


def _prepare_training_data(
    merged: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    y = merged["PHQ9_TOTAL"]
    mask = y.notna()
    df = merged.loc[mask].copy()
    y = y.loc[mask].astype("float32")

    if config.low_mem and config.row_subsample and len(df) > config.row_subsample:
        df = df.sample(n=config.row_subsample, random_state=config.random_state)
        y = y.loc[df.index]
        print(f"[INFO] Row subsample -> {len(df)} rows")

    drop_cols = {
        column
        for column in df.columns
        if column == "SEQN" or column == "PHQ9_TOTAL" or column.startswith("DPQ")
    }
    x = df.drop(columns=list(drop_cols), errors="ignore")
    x = x.loc[:, x.notna().sum().ge(int(0.6 * len(x)))].copy()
    return x, y, {}


def _score_features(
    pipe: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    col_domain: Dict[str, str],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat_proc = pipe.named_steps["prep"].get_feature_names_out().tolist()

    try:
        import shap

        x_proc_full = pipe.named_steps["prep"].transform(x_test)
        if config.low_mem and config.shap_sample and x_proc_full.shape[0] > config.shap_sample:
            idx = np.random.RandomState(config.random_state).choice(
                x_proc_full.shape[0], config.shap_sample, replace=False
            )
            x_proc = x_proc_full[idx]
        else:
            x_proc = x_proc_full

        explainer = shap.Explainer(
            pipe.named_steps["model"], feature_names=feat_proc, algorithm="tree"
        )
        shap_values = explainer(x_proc)
        importance = np.abs(shap_values.values).mean(axis=0)
        imp_df = pd.DataFrame({"feature_proc": feat_proc, "shap_importance": importance})
    except Exception as exc:
        print(f"[WARN] SHAP failed ({exc}); using permutation importance.")
        result = permutation_importance(
            pipe,
            x_test,
            y_test,
            n_repeats=config.pi_repeats,
            random_state=config.random_state,
            n_jobs=-1,
            scoring="r2",
        )
        raw_names = list(getattr(pipe.named_steps["prep"], "feature_names_in_", x_test.columns))
        imp_df = pd.DataFrame(
            {"feature_proc": raw_names, "shap_importance": result.importances_mean}
        )

    imp_df["feature_base"] = imp_df["feature_proc"].map(base_feature_name)
    ranked = imp_df.groupby("feature_base", as_index=False)["shap_importance"].sum()
    ranked["domain"] = ranked["feature_base"].map(lambda name: col_domain.get(name, "other"))
    ranked["group_rel"] = ranked.groupby("domain")["shap_importance"].transform(
        lambda values: values / (values.sum() + 1e-9)
    )
    keep = (
        ranked.query("group_rel >= @config.prune_relative_threshold")
        .sort_values(["domain", "shap_importance"], ascending=[True, False])
        .groupby("domain", as_index=False)
        .head(config.topk_per_group)
    )
    return ranked, keep


def _fit_domain_model(
    domains_selected: List[str],
    keep: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    col_domain: Dict[str, str],
    config: PipelineConfig,
):
    kept_bases = set(keep["feature_base"].tolist())
    base_cols = [
        column
        for column in x_train.columns
        if col_domain.get(column, "other") in domains_selected and column in kept_bases
    ]
    if not base_cols:
        return {"r2": -np.inf, "mae": np.inf, "n_features": 0, "model": None, "base_cols": []}

    x_all = pd.concat([x_train, x_test], axis=0)
    y_all = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)
    x_sub = x_all[base_cols].copy()
    prep, _, _ = _build_preprocessor(
        x_sub, config.ohe_min_frequency, config.ohe_max_categories
    )
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_sub, y_all, test_size=0.25, random_state=12
    )

    if not config.force_gam:
        try:
            from interpret.glassbox import ExplainableBoostingRegressor

            estimator = ExplainableBoostingRegressor(
                random_state=12,
                max_bins=128,
                interactions=0,
                outer_bags=4,
                inner_bags=0,
                learning_rate=0.02,
                n_jobs=-1,
            )
            model = Pipeline([("prep", prep), ("model", estimator)])
            model.fit(x_tr, y_tr)
            y_hat = model.predict(x_val)
            return {
                "r2": r2_score(y_val, y_hat),
                "mae": mean_absolute_error(y_val, y_hat),
                "n_features": int(model.named_steps["prep"].get_feature_names_out().size),
                "model": model,
                "base_cols": base_cols,
            }
        except Exception as exc:
            print(f"[WARN] EBM unavailable ({exc}); falling back to LinearGAM.")

    from pygam import LinearGAM

    x_tr_proc = prep.fit_transform(x_tr)
    x_val_proc = prep.transform(x_val)
    gam = LinearGAM().gridsearch(x_tr_proc, y_tr, progress=False)
    y_hat = gam.predict(x_val_proc)

    class _GamWrapper:
        def __init__(self, preprocessor, estimator):
            self.named_steps = {"prep": preprocessor, "model": estimator}

    return {
        "r2": r2_score(y_val, y_hat),
        "mae": mean_absolute_error(y_val, y_hat),
        "n_features": int(x_tr_proc.shape[1]),
        "model": _GamWrapper(prep, gam),
        "base_cols": base_cols,
    }


def _domain_prioritization(
    keep: pd.DataFrame,
    ranked: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    col_domain: Dict[str, str],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    all_domains = sorted(set(col_domain.get(c, "other") for c in keep["feature_base"]))
    if not all_domains:
        all_domains = sorted(set(col_domain.get(c, "other") for c in ranked["feature_base"]))

    domain_model = _fit_domain_model(
        all_domains, keep, x_train, x_test, y_train, y_test, col_domain, config
    )
    if domain_model["model"] is None:
        raise RuntimeError("Domain model could not be fit because no domain features were selected.")

    pipe_dom = domain_model["model"]
    prep = pipe_dom.named_steps["prep"]
    estimator = pipe_dom.named_steps["model"]
    eval_cols = domain_model["base_cols"]
    x_eval_proc_full = prep.transform(x_test[eval_cols])

    if config.low_mem and config.eval_sample and x_eval_proc_full.shape[0] > config.eval_sample:
        row_idx = np.random.RandomState(7).choice(
            x_eval_proc_full.shape[0], config.eval_sample, replace=False
        )
        x_eval_proc = x_eval_proc_full[row_idx]
        y_eval = y_test.iloc[row_idx]
    else:
        x_eval_proc = x_eval_proc_full
        y_eval = y_test

    feat_names = prep.get_feature_names_out().tolist()
    importance = _manual_permutation_importance(
        estimator,
        x_eval_proc,
        y_eval,
        n_repeats=config.pi_repeats,
        random_state=7,
    )
    dom_imp = pd.DataFrame({"feature_proc": feat_names, "perm_importance": importance})
    dom_imp["domain"] = dom_imp["feature_proc"].map(
        lambda feature: col_domain.get(base_feature_name(feature), "other")
    )
    dom_summary = (
        dom_imp.groupby("domain", as_index=False)["perm_importance"]
        .sum()
        .sort_values("perm_importance", ascending=False)
    )

    import pygad

    domain_list = all_domains
    dimensions = len(domain_list)
    alpha, beta, gamma = 1.0, 0.0005, 0.01

    def fitness_func(ga_instance, solution, solution_idx):
        bits = [int(round(bit)) for bit in solution]
        selected = [domain_list[index] for index, bit in enumerate(bits) if bit == 1]
        if not selected:
            return -1e9
        result = _fit_domain_model(
            selected, keep, x_train, x_test, y_train, y_test, col_domain, config
        )
        if not np.isfinite(result["r2"]):
            return -1e9
        return float(alpha * result["r2"] - beta * result["n_features"] - gamma * len(selected))

    initial_population = np.random.randint(0, 2, size=(config.ga_population, dimensions))
    ga = pygad.GA(
        num_generations=config.ga_generations,
        num_parents_mating=min(6, config.ga_population),
        fitness_func=fitness_func,
        sol_per_pop=config.ga_population,
        num_genes=dimensions,
        gene_space=[0, 1],
        gene_type=int,
        initial_population=initial_population,
        mutation_probability=0.12,
        allow_duplicate_genes=True,
        suppress_warnings=True,
    )
    ga.run()
    best_solution, best_fitness, _ = ga.best_solution()
    best_bits = [int(round(bit)) for bit in best_solution]
    best_domains = [domain_list[index] for index, bit in enumerate(best_bits) if bit == 1]
    best_result = _fit_domain_model(
        best_domains, keep, x_train, x_test, y_train, y_test, col_domain, config
    )
    ga_out = {
        "domain_list": domain_list,
        "best_domains": best_domains,
        "fitness": float(best_fitness),
        "r2": float(best_result["r2"]),
        "mae": float(best_result["mae"]),
        "n_features": int(best_result["n_features"]),
    }
    dom_summary["selected_by_GA"] = dom_summary["domain"].isin(best_domains).astype(int)
    return dom_summary, ga_out


def _manual_permutation_importance(
    estimator,
    x_matrix,
    y_true: pd.Series,
    n_repeats: int,
    random_state: int,
) -> np.ndarray:
    """Permutation importance for estimators that only expose ``predict``."""
    rng = np.random.RandomState(random_state)
    x_base = np.asarray(x_matrix)
    y_values = np.asarray(y_true)
    baseline = r2_score(y_values, estimator.predict(x_base))
    importances = np.zeros(x_base.shape[1], dtype=float)

    for feature_idx in range(x_base.shape[1]):
        drops = []
        for _ in range(n_repeats):
            x_perm = x_base.copy()
            x_perm[:, feature_idx] = rng.permutation(x_perm[:, feature_idx])
            drops.append(baseline - r2_score(y_values, estimator.predict(x_perm)))
        importances[feature_idx] = float(np.mean(drops))
    return importances


def _run_binary_comparison(
    keep: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    col_domain: Dict[str, str],
    config: PipelineConfig,
) -> None:
    from scipy.stats import wilcoxon
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    y_train_bin = (y_train >= config.phq9_binary_threshold).astype(int)
    y_test_bin = (y_test >= config.phq9_binary_threshold).astype(int)
    if y_train_bin.nunique() < 2 or y_test_bin.nunique() < 2:
        print("[WARN] Binary comparison skipped: train or test split has one class.")
        return

    def select_cols(base_names: List[str]) -> List[str]:
        base_names = set(base_names)
        return [column for column in x_train.columns if column in base_names]

    def sgfr_proxy_rank(topk_per_domain: int) -> List[str]:
        selected = []
        for domain in sorted(set(col_domain.get(column, "other") for column in x_train.columns)):
            cols = [column for column in x_train.columns if col_domain.get(column, "other") == domain]
            numeric_cols = [column for column in cols if pd.api.types.is_numeric_dtype(x_train[column])]
            if len(numeric_cols) < 2:
                selected.extend(cols[:topk_per_domain])
                continue
            from sklearn.decomposition import PCA

            x_group = x_train[numeric_cols].fillna(x_train[numeric_cols].median(numeric_only=True))
            pca = PCA(n_components=1, random_state=7).fit(x_group)
            order = np.argsort(np.abs(pca.components_[0]))[::-1]
            selected.extend([numeric_cols[index] for index in order[:topk_per_domain]])
        return list(dict.fromkeys(selected))

    our_cols = select_cols(keep["feature_base"].tolist())
    sgfr_cols = select_cols(sgfr_proxy_rank(config.topk_per_group))
    if not our_cols or not sgfr_cols:
        print("[WARN] Binary comparison skipped: empty feature set.")
        return

    def transform_cols(cols: List[str]):
        preprocessor, _, _ = _build_preprocessor(
            x_train[cols], config.ohe_min_frequency, config.ohe_max_categories
        )
        return preprocessor.fit_transform(x_train[cols]), preprocessor.transform(x_test[cols])

    x_train_ours, x_test_ours = transform_cols(our_cols)
    x_train_sgfr, x_test_sgfr = transform_cols(sgfr_cols)
    classifiers = {
        "LR": LogisticRegression(max_iter=2000),
        "SVM": SVC(probability=True),
        "RF": RandomForestClassifier(n_estimators=400, random_state=config.random_state),
        "GBC": GradientBoostingClassifier(random_state=config.random_state),
    }

    def eval_panel(x_tr, x_te, tag: str):
        rows = []
        roc_curves = {}
        for name, base in classifiers.items():
            calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
            calibrated.fit(x_tr, y_train_bin)
            prob = calibrated.predict_proba(x_te)[:, 1]
            pred = (prob >= 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_test_bin, prob)
            roc_curves[name] = (fpr, tpr)
            rows.append(
                {
                    "Model": name,
                    "Tag": tag,
                    "AUC": roc_auc_score(y_test_bin, prob),
                    "MacroF1": f1_score(y_test_bin, pred, average="macro"),
                    "BalancedAcc": balanced_accuracy_score(y_test_bin, pred),
                    "Brier": brier_score_loss(y_test_bin, prob),
                }
            )
        return pd.DataFrame(rows), roc_curves

    results_ours, roc_ours = eval_panel(x_train_ours, x_test_ours, "Ours")
    results_sgfr, roc_sgfr = eval_panel(x_train_sgfr, x_test_sgfr, "SGFR")
    results = pd.concat([results_ours, results_sgfr], axis=0).reset_index(drop=True)
    results.to_csv(config.output_dir / "cmp_panel_metrics.csv", index=False)

    rank_mat = pd.DataFrame(0.0, index=["Ours", "SGFR"], columns=["Ours", "SGFR"])
    rej_map = pd.DataFrame("", index=["Ours", "SGFR"], columns=["Ours", "SGFR"])
    models = sorted(results["Model"].unique())
    perf = {
        tag: results.query("Tag == @tag").set_index("Model")["AUC"]
        for tag in ("Ours", "SGFR")
    }
    for row, col in itertools.permutations(["Ours", "SGFR"], 2):
        diff = perf[row].loc[models].values - perf[col].loc[models].values
        try:
            _, p_value = wilcoxon(
                diff, zero_method="wilcox", alternative="two-sided", correction=True, mode="auto"
            )
        except ValueError:
            p_value = 1.0
        rank_mat.loc[row, col] = float((diff > 0).sum())
        rej_map.loc[row, col] = "blue" if p_value < 0.05 and diff.mean() > 0 else ("red" if p_value < 0.05 else "")

    rank_mat.to_csv(config.output_dir / "wilcoxon_positive_rank.csv")
    rej_map.to_csv(config.output_dir / "wilcoxon_rejection_map.csv")
    _plot_roc_grid(roc_ours, roc_sgfr, config.output_dir / "cmp_roc_grid.png")
    print("[SAVE] comparative binary evaluation artifacts")


def _plot_roc_grid(roc_ours, roc_sgfr, output_path: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for ax, name in zip(axs.ravel(), ["LR", "SVM", "RF", "GBC"]):
        fpr, tpr = roc_ours[name]
        ax.plot(fpr, tpr, label="Ours")
        fpr2, tpr2 = roc_sgfr[name]
        ax.plot(fpr2, tpr2, linestyle="--", label="SGFR")
        ax.plot([0, 1], [0, 1], linestyle=":")
        ax.set_title(f"ROC: {name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_outputs(keep: pd.DataFrame, dom_summary: pd.DataFrame, output_dir: Path) -> None:
    top = keep.sort_values("shap_importance", ascending=False).head(20)
    plt.figure(figsize=(10, 7))
    labels = top["feature_base"] + " [" + top["domain"] + "]"
    plt.barh(labels[::-1], top["shap_importance"][::-1])
    plt.title("Top 20 Pruned Features")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_top_pruned_features.png", dpi=200)
    plt.close()

    domain_scores = dom_summary.sort_values("perm_importance", ascending=False)
    plt.figure(figsize=(8, 5))
    labels = domain_scores["domain"] + domain_scores["selected_by_GA"].map({1: " *", 0: ""})
    plt.barh(labels[::-1], domain_scores["perm_importance"][::-1])
    plt.title("Domain Importance and GA Selection")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_domain_priority_final.png", dpi=200)
    plt.close()


def run_pipeline(config: PipelineConfig) -> dict:
    """Run the full domain-aware causal feature-selection workflow."""
    ensure_dir(config.output_dir)
    modules = read_all_xpt(config.input_dir)
    merged, col_domain = merge_modules_person_level(modules)
    del modules
    gc.collect()

    merged = downcast_numeric(build_phq9(merged))
    write_json(config.output_dir / "domain_mapping.json", DOMAIN_BY_MODULE)
    x, y, _ = _prepare_training_data(merged, config)
    del merged
    gc.collect()

    prep, _, _ = _build_preprocessor(x, config.ohe_min_frequency, config.ohe_max_categories)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=config.random_state
    )
    del x
    gc.collect()

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=config.random_state,
    )
    pipe = Pipeline([("prep", prep), ("model", regressor)])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    metrics = {
        "holdout_r2": float(r2_score(y_test, y_pred)),
        "holdout_mae": float(mean_absolute_error(y_test, y_pred)),
    }
    print(f"[METRIC] Holdout R2={metrics['holdout_r2']:.3f} | MAE={metrics['holdout_mae']:.3f}")

    ranked, keep = _score_features(pipe, x_test, y_test, col_domain, config)
    ranked.sort_values(["domain", "shap_importance"], ascending=[True, False]).to_csv(
        config.output_dir / "groupwise_ranked_features_full.csv", index=False
    )
    keep.to_csv(config.output_dir / "groupwise_pruned_features.csv", index=False)
    print("[SAVE] groupwise feature artifacts")

    dom_summary, ga_out = _domain_prioritization(
        keep, ranked, x_train, x_test, y_train, y_test, col_domain, config
    )
    dom_summary.to_csv(config.output_dir / "domain_importance_ebm.csv", index=False)
    dom_summary.to_csv(config.output_dir / "domain_priority_final.csv", index=False)
    write_json(config.output_dir / "ga_domain_selection.json", ga_out)
    print("[SAVE] domain prioritization artifacts")

    _plot_outputs(keep, dom_summary, config.output_dir)

    if config.binary_outcome:
        _run_binary_comparison(keep, x_train, x_test, y_train, y_test, col_domain, config)
    else:
        print("[INFO] Binary comparative evaluation skipped.")

    write_json(config.output_dir / "run_metrics.json", metrics)
    return metrics
