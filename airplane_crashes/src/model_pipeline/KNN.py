# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# 1. PIPELINE SETUP
# =========================
def build_knn_pipeline(use_smote=False):
    """
    Pipeline:
    Feature Selection (filter) -> [optional SMOTE] -> KNN
    """


    steps = [
        ("feature_selection", SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42))),
    ]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))

    steps.append(("knn", KNeighborsClassifier()))

    if use_smote:
        return ImbPipeline(steps)
    else:
        return Pipeline(steps)


# =========================
# 2. GRID SEARCH (CV)
# =========================
def tune_knn(X_train, y_train, scoring="f1", fast=False, use_smote=False):
    print("\n" + "=" * 60)
    print("STEP 1/5: HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"  Input shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"  Scoring metric: {scoring}")
    print(f"  Mode: {'FAST - GridSearchCV' if fast else 'FULL - BayesSearchCV'}")
    print(f"  SMOTE: {'ON' if use_smote else 'OFF'}")

    pipe = build_knn_pipeline(use_smote=use_smote)
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]

    if fast:
        param_grid = {
            "feature_selection__k": list(range(5, n_features + 1, 5)),
            "knn__n_neighbors": list(range(1, n_samples // 20, 5)),
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"],
        }
        cv = 3

        n_combinations = 1
        for v in param_grid.values():
            n_combinations *= len(v)
        total_fits = n_combinations * cv
        # print(f"  Grid: {n_combinations} param combos x {cv} folds = {total_fits} fits")
        print("  Starting GridSearchCV ...")

        # t0 = time.time()
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )

    else:
        search_space = {
            "feature_selection__k": Integer(2, n_features),
            "knn__n_neighbors": Integer(1, n_samples // 10), 
            #"knn__n_neighbors": Integer(1, min(n_samples // 3, 100)), 

            "knn__weights": Categorical(["uniform", "distance"]),
            "knn__metric": Categorical(["euclidean", "manhattan", "chebyshev"]),
        }
        cv = 10
        n_iter = 150

        # n_combos = (n_features - 1) * (n_samples // 10) * 2 * 3
        # n_iter = min(150, n_combos)


        # print(f"  Search space: {list(search_space.keys())}")
        # print(f"  Bayes iterations: {n_iter} x {cv} folds = {n_iter * cv} fits")
        # print("  Starting BayesSearchCV ...")

        # t0 = time.time()
        search = BayesSearchCV(
            pipe,
            search_spaces=search_space,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            n_jobs=-1,
            #verbose=1,
            random_state=42,
        )

    search.fit(X_train, y_train)
    # elapsed = time.time() - t0

    # print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV score: {search.best_score_:.4f}")
    return search.best_estimator_, search


def plot_cv_splits(X_train, n_splits=5, figsize=(12, 4)):
    n_samples = len(X_train)
    kf = KFold(n_splits=n_splits)

    fig, ax = plt.subplots(figsize=figsize)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        ax.barh(fold_idx, len(train_idx[:val_idx[0]]),
                left=0, color="#185FA5", height=0.6)
        ax.barh(fold_idx, len(val_idx),
                left=val_idx[0], color="#D85A30", height=0.6)
        ax.barh(fold_idx, n_samples - val_idx[-1] - 1,
                left=val_idx[-1] + 1, color="#185FA5", height=0.6)
        ax.text(n_samples + 10, fold_idx,
                f"train {len(train_idx)} / val {len(val_idx)}",
                va="center", fontsize=9, color="gray")

    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_splits)])
    ax.set_xlabel("Sample index")
    ax.set_title(f"{n_splits}-Fold CV split — {n_samples} samples")
    ax.set_xlim(0, n_samples + 160)

    legend = [
        plt.Rectangle((0, 0), 1, 1, color="#185FA5", label="Train"),
        plt.Rectangle((0, 0), 1, 1, color="#D85A30", label="Validation"),
    ]
    ax.legend(handles=legend, loc="lower right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =========================
# 3. FEATURE IMPORTANCE (INTERPRETATION ONLY)
# =========================
def permutation_importance_report(model, X_val, y_val, feature_cols,
                                  scoring="f1", n_repeats=10, plot=True):
    print("\n" + "=" * 60)
    print("STEP 2/5: FEATURE IMPORTANCE (Permutation)")
    print("=" * 60)
    print("  Computing permutation importance ...")
    print(f"  n_repeats={n_repeats}, n_features={len(feature_cols)}, "
          f"X_val shape={X_val.shape}")
    print(f"  (each feature will be shuffled and re-scored {n_repeats}x)")

    # t0 = time.time()
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=42,
        n_jobs=-1
    )
    # elapsed = time.time() - t0
    # print(f"  Done in {elapsed:.1f}s")

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # print("\n  Permutation Importance table:")
    # print(importance_df.to_string(index=False))

    if plot:
        plot_feature_importance(importance_df, scoring=scoring)

    return importance_df


def plot_feature_importance(importance_df, scoring="f1", top_n=20):
    df = importance_df.copy()
    if top_n is not None:
        df = df.nlargest(top_n, "importance_mean")

    df = df.sort_values("importance_mean", ascending=True)

    colors = []
    for v in df["importance_mean"]:
        if v > 0.005:
            colors.append("#2ca02c")
        elif v < -0.005:
            colors.append("#d62728")
        else:
            colors.append("#999999")

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(df)), 6))
    ax.bar(
        df["feature"], df["importance_mean"],
        yerr=df["importance_std"],
        color=colors, edgecolor="black", linewidth=0.6,
        error_kw={"ecolor": "black", "capsize": 3, "alpha": 0.6}
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Permutation importance (drop in {scoring} when shuffled)")
    ax.set_title("Feature Importance (Permutation)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.show()


# =========================
# 4. EVALUATION
# =========================
def evaluate(model, X, y, title="SET", plot=False):
    k_selected = model.named_steps["feature_selection"].k
    print(
        f"\n  Evaluating on {title} set "
        f"(samples={X.shape[0]}, original_features={X.shape[1]}, "
        f"selected_features={k_selected}) ..."
    )

    t0 = time.time()
    preds = model.predict(X)
    elapsed = time.time() - t0
    # print(f"  Predict done in {elapsed:.1f}s")

    print(f"\n  ===== {title} =====")
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds))
    if plot:
        ConfusionMatrixDisplay.from_predictions(y, preds, cmap="winter_r", display_labels=['Normal', 'Accident'])
        plt.title(f"KNN Confusion Matrix - {title}")
        plt.show()


# =========================
# 5. FULL PIPELINE
# =========================
def run_knn_pipeline(X_train, y_train,
                     X_val, y_val,
                     X_test, y_test, feature_cols,
                     scoring="f1",
                     fast=True,
                     show_importance=True,
                     use_smote=False):
    """
    fast=True             -> small grid, fewer CV folds, fewer plots
    show_importance=False -> skip permutation importance entirely (fastest)
    use_smote=True        -> SMOTE inside CV folds (no leakage)
    use_smote=False       -> keine Oversampling (Baseline)
    """
    # t_global = time.time()
    print("\n" + "#" * 60)
    print("# KNN PIPELINE START")
    print(f"# SMOTE: {'ON' if use_smote else 'OFF'}")
    print("#" * 60)

    plot_cv_splits(X_train, n_splits=3 if fast else 10)

    # 1. TRAIN + CV SEARCH
    model, grid = tune_knn(X_train, y_train, scoring=scoring, fast=fast, use_smote=use_smote)

    # 2. FEATURE IMPORTANCE
    if show_importance:
        n_repeats = 3 if fast else 10
        _ = permutation_importance_report(
            model, X_val, y_val, feature_cols,
            scoring=scoring, n_repeats=n_repeats, plot=True
        )
    else:
        print("\n[STEP 2/5 skipped: show_importance=False]")

    # 3. TRAIN PERFORMANCE CHECK
    print("\n" + "=" * 60)
    print("STEP 3/5: TRAIN PERFORMANCE (overfit check)")
    print("=" * 60)
    evaluate(model, X_train, y_train, title="TRAIN")

    # 4. VALIDATION EVALUATION
    print("\n" + "=" * 60)
    print("STEP 4/5: VALIDATION EVALUATION")
    print("=" * 60)
    evaluate(model, X_val, y_val, title="VALIDATION")

    # 5. FINAL TEST
    print("\n" + "=" * 60)
    print("STEP 5/5: FINAL TEST EVALUATION")
    print("=" * 60)
    evaluate(model, X_test, y_test, title="TEST", plot=True)

    # total = time.time() - t_global
    print("\n" + "#" * 60)
    # print(f"# PIPELINE DONE in {total:.1f}s total")
    print(f"# PIPELINE DONE")
    print("#" * 60)

    return model, grid

