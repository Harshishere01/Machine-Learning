"""
Reproducible state-grouped cross-validation for the NFHS-5 child-marriage project.

Run from the Project-C folder:
    python src/grouped_cv.py --data data/datafile.csv --output results

The target is district-level prevalence of women aged 20–24 married before age 18.
GroupKFold keeps all districts from a State/UT in exactly one fold.
All learned preprocessing is fitted inside each fold's sklearn Pipeline.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


TARGET = "Women age 20-24 years married before age 18 years (%)"
DISTRICT_COL = "District Names"
STATE_COL = "State/UT"
SEED = 42


def numeric_series(series: pd.Series) -> pd.Series:
    """Convert numeric strings; parenthesised estimates retain their value; '*' becomes NaN."""
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    cleaned = cleaned.replace({"*": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def load_and_prepare(path: Path):
    df = pd.read_csv(path)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    required = {TARGET, DISTRICT_COL, STATE_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["District"] = df[DISTRICT_COL].astype(str).str.strip()
    df["State"] = (
        df[STATE_COL].astype(str).str.strip()
        .replace({"Maharastra": "Maharashtra"})
    )

    # Convert all non-geographic columns to numeric.
    excluded = {DISTRICT_COL, STATE_COL, "District", "State"}
    numeric_cols = [c for c in df.columns if c not in excluded]
    for col in numeric_cols:
        df[col] = numeric_series(df[col])

    if df[TARGET].isna().any():
        raise ValueError(
            f"Target contains {int(df[TARGET].isna().sum())} missing/non-numeric rows. "
            "Resolve target missingness explicitly before running CV."
        )

    # Mirror the submitted project's feature-screening rules:
    # drop columns with >40% missingness, sample-size/count columns, and the target.
    missing_pct = df[numeric_cols].isna().mean()
    dropped_missing = [c for c in numeric_cols if missing_pct[c] > 0.40]
    sample_size_cols = [c for c in numeric_cols if c.startswith("Number of")]
    feature_cols = [
        c for c in numeric_cols
        if c not in dropped_missing + sample_size_cols + [TARGET]
    ]

    if not feature_cols:
        raise ValueError("No predictors remain after feature screening.")

    X = df[feature_cols].copy()
    y = df[TARGET].astype(float).to_numpy()
    groups = df["State"].to_numpy()

    audit = {
        "rows": int(len(df)),
        "states": int(pd.Series(groups).nunique()),
        "features_used": int(len(feature_cols)),
        "features_dropped_over_40pct_missing": dropped_missing,
        "sample_size_columns_excluded": sample_size_cols,
        "target": TARGET,
        "group_column": "State",
        "missing_predictor_cells_percent_before_imputation": float(X.isna().mean().mean() * 100),
        "districts_per_state_min": int(pd.Series(groups).value_counts().min()),
        "districts_per_state_max": int(pd.Series(groups).value_counts().max()),
        "state_name_normalisation": {"Maharastra": "Maharashtra"},
    }
    return X, y, groups, feature_cols, audit


def make_models(feature_cols):
    # Imputation and scaling are inside each estimator pipeline, so each fold
    # learns medians/scaling parameters from its training states only.
    numeric_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), feature_cols)
        ],
        remainder="drop",
    )
    return {
        "Ridge": Pipeline([
            ("preprocess", numeric_preprocessor),
            ("model", Ridge(alpha=30.0)),
        ]),
        "SVR_RBF": Pipeline([
            ("preprocess", numeric_preprocessor),
            ("model", SVR(kernel="rbf", C=10.0, epsilon=1.0, gamma="scale")),
        ]),
        "Random_Forest": Pipeline([
            ("preprocess", numeric_preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=3,
                random_state=SEED,
                n_jobs=-1,
            )),
        ]),
    }


def regression_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def run_grouped_cv(X, y, groups, models, output_dir: Path):
    n_groups = pd.Series(groups).nunique()
    n_splits = min(5, n_groups)
    if n_splits < 2:
        raise ValueError("GroupKFold requires at least two distinct groups.")

    cv = GroupKFold(n_splits=n_splits)
    fold_rows, result_rows, prediction_frames = [], [], []

    # OOF predictions are filled exactly once per row.
    for model_name, estimator in models.items():
        oof = np.full(len(y), np.nan, dtype=float)

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            overlap = train_groups.intersection(test_groups)
            if overlap:
                raise RuntimeError(f"Group leakage in fold {fold}: {sorted(overlap)}")

            fitted = estimator.fit(X.iloc[train_idx], y[train_idx])
            pred = fitted.predict(X.iloc[test_idx])
            oof[test_idx] = pred

            m = regression_metrics(y[test_idx], pred)
            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_states": int(len(train_groups)),
                "test_states": int(len(test_groups)),
                "state_overlap": int(len(overlap)),
                **m,
            })

        if np.isnan(oof).any():
            raise RuntimeError(f"Some rows did not receive OOF predictions for {model_name}.")

        pooled = regression_metrics(y, oof)
        result_rows.append({
            "model": model_name,
            "n_splits": n_splits,
            "evaluation": "pooled out-of-fold predictions; each state held out once",
            **pooled,
        })
        prediction_frames.append(pd.DataFrame({
            "row_index": np.arange(len(y)),
            "state": groups,
            "actual": y,
            "predicted": oof,
            "model": model_name,
        }))

    # Baseline: predict the mean target among the training states in each fold.
    baseline_oof = np.full(len(y), np.nan, dtype=float)
    for _, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        baseline_oof[test_idx] = float(np.mean(y[train_idx]))
    baseline_metrics = regression_metrics(y, baseline_oof)
    result_rows.append({
        "model": "Training_fold_mean_baseline",
        "n_splits": n_splits,
        "evaluation": "pooled out-of-fold predictions; each state held out once",
        **baseline_metrics,
    })
    prediction_frames.append(pd.DataFrame({
        "row_index": np.arange(len(y)),
        "state": groups,
        "actual": y,
        "predicted": baseline_oof,
        "model": "Training_fold_mean_baseline",
    }))

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output_dir / "grouped_cv_fold_metrics.csv", index=False)
    pd.DataFrame(result_rows).sort_values("RMSE").to_csv(
        output_dir / "grouped_cv_results.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "grouped_cv_predictions.csv", index=False
    )

    return pd.DataFrame(result_rows).sort_values("RMSE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/datafile.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    X, y, groups, feature_cols, audit = load_and_prepare(args.data)
    models = make_models(feature_cols)
    results = run_grouped_cv(X, y, groups, models, args.output)

    audit["n_splits"] = int(min(5, pd.Series(groups).nunique()))
    audit["models_evaluated"] = list(models.keys())
    audit["baseline"] = "training-fold target mean"
    with (args.output / "grouped_cv_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("\nGrouped cross-validation results (lower MAE/RMSE is better):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved outputs to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
