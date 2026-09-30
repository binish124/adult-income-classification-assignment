"""Train and compare two models for the supplied Adult Income CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "adult_raw.csv"
MODEL_DIR = ROOT / "models"
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "marital_status", "occupation"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "income"
LABELS = {"<=50K": 0, ">50K": 1}


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    data = pd.read_csv(path, skipinitialspace=True)
    data.columns = data.columns.str.strip()
    required = set(FEATURES + [TARGET])
    missing_columns = required - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Convert Adult's literal '?' markers and empty strings to true missing values.
    for col in CATEGORICAL + [TARGET]:
        data[col] = data[col].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
    for col in CATEGORICAL:
        data[col] = data[col].astype(object).where(data[col].notna(), np.nan)
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[TARGET] = data[TARGET].str.rstrip(".")
    unknown_labels = set(data[TARGET].dropna().unique()) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Unexpected income labels: {sorted(unknown_labels)}")
    missing_target = int(data[TARGET].isna().sum())
    if missing_target:
        data = data.dropna(subset=[TARGET]).copy()
    if data.empty:
        raise ValueError("No labeled rows remain after cleaning")

    summary = {
        "source_rows": int(len(data) + missing_target),
        "labeled_rows": int(len(data)),
        "missing_target_rows_removed": missing_target,
        "missing_by_feature": {col: int(data[col].isna().sum()) for col in FEATURES},
        "class_counts": {label: int((data[TARGET] == label).sum()) for label in LABELS},
    }
    return data[FEATURES], data[TARGET].map(LABELS).astype(int), summary


def make_pipeline(model_name: str) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=150, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])


def metrics(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    prediction = model.predict(x)
    return {
        "accuracy": round(float(accuracy_score(y, prediction)), 4),
        "precision": round(float(precision_score(y, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y, prediction, zero_division=0)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    x, y, data_summary = load_data(args.data)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=42,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42,
    )
    models = {name: make_pipeline(name) for name in ("logistic_regression", "random_forest")}
    for model in models.values():
        model.fit(x_train, y_train)
    validation = {name: metrics(model, x_val, y_val) for name, model in models.items()}
    selected_name = max(validation, key=lambda name: validation[name]["f1"])
    # Final refit uses training and validation rows; the untouched test set is evaluated once.
    x_final = pd.concat([x_train, x_val])
    y_final = pd.concat([y_train, y_val])
    final_model = make_pipeline(selected_name).fit(x_final, y_final)
    test = metrics(final_model, x_test, y_test)

    # Model-agnostic feature impact: average change in test F1 when one input is shuffled.
    baseline_f1 = test["f1"]
    rng = np.random.default_rng(42)
    impacts = {}
    for feature in FEATURES:
        shuffled_scores = []
        for _ in range(3):
            permuted = x_test.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            shuffled_scores.append(f1_score(y_test, final_model.predict(permuted), zero_division=0))
        impacts[feature] = round(float(baseline_f1 - np.mean(shuffled_scores)), 4)
    impacts = dict(sorted(impacts.items(), key=lambda item: item[1], reverse=True))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "model.joblib")
    report = {
        "dataset": data_summary,
        "split": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "validation_metrics": validation,
        "selection_metric": "validation F1 for the >50K class",
        "selected_model": selected_name,
        "test_metrics": test,
        "test_majority_class_accuracy": round(float((y_test == 0).mean()), 4),
        "feature_permutation_f1_drop": impacts,
        "positive_class": ">50K",
        "random_state": 42,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
