import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean = clean.drop_duplicates()
    if "Date" in clean.columns:
        clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
        clean["YearFromDate"] = clean["Date"].dt.year
        clean["MonthFromDate"] = clean["Date"].dt.month
    return clean


def infer_task_type(target_series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target_series):
        return "regression"
    return "classification"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    try:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
    except TypeError:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", categorical_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def select_features_by_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    min_abs_corr: float = 0.05,
    max_numeric_features: int = 8,
) -> Tuple[list[str], pd.DataFrame]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    correlations = []
    y_numeric = pd.to_numeric(y, errors="coerce")
    for col in numeric_features:
        series = pd.to_numeric(X[col], errors="coerce")
        pair = pd.concat([series, y_numeric], axis=1).dropna()
        if pair.empty or pair.iloc[:, 0].nunique() < 2:
            corr_value = np.nan
        else:
            corr_value = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        correlations.append(
            {
                "feature": col,
                "correlation": corr_value,
                "abs_correlation": abs(corr_value) if pd.notna(corr_value) else np.nan,
            }
        )

    corr_df = pd.DataFrame(correlations)
    if corr_df.empty:
        selected_numeric = []
    else:
        ranked = corr_df.dropna(subset=["abs_correlation"]).sort_values(
            by="abs_correlation", ascending=False
        )
        selected_numeric = ranked.loc[
            ranked["abs_correlation"] >= min_abs_corr, "feature"
        ].tolist()
        if not selected_numeric:
            selected_numeric = ranked.head(max_numeric_features)["feature"].tolist()
        elif len(selected_numeric) > max_numeric_features:
            selected_numeric = selected_numeric[:max_numeric_features]

    if not selected_numeric and numeric_features:
        selected_numeric = numeric_features[
            : min(max_numeric_features, len(numeric_features))
        ]

    selected_features = selected_numeric + categorical_features
    if corr_df.empty:
        corr_df = pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])
    else:
        corr_df = corr_df.sort_values(by="abs_correlation", ascending=False)

    return selected_features, corr_df


def get_models(task_type: str) -> Dict[str, object]:
    if task_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        }
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=160,
            max_depth=16,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
) -> Tuple[float, float]:
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    if task_type == "regression":
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return r2, rmse
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    return acc, f1


def tune_model(
    pipeline: Pipeline,
    model_name: str,
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    if task_type == "regression" and model_name == "RandomForestRegressor":
        params = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        }
    elif task_type == "classification" and model_name == "RandomForestClassifier":
        params = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        }
    else:
        return pipeline

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=4,
        cv=3,
        random_state=42,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def cross_validation_score(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, task_type: str
) -> float:
    if task_type == "regression":
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    else:
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_weighted")
    return float(scores.mean())


def write_metrics_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
