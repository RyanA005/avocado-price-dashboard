import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
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
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def get_models(task_type: str) -> Dict[str, object]:
    if task_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        }
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
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


def cross_validation_score(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, task_type: str) -> float:
    if task_type == "regression":
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    else:
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_weighted")
    return float(scores.mean())


def write_metrics_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

