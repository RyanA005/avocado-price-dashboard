import argparse
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_core import (
    build_preprocessor,
    clean_dataframe,
    cross_validation_score,
    evaluate_pipeline,
    get_models,
    infer_task_type,
    tune_model,
    write_metrics_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/avocado.csv")
    parser.add_argument("--target", type=str, default="AveragePrice")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--sample_size", type=int, default=6000)
    parser.add_argument(
        "--full_eval",
        action="store_true",
        help="Enable hyperparameter tuning and k-fold CV (slower).",
    )
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)
    df = clean_dataframe(df)
    df = df.dropna(subset=[args.target])
    if args.sample_size and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42)

    X = df.drop(columns=[args.target])
    y = df[args.target]
    task_type = infer_task_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    models = get_models(task_type)

    results = []
    best_pipeline = None
    best_score = float("-inf")

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        if args.full_eval:
            trained_pipeline = tune_model(pipeline, model_name, task_type, X_train, y_train)
            cv = cross_validation_score(trained_pipeline, X, y, task_type)
        else:
            trained_pipeline = pipeline
            cv = float("nan")

        m1, m2 = evaluate_pipeline(trained_pipeline, X_train, X_test, y_train, y_test, task_type)

        if task_type == "regression":
            metric_1_name, metric_2_name = "R2", "RMSE"
            selection_score = m1
        else:
            metric_1_name, metric_2_name = "Accuracy", "F1"
            selection_score = m2

        results.append(
            {
                "model": model_name,
                metric_1_name: m1,
                metric_2_name: m2,
                "cv_score": cv,
            }
        )

        if selection_score > best_score:
            best_score = selection_score
            best_pipeline = trained_pipeline

    report_df = pd.DataFrame(results).sort_values(by="cv_score", ascending=False)
    report_path = os.path.join(args.artifacts_dir, "model_report.csv")
    report_df.to_csv(report_path, index=False)

    model_path = os.path.join(args.artifacts_dir, "best_model.joblib")
    joblib.dump(best_pipeline, model_path)

    metrics_payload = {
        "task_type": task_type,
        "target": args.target,
        "best_model": report_df.iloc[0]["model"],
        "report_path": report_path,
        "model_path": model_path,
    }
    write_metrics_json(os.path.join(args.artifacts_dir, "metrics_summary.json"), metrics_payload)

    print(report_df)
    print(f"\nSaved best model to: {model_path}")


if __name__ == "__main__":
    main()

