from __future__ import annotations

import base64
import io
import logging
from typing import Iterable

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, ctx, dcc, html
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.ml_core import (
    build_preprocessor,
    clean_dataframe,
    evaluate_pipeline,
    get_models,
    infer_task_type,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_DATA_PATH = "data/avocado.csv"
EXCLUDED_FEATURES = {"id", "Date"}

CURRENT_DF = pd.DataFrame()
CURRENT_DATASET_NAME = "No dataset loaded"
CURRENT_TARGET = None
TRAINED_MODEL = None
TRAINED_FEATURES: list[str] = []
TRAINED_TARGET = None
TRAINED_TASK_TYPE = None
LAST_TRAIN_MESSAGE = ""
LAST_PREDICTION_MESSAGE = ""
UPLOAD_MESSAGE = "Upload a CSV file to replace the current dataset."


def load_dataframe(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return clean_dataframe(frame)


def set_current_dataframe(frame: pd.DataFrame, dataset_name: str) -> None:
    global CURRENT_DF, CURRENT_DATASET_NAME, CURRENT_TARGET
    global TRAINED_MODEL, TRAINED_FEATURES, TRAINED_TARGET, TRAINED_TASK_TYPE
    global LAST_TRAIN_MESSAGE, LAST_PREDICTION_MESSAGE

    CURRENT_DF = frame.copy()
    CURRENT_DATASET_NAME = dataset_name
    CURRENT_TARGET = None
    TRAINED_MODEL = None
    TRAINED_FEATURES = []
    TRAINED_TARGET = None
    TRAINED_TASK_TYPE = None
    LAST_TRAIN_MESSAGE = ""
    LAST_PREDICTION_MESSAGE = ""


def load_initial_dataset() -> None:
    try:
        set_current_dataframe(load_dataframe(DEFAULT_DATA_PATH), "Bundled dataset")
    except Exception:
        set_current_dataframe(pd.DataFrame(), "No dataset loaded")


def parse_uploaded_file(contents: str, filename: str | None) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    if filename and filename.lower().endswith(".csv"):
        frame = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    else:
        frame = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return clean_dataframe(frame)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and column not in EXCLUDED_FEATURES
    ]


def categorical_columns(frame: pd.DataFrame, target: str | None) -> list[str]:
    return [
        column
        for column in frame.columns
        if column != target
        and column not in EXCLUDED_FEATURES
        and not pd.api.types.is_numeric_dtype(frame[column])
    ]


def feature_columns(frame: pd.DataFrame, target: str | None) -> list[str]:
    return [
        column
        for column in frame.columns
        if column != target and column not in EXCLUDED_FEATURES
    ]


def option_list(columns: Iterable[str]) -> list[dict]:
    return [{"label": column, "value": column} for column in columns]


def empty_figure(title: str, message: str) -> dict:
    figure = px.bar(title=title)
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.update_layout(template="plotly_white")
    return figure


def build_target_figure(frame: pd.DataFrame, target: str, category: str | None):
    if not target or target not in frame.columns:
        return empty_figure(
            "Average Target by Category", "Select a valid target column."
        )
    if not category or category not in frame.columns:
        return empty_figure(
            "Average Target by Category",
            "Select a categorical variable for the first chart.",
        )

    chart_df = (
        frame[[category, target]]
        .dropna()
        .groupby(category, as_index=False)[target]
        .mean()
        .sort_values(by=target, ascending=False)
    )
    if chart_df.empty:
        return empty_figure(
            "Average Target by Category", "No valid rows found for this chart."
        )

    figure = px.bar(
        chart_df,
        x=category,
        y=target,
        title=f"Average {target} by {category}",
    )
    figure.update_layout(
        template="plotly_white", xaxis_title=category, yaxis_title=f"Average {target}"
    )
    return figure


def build_correlation_figure(frame: pd.DataFrame, target: str):
    if not target or target not in frame.columns:
        return empty_figure(
            "Absolute Correlation Strength", "Select a valid target column."
        )
    if not pd.api.types.is_numeric_dtype(frame[target]):
        return empty_figure(
            "Absolute Correlation Strength",
            "The selected target must be numerical.",
        )

    numeric_cols = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and column != target
    ]
    if not numeric_cols:
        return empty_figure(
            "Absolute Correlation Strength",
            "No additional numerical columns are available.",
        )

    records = []
    for column in numeric_cols:
        pair = frame[[column, target]].dropna()
        if len(pair) < 2 or pair[column].nunique() < 2:
            continue
        corr_value = pair[column].corr(pair[target])
        if pd.notna(corr_value):
            records.append(
                {
                    "feature": column,
                    "abs_correlation": abs(float(corr_value)),
                }
            )

    if not records:
        return empty_figure(
            "Absolute Correlation Strength",
            "Correlation could not be computed for the selected target.",
        )

    corr_df = pd.DataFrame(records).sort_values(by="abs_correlation", ascending=False)
    figure = px.bar(
        corr_df,
        x="feature",
        y="abs_correlation",
        title=f"Absolute Correlation With {target}",
    )
    figure.update_layout(
        template="plotly_white",
        xaxis_title="Feature",
        yaxis_title="Absolute correlation",
    )
    return figure


def build_prediction_placeholder(selected_features: list[str]) -> str:
    if not selected_features:
        return "Select features first, then enter values in the same order separated by commas."
    return "Enter values in this order: " + ", ".join(selected_features)


def build_model_metric_cards(
    frame: pd.DataFrame,
    target: str | None,
    selected_features: list[str],
) -> tuple[list[html.Div], list[dict], str, str]:
    if (
        frame.empty
        or not target
        or target not in frame.columns
        or not selected_features
    ):
        return [], [], "None", "Select features and a target to view model metrics."

    cleaned_features = [
        feature
        for feature in selected_features
        if feature in frame.columns and feature != target
    ]
    if not cleaned_features:
        return [], [], "None", "Select at least one usable feature to view model metrics."

    y = frame[target].copy()
    task_type = infer_task_type(y)
    X = frame[cleaned_features].copy()

    if len(frame) < 10:
        return [], [], task_type, "Dataset is too small to estimate model metrics."

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    preprocessor = build_preprocessor(X_train)
    models = get_models(task_type)

    cards = []
    options = []
    for model_name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        metric_1, metric_2 = evaluate_pipeline(
            pipeline, X_train, X_test, y_train, y_test, task_type
        )
        if task_type == "regression":
            label = f"{model_name} - R^2: {metric_1:.4f}, RMSE: {metric_2:.4f}"
            subtitle = f"R^2: {metric_1:.4f} | RMSE: {metric_2:.4f}"
        else:
            label = f"{model_name} - Accuracy: {metric_1:.4f}, F1: {metric_2:.4f}"
            subtitle = f"Accuracy: {metric_1:.4f} | F1: {metric_2:.4f}"
        cards.append(
            html.Div(
                [
                    html.Div(
                        model_name, style={"fontWeight": "700", "marginBottom": "4px"}
                    ),
                    html.Div(
                        subtitle, style={"color": "#475569", "fontSize": "0.94rem"}
                    ),
                ],
                style={
                    "padding": "10px 12px",
                    "border": "1px solid #dbe3ee",
                    "borderRadius": "12px",
                    "background": "#f8fafc",
                    "marginBottom": "8px",
                },
            )
        )
        options.append({"label": label, "value": model_name})

    return cards, options, task_type, ""


def parse_prediction_values(
    raw_value: str | None, selected_features: list[str], frame: pd.DataFrame
):
    if not raw_value or not raw_value.strip():
        return None, "Enter comma-separated values for all selected features."

    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != len(selected_features):
        return None, (
            f"Expected {len(selected_features)} values, but received {len(parts)}. "
            "Match the selected feature order exactly."
        )

    parsed_row = {}
    for feature, raw_part in zip(selected_features, parts):
        if feature not in frame.columns:
            return None, f"Unknown feature in prediction input: {feature}"
        if pd.api.types.is_numeric_dtype(frame[feature]):
            try:
                parsed_row[feature] = float(raw_part)
            except ValueError:
                return None, f"Feature '{feature}' must be numeric."
        else:
            if not raw_part:
                return None, f"Feature '{feature}' cannot be empty."
            parsed_row[feature] = raw_part

    return parsed_row, ""


load_initial_dataset()

app = Dash(__name__)
app.title = "Dataset ML Studio"
server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Dataset ML Studio", style={"marginBottom": "6px"}),
                html.Div(
                    "Upload a dataset, choose a numerical target, inspect the bars, train a model, and predict from the trained pipeline.",
                    style={"color": "#5b6472", "fontSize": "0.98rem"},
                ),
            ],
            style={"marginBottom": "18px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Upload", style={"marginTop": "0"}),
                        dcc.Upload(
                            id="dataset-upload",
                            children=html.Div(
                                ["Drag and drop or ", html.A("select a CSV file")]
                            ),
                            style={
                                "width": "60%",
                                "padding": "22px",
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderRadius": "14px",
                                "textAlign": "center",
                                "background": "#f8fafc",
                                "cursor": "pointer",
                            },
                            multiple=False,
                        ),
                        html.Div(
                            id="upload-status",
                            style={
                                "marginTop": "10px",
                                "color": "#334155",
                            },
                        ),
                    ],
                    style={
                        "padding": "18px",
                        "border": "1px solid #dbe3ee",
                        "borderRadius": "16px",
                        "background": "white",
                    },
                ),
                html.Div(
                    [
                        html.H3("Select Target", style={"marginTop": "0"}),
                        html.Label("Target variable"),
                        dcc.Dropdown(
                            id="target-dropdown",
                            options=option_list(numeric_columns(CURRENT_DF)),
                            value=(
                                numeric_columns(CURRENT_DF)[0]
                                if numeric_columns(CURRENT_DF)
                                else None
                            ),
                            clearable=False,
                        ),
                        html.Div(
                            "The dropdown only lists numerical columns from the current dataset.",
                            style={
                                "marginTop": "8px",
                                "fontSize": "0.9rem",
                                "color": "#5b6472",
                            },
                        ),
                    ],
                    style={
                        "padding": "18px",
                        "border": "1px solid #dbe3ee",
                        "borderRadius": "16px",
                        "background": "white",
                    },
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px",
                "marginBottom": "16px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Barcharts", style={"marginTop": "0"}),
                        html.Div(
                            "Category variable",
                            style={"marginBottom": "6px", "fontWeight": "600"},
                        ),
                        dcc.RadioItems(
                            id="category-radio",
                            options=option_list(
                                categorical_columns(CURRENT_DF, CURRENT_TARGET)
                            ),
                            value=(
                                categorical_columns(CURRENT_DF, CURRENT_TARGET)[0]
                                if categorical_columns(CURRENT_DF, CURRENT_TARGET)
                                else None
                            ),
                            inline=True,
                            style={"marginBottom": "14px"},
                        ),
                        dcc.Graph(
                            id="category-bar-chart",
                            figure=build_target_figure(
                                CURRENT_DF,
                                CURRENT_TARGET
                                or (
                                    numeric_columns(CURRENT_DF)[0]
                                    if numeric_columns(CURRENT_DF)
                                    else None
                                ),
                                (
                                    categorical_columns(
                                        CURRENT_DF,
                                        CURRENT_TARGET
                                        or (
                                            numeric_columns(CURRENT_DF)[0]
                                            if numeric_columns(CURRENT_DF)
                                            else None
                                        ),
                                    )[0]
                                    if categorical_columns(
                                        CURRENT_DF,
                                        CURRENT_TARGET
                                        or (
                                            numeric_columns(CURRENT_DF)[0]
                                            if numeric_columns(CURRENT_DF)
                                            else None
                                        ),
                                    )
                                    else None
                                ),
                            ),
                        ),
                    ],
                    style={
                        "padding": "18px",
                        "border": "1px solid #dbe3ee",
                        "borderRadius": "16px",
                        "background": "white",
                    },
                ),
                html.Div(
                    [
                        html.H3("Correlation", style={"marginTop": "0"}),
                        dcc.Graph(
                            id="corr-bar-chart",
                            figure=build_correlation_figure(
                                CURRENT_DF,
                                CURRENT_TARGET
                                or (
                                    numeric_columns(CURRENT_DF)[0]
                                    if numeric_columns(CURRENT_DF)
                                    else None
                                ),
                            ),
                        ),
                    ],
                    style={
                        "padding": "18px",
                        "border": "1px solid #dbe3ee",
                        "borderRadius": "16px",
                        "background": "white",
                    },
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px",
                "marginBottom": "16px",
            },
        ),
        html.Div(
            [
                html.H3("Train", style={"marginTop": "0"}),
                html.Div(
                    "Choose the features to include in the regression pipeline. Categorical columns are one-hot encoded and missing values are handled automatically.",
                    style={"marginBottom": "10px", "color": "#5b6472"},
                ),
                dcc.Checklist(
                    id="feature-checklist",
                    options=option_list(feature_columns(CURRENT_DF, CURRENT_TARGET)),
                    value=feature_columns(CURRENT_DF, CURRENT_TARGET),
                    labelStyle={
                        "display": "inline-block",
                        "marginRight": "14px",
                        "marginBottom": "8px",
                    },
                    inputStyle={"marginRight": "6px"},
                ),
                html.Div(
                    [
                        html.Div(
                            "Model selection",
                            style={
                                "marginTop": "10px",
                                "marginBottom": "6px",
                                "fontWeight": "600",
                            },
                        ),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[],
                            value=None,
                            clearable=False,
                            placeholder="Model metrics will populate here",
                        ),
                        html.Div(id="model-metrics", style={"marginTop": "12px"}),
                    ],
                    style={"marginTop": "8px"},
                ),
                html.Button(
                    "Train Model",
                    id="train-btn",
                    n_clicks=0,
                    style={
                        "marginTop": "12px",
                        "padding": "10px 18px",
                        "border": "0",
                        "borderRadius": "10px",
                        "background": "#0f172a",
                        "color": "white",
                        "cursor": "pointer",
                    },
                ),
                html.Div(
                    id="train-status",
                    children="Status: idle",
                    style={
                        "marginTop": "10px",
                        "color": "#475569",
                        "fontSize": "0.95rem",
                    },
                ),
                html.Div(
                    id="train-output", style={"marginTop": "10px", "fontWeight": "600"}
                ),
            ],
            style={
                "padding": "18px",
                "border": "1px solid #dbe3ee",
                "borderRadius": "16px",
                "background": "white",
                "marginBottom": "16px",
            },
        ),
        html.Div(
            [
                html.H3("Predict", style={"marginTop": "0"}),
                html.Div(
                    "Enter feature values as a comma-separated list using the same order as the selected features above.",
                    style={"marginBottom": "10px", "color": "#5b6472"},
                ),
                html.Div(
                    [
                        dcc.Input(
                            id="prediction-input",
                            type="text",
                            placeholder=build_prediction_placeholder(
                                feature_columns(CURRENT_DF, CURRENT_TARGET)
                            ),
                            style={
                                "flex": "1",
                                "padding": "10px 12px",
                                "borderRadius": "10px",
                                "border": "1px solid #cbd5e1",
                            },
                        ),
                        html.Button(
                            "Predict",
                            id="predict-btn",
                            n_clicks=0,
                            style={
                                "padding": "10px 18px",
                                "border": "0",
                                "borderRadius": "10px",
                                "background": "#0f172a",
                                "color": "white",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(
                            id="prediction-output",
                            style={
                                "alignSelf": "center",
                                "fontWeight": "700",
                                "marginLeft": "10px",
                            },
                        ),
                    ],
                    style={"display": "flex", "gap": "10px", "alignItems": "center"},
                ),
            ],
            style={
                "padding": "18px",
                "border": "1px solid #dbe3ee",
                "borderRadius": "16px",
                "background": "white",
            },
        ),
    ],
    style={
        "maxWidth": "1260px",
        "margin": "24px auto",
        "padding": "0 18px 28px",
        "fontFamily": "Arial, sans-serif",
        "background": "#eef3f8",
        "minHeight": "100vh",
    },
)


@app.callback(
    Output("upload-status", "children"),
    Output("target-dropdown", "options"),
    Output("target-dropdown", "value"),
    Output("category-radio", "options"),
    Output("category-radio", "value"),
    Output("category-bar-chart", "figure"),
    Output("corr-bar-chart", "figure"),
    Output("feature-checklist", "options"),
    Output("feature-checklist", "value"),
    Output("model-dropdown", "options"),
    Output("model-dropdown", "value"),
    Output("model-metrics", "children"),
    Output("prediction-input", "placeholder"),
    Input("dataset-upload", "contents"),
    Input("target-dropdown", "value"),
    Input("category-radio", "value"),
    State("dataset-upload", "filename"),
    State("feature-checklist", "value"),
)
def refresh_view(contents, target_value, category_value, filename, selected_features):
    global CURRENT_TARGET

    triggered = ctx.triggered_id
    if triggered == "dataset-upload" and contents:
        try:
            uploaded_frame = parse_uploaded_file(contents, filename)
            dataset_name = filename or "Uploaded dataset"
            set_current_dataframe(uploaded_frame, dataset_name)
        except Exception as exc:
            return (
                f"Upload failed: {exc}",
                option_list(numeric_columns(CURRENT_DF)),
                CURRENT_TARGET,
                option_list(categorical_columns(CURRENT_DF, CURRENT_TARGET)),
                category_value,
                build_target_figure(CURRENT_DF, CURRENT_TARGET, category_value),
                build_correlation_figure(CURRENT_DF, CURRENT_TARGET),
                option_list(feature_columns(CURRENT_DF, CURRENT_TARGET)),
                feature_columns(CURRENT_DF, CURRENT_TARGET),
                [],
                None,
                [],
                build_prediction_placeholder(
                    feature_columns(CURRENT_DF, CURRENT_TARGET)
                ),
            )

    frame = CURRENT_DF
    numeric_cols = numeric_columns(frame)
    if not numeric_cols:
        CURRENT_TARGET = None
        target_options = []
        resolved_target = None
    else:
        target_options = option_list(numeric_cols)
        resolved_target = (
            target_value if target_value in numeric_cols else numeric_cols[0]
        )
        CURRENT_TARGET = resolved_target

    categorical_cols = categorical_columns(frame, resolved_target)
    category_options = option_list(categorical_cols)
    resolved_category = (
        category_value
        if category_value in categorical_cols
        else (categorical_cols[0] if categorical_cols else None)
    )

    feature_cols = feature_columns(frame, resolved_target)
    feature_options = option_list(feature_cols)
    if selected_features is None:
        resolved_features = feature_cols
    else:
        resolved_features = [
            feature for feature in selected_features if feature in feature_cols
        ]

    upload_message = f"Loaded {CURRENT_DATASET_NAME}. Rows: {len(frame):,}, columns: {len(frame.columns):,}."
    target_figure = build_target_figure(frame, resolved_target, resolved_category)
    corr_figure = build_correlation_figure(frame, resolved_target)
    model_cards, model_options, task_type, model_message = build_model_metric_cards(
        frame, resolved_target, resolved_features
    )
    model_value = model_options[0]["value"] if model_options else None
    placeholder = build_prediction_placeholder(resolved_features)

    return (
        upload_message,
        target_options,
        resolved_target,
        category_options,
        resolved_category,
        target_figure,
        corr_figure,
        feature_options,
        resolved_features,
        model_options,
        model_value,
        (
            model_cards
            if model_cards
            else (
                [html.Div(model_message, style={"color": "#475569"})]
                if model_message
                else []
            )
        ),
        placeholder,
    )


@app.callback(
    Output("train-output", "children"),
    Input("train-btn", "n_clicks"),
    State("target-dropdown", "value"),
    State("feature-checklist", "value"),
    State("model-dropdown", "value"),
    running=[
        (Output("train-status", "children"), "Status: training...", "Status: idle")
    ],
    prevent_initial_call=True,
)
def train_model(n_clicks, target_value, selected_features, selected_model):
    global TRAINED_MODEL, TRAINED_FEATURES, TRAINED_TARGET, TRAINED_TASK_TYPE
    global LAST_TRAIN_MESSAGE, LAST_PREDICTION_MESSAGE

    logger.info(
        "Train clicked: n_clicks=%s target=%s features=%s",
        n_clicks,
        target_value,
        selected_features,
    )
    try:
        logger.info("Train stage: start")
        if CURRENT_DF.empty:
            LAST_TRAIN_MESSAGE = "Upload a dataset before training."
            logger.info("Train result: %s", LAST_TRAIN_MESSAGE)
            return LAST_TRAIN_MESSAGE
        if not target_value or target_value not in CURRENT_DF.columns:
            LAST_TRAIN_MESSAGE = "Choose a valid numerical target before training."
            logger.info("Train result: %s", LAST_TRAIN_MESSAGE)
            return LAST_TRAIN_MESSAGE
        if not selected_features:
            LAST_TRAIN_MESSAGE = "Select at least one feature before training."
            logger.info("Train result: %s", LAST_TRAIN_MESSAGE)
            return LAST_TRAIN_MESSAGE

        cleaned_features = [
            feature
            for feature in selected_features
            if feature in CURRENT_DF.columns and feature != target_value
        ]
        logger.info("Train stage: cleaned_features=%s", cleaned_features)
        if not cleaned_features:
            LAST_TRAIN_MESSAGE = (
                "Selected features do not contain any usable predictors."
            )
            logger.info("Train result: %s", LAST_TRAIN_MESSAGE)
            return LAST_TRAIN_MESSAGE

        X = CURRENT_DF[cleaned_features].copy()
        y = CURRENT_DF[target_value].copy()
        task_type = infer_task_type(y)
        logger.info(
            "Train stage: task_type=%s rows=%s cols=%s",
            task_type,
            len(X),
            len(X.columns),
        )
        if task_type != "regression":
            LAST_TRAIN_MESSAGE = "The selected target is not numerical. Choose a numerical target for regression."
            logger.info("Train result: %s", LAST_TRAIN_MESSAGE)
            return LAST_TRAIN_MESSAGE

        if len(CURRENT_DF) < 10:
            LAST_TRAIN_MESSAGE = "Dataset is too small to train a stable model."
            logger.info("Train result: %s", LAST_TRAIN_MESSAGE)
            return LAST_TRAIN_MESSAGE

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
        logger.info(
            "Train stage: split complete train_rows=%s test_rows=%s",
            len(X_train),
            len(X_test),
        )

        preprocessor = build_preprocessor(X_train)
        models = get_models(task_type)
        if selected_model not in models:
            selected_model = next(iter(models))
        logger.info("Train stage: selected_model=%s", selected_model)
        logger.info("Train stage: models=%s", list(models.keys()))
        best_pipeline = None
        best_score = float("-inf")
        best_model_name = ""

        for model_name, model in models.items():
            logger.info("Train stage: evaluating %s", model_name)
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            score, _ = evaluate_pipeline(
                pipeline, X_train, X_test, y_train, y_test, task_type
            )
            logger.info("Train stage: %s score=%s", model_name, score)
            if model_name == selected_model:
                best_score = score
                best_model_name = model_name
                best_pipeline = pipeline
            elif best_pipeline is None and score > best_score:
                best_score = score
                best_model_name = model_name
                best_pipeline = pipeline

        if best_pipeline is None:
            LAST_TRAIN_MESSAGE = "Training failed to produce a valid model."
            logger.info("Train result: %s", LAST_TRAIN_MESSAGE)
            return LAST_TRAIN_MESSAGE

        logger.info(
            "Train stage: fitting best model=%s on full dataset", best_model_name
        )
        best_pipeline.fit(X, y)
        TRAINED_MODEL = best_pipeline
        TRAINED_FEATURES = cleaned_features
        TRAINED_TARGET = target_value
        TRAINED_TASK_TYPE = task_type
        LAST_PREDICTION_MESSAGE = ""
        if task_type == "regression":
            LAST_TRAIN_MESSAGE = f"Trained {best_model_name}. R^2: {best_score:.4f}"
        else:
            LAST_TRAIN_MESSAGE = (
                f"Trained {best_model_name}. Accuracy: {best_score:.4f}"
            )
        logger.info("Train result: %s", LAST_TRAIN_MESSAGE)
        return LAST_TRAIN_MESSAGE
    except Exception as exc:
        LAST_TRAIN_MESSAGE = f"Training failed: {exc}"
        logger.exception("Training failed at train callback")
        return LAST_TRAIN_MESSAGE


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("prediction-input", "value"),
    State("feature-checklist", "value"),
    State("target-dropdown", "value"),
)
def predict_target(n_clicks, raw_prediction, selected_features, target_value):
    global LAST_PREDICTION_MESSAGE

    if n_clicks == 0:
        return LAST_PREDICTION_MESSAGE
    if TRAINED_MODEL is None or not TRAINED_FEATURES:
        LAST_PREDICTION_MESSAGE = "Train a model before predicting."
        return LAST_PREDICTION_MESSAGE
    if selected_features != TRAINED_FEATURES:
        LAST_PREDICTION_MESSAGE = (
            "Retrain the model after changing the selected features."
        )
        return LAST_PREDICTION_MESSAGE
    if TRAINED_TARGET is None or target_value != TRAINED_TARGET:
        LAST_PREDICTION_MESSAGE = "Retrain the model after changing the target."
        return LAST_PREDICTION_MESSAGE

    parsed_row, error = parse_prediction_values(
        raw_prediction, TRAINED_FEATURES, CURRENT_DF
    )
    if error:
        LAST_PREDICTION_MESSAGE = error
        return LAST_PREDICTION_MESSAGE

    sample = pd.DataFrame([parsed_row], columns=TRAINED_FEATURES)
    prediction = float(TRAINED_MODEL.predict(sample)[0])
    LAST_PREDICTION_MESSAGE = f"Predicted {TRAINED_TARGET}: {prediction:.4f}"
    return LAST_PREDICTION_MESSAGE


if __name__ == "__main__":
    app.run(debug=False)
