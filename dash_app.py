import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.ml_core import build_preprocessor, clean_dataframe


DATA_PATH = "data/avocado.csv"
TARGET = "AveragePrice"
MODEL_PATH = "artifacts/best_model.joblib"

STARTUP_ERROR = ""
MODEL_ERROR = ""
MODEL = None
df = pd.DataFrame()
clean_df = pd.DataFrame()
feature_cols = []
model_feature_cols = []
input_feature_cols = []
numeric_feature_cols = []
categorical_feature_cols = []

try:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    clean_df = clean_dataframe(df)
    feature_cols = [c for c in clean_df.columns if c != TARGET]
    if "Date" in feature_cols:
        # Keep deployment robust by avoiding datetime handling differences.
        feature_cols.remove("Date")
    model_feature_cols = list(feature_cols)
    input_feature_cols = [c for c in model_feature_cols if c in df.columns]
    numeric_feature_cols = [c for c in input_feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_feature_cols = [c for c in input_feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
except Exception as exc:
    STARTUP_ERROR = (
        "Dataset initialization failed. Confirm data/avocado.csv is present "
        f"and readable. Error: {exc}"
    )


def get_model() -> object | None:
    global MODEL, MODEL_ERROR
    if MODEL is not None:
        return MODEL
    if STARTUP_ERROR:
        return None
    try:
        MODEL = joblib.load(MODEL_PATH)
        return MODEL
    except FileNotFoundError:
        try:
            sample_df = clean_df.dropna(subset=[TARGET])
            if len(sample_df) > 6000:
                sample_df = sample_df.sample(n=6000, random_state=42)
            X = sample_df[feature_cols]
            y = sample_df[TARGET]
            fallback = Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(X)),
                    ("model", LinearRegression()),
                ]
            )
            fallback.fit(X, y)
            MODEL = fallback
            return MODEL
        except Exception as exc:
            MODEL_ERROR = f"Fallback model training failed: {exc}"
            return None
    except Exception as exc:
        MODEL_ERROR = f"Model loading failed: {exc}"
        return None


def numeric_default(col: str) -> float:
    return float(df[col].dropna().median())


def text_default(col: str) -> str:
    series = df[col].dropna().astype(str)
    return series.mode().iloc[0] if not series.empty else ""


def category_options(col: str) -> list[dict]:
    values = (
        df[col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return [{"label": v, "value": v} for v in values]


app = Dash(__name__)
app.title = "Avocado Price Dashboard"
server = app.server

if STARTUP_ERROR:
    app.layout = html.Div(
        [
            html.H2("Avocado Price Dashboard"),
            html.P("The app is running, but required startup data is unavailable."),
            html.Pre(STARTUP_ERROR, style={"whiteSpace": "pre-wrap"}),
        ],
        style={"maxWidth": "1100px", "margin": "20px auto"},
    )
else:
    app.layout = html.Div(
        [
            html.H2("Avocado Price Dashboard"),
            html.Br(),
            html.Label("Visualization"),
            dcc.Dropdown(
                id="viz-mode",
                options=[
                    {"label": "Price Trend Over Time", "value": "trend"},
                    {"label": "Average Price by Region", "value": "region"},
                    {"label": "Price vs Total Volume", "value": "volume"},
                ],
                value="trend",
                clearable=False,
            ),
            dcc.Graph(id="viz-graph"),
            html.Br(),
            html.H4("Prediction"),
            html.Div(
                [
                    html.Strong("Feature guide: "),
                    html.Ul(
                        [
                            html.Li("id: row identifier (should not have profound affect on prediction)."),
                            html.Li("Date: week of observation (YYYY-MM-DD)."),
                            html.Li("Total Volume: total number of avocados sold."),
                            html.Li("4046: count sold for PLU 4046 (small Hass)."),
                            html.Li("4225: count sold for PLU 4225 (large Hass)."),
                            html.Li("4770: count sold for PLU 4770 (extra-large Hass)."),
                            html.Li("Total Bags / Small Bags / Large Bags / XLarge Bags: bagged avocado counts by bag size."),
                            html.Li("type: avocado type (conventional or organic)."),
                            html.Li("year: calendar year of the record."),
                            html.Li("region: market/city region."),
                        ],
                        style={"marginTop": "6px", "marginBottom": "10px"},
                    ),
                ],
                style={"fontSize": "0.95rem"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(col),
                            dcc.Input(
                                id=f"input-{col}",
                                type="number",
                                value=float(df[col].dropna().median()),
                                debounce=True,
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    )
                    for col in numeric_feature_cols
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(col),
                            dcc.Dropdown(
                                id=f"input-{col}",
                                options=category_options(col),
                                value=text_default(col),
                                clearable=False,
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    )
                    for col in categorical_feature_cols
                ]
            ),
            html.Button("Predict Price", id="predict-btn", n_clicks=0),
            html.Div(id="prediction-output"),
        ],
        style={"maxWidth": "1100px", "margin": "20px auto"},
    )


if not STARTUP_ERROR:
    @app.callback(
        Output("viz-graph", "figure"),
        Input("viz-mode", "value"),
    )
    def update_visualization(mode):
        if mode == "trend":
            trend_source = df.copy()
            trend_source["Date"] = pd.to_datetime(trend_source["Date"], errors="coerce")
            trend_source = trend_source.dropna(subset=["Date", TARGET, "type"])
            if trend_source.empty:
                return px.line(
                    title="Monthly Average Price Trend by Type (no valid date rows found)"
                )
            trend_df = (
                trend_source.groupby([pd.Grouper(key="Date", freq="ME"), "type"], as_index=False)[TARGET]
                .mean()
            )
            return px.line(
                trend_df,
                x="Date",
                y=TARGET,
                color="type",
                title="Monthly Average Price Trend by Type",
            )
        if mode == "region":
            region_df = (
                df.groupby("region", as_index=False)[TARGET]
                .mean()
                .sort_values(by=TARGET, ascending=False)
                .head(15)
            )
            return px.bar(
                region_df,
                x="region",
                y=TARGET,
                title="Top 15 Regions by Average Price",
            )
        sampled = df[["Total Volume", TARGET, "type"]].dropna().sample(
            n=min(3500, len(df)), random_state=42
        )
        return px.scatter(
            sampled,
            x="Total Volume",
            y=TARGET,
            color="type",
            opacity=0.45,
            title="Average Price vs Total Volume",
        )


if not STARTUP_ERROR:
    @app.callback(
        Output("prediction-output", "children"),
        Input("predict-btn", "n_clicks"),
        [State(f"input-{col}", "value") for col in input_feature_cols],
    )
    def predict_price(n_clicks, *values):
        if n_clicks == 0:
            return ""

        model = get_model()
        if model is None:
            return MODEL_ERROR or "Model is still warming up. Please try again in a few seconds."

        user_input = {}
        for col, val in zip(input_feature_cols, values):
            if col in numeric_feature_cols:
                user_input[col] = numeric_default(col) if val is None else float(val)
            else:
                user_input[col] = text_default(col) if val in (None, "") else str(val)

        if "YearFromDate" in model_feature_cols:
            if "Date" in user_input:
                parsed = pd.to_datetime(user_input["Date"], errors="coerce")
                user_input["YearFromDate"] = parsed.year if not pd.isna(parsed) else int(df["year"].median())
            else:
                user_input["YearFromDate"] = int(df["year"].median())
        if "MonthFromDate" in model_feature_cols:
            if "Date" in user_input:
                parsed = pd.to_datetime(user_input["Date"], errors="coerce")
                user_input["MonthFromDate"] = parsed.month if not pd.isna(parsed) else 1
            else:
                user_input["MonthFromDate"] = 1

        for col in model_feature_cols:
            if col not in user_input:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    user_input[col] = numeric_default(col)
                elif col in df.columns:
                    user_input[col] = text_default(col)
                else:
                    user_input[col] = 0

        sample = pd.DataFrame([user_input], columns=model_feature_cols)
        prediction = float(model.predict(sample)[0])
        return f"Predicted AveragePrice: {prediction:.4f}"


if __name__ == "__main__":
    app.run(debug=False)