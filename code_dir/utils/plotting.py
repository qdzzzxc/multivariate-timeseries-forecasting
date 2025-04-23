import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def plot_forecasts(
    df: pd.DataFrame,
    models_predictions: dict[str, pd.DataFrame],
    start_date: str | None = None,
    end_date: str | None = None,
):
    model_names = list(models_predictions.keys())
    n_models = len(model_names)
    rows = int(np.ceil(n_models / 2))
    cols = 2

    filtered_df = df.copy()
    if start_date is not None and end_date is not None:
        mask = (filtered_df["timestamp"] >= start_date) & (
            filtered_df["timestamp"] <= end_date
        )
        filtered_df = filtered_df[mask]

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=model_names, vertical_spacing=0.1
    )

    for i, model_name in enumerate(model_names):
        row = i // 2 + 1
        col = i % 2 + 1
        model_pred = models_predictions[model_name]

        if start_date is not None and end_date is not None:
            mask = (df["timestamp"] >= start_date) & (
                df["timestamp"] <= end_date
            )
            filtered_mean = model_pred["mean"].values[mask]
            filtered_upper = (
                model_pred["0.9"].values[mask] if "0.9" in model_pred else None
            )
            filtered_lower = (
                model_pred["0.1"].values[mask] if "0.1" in model_pred else None
            )
        else:
            filtered_mean = model_pred["mean"]
            filtered_upper = model_pred["0.9"] if "0.9" in model_pred else None
            filtered_lower = model_pred["0.1"] if "0.1" in model_pred else None

        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"],
                y=filtered_df["target"],
                name="Test",
                line=dict(color="#50C878"),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"],
                y=filtered_mean,
                name=f"{model_name} (mean)",
                line=dict(color="#D70040", dash="dot"),
            ),
            row=row,
            col=col,
        )

        if filtered_upper is not None and filtered_lower is not None:
            fig.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"],
                    y=filtered_upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"],
                    y=filtered_lower,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255, 127, 14, 0.6)",
                    line=dict(width=0),
                    name=f"{model_name} CI (0.1-0.9)",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title="Forecasts for Different Models",
        template="plotly_white",
        height=300 * rows,
        width=1400,
        showlegend=False,
    )

    for i in range(1, rows * cols + 1):
        fig.update_xaxes(
            title_text="Date",
            row=i // cols + 1,
            col=i % cols if i % cols != 0 else cols,
        )
        fig.update_yaxes(
            title_text="National Demand",
            row=i // cols + 1,
            col=i % cols if i % cols != 0 else cols,
        )

    fig.show()

def plot_single_forecast(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    actual_col: str = "actual",
    pred_col: str = "prediction",
    upper_col: str = "upper_bound",
    lower_col: str = "lower_bound",
    start_date: str | None = None,
    end_date: str | None = None,
    title: str = "Forecast",
    model_name: str = "Model",
    height: int=600,
    width: int=1000,
):
    filtered_df = df.copy()
    if start_date is not None and end_date is not None:
        mask = (filtered_df[timestamp_col] >= start_date) & (filtered_df[timestamp_col] <= end_date)
        filtered_df = filtered_df[mask]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=filtered_df[timestamp_col],
            y=filtered_df[actual_col],
            name="Actual",
            line=dict(color="#50C878"),
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=filtered_df[timestamp_col],
            y=filtered_df[pred_col],
            name=f"{model_name} (prediction)",
            line=dict(color="#D70040", dash="dot"),
        )
    )
    
    has_upper = upper_col in filtered_df.columns
    has_lower = lower_col in filtered_df.columns
    
    if has_upper and has_lower:
        fig.add_trace(
            go.Scatter(
                x=filtered_df[timestamp_col],
                y=filtered_df[upper_col],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=filtered_df[timestamp_col],
                y=filtered_df[lower_col],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(255, 127, 14, 0.6)",
                line=dict(width=0),
                name=f"{model_name} CI",
            )
        )
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        width=width,
        xaxis_title="Date",
        yaxis_title="Value",
    )
    
    return fig