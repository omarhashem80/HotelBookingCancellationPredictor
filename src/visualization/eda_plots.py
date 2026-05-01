from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.io import read_config


def create_histogram_with_box_plot(
    df: pd.DataFrame, col_name: str, base_path: Path
):
    plot_name = f"{col_name}_dist_histogram_box_plot"
    fig = make_subplots(rows=2, cols=1)
    histo = go.Histogram(x=df[col_name], name="Histogram")
    fig.add_trace(histo, row=1, col=1)
    box_plot = go.Box(x=df[col_name], name="Box")
    fig.add_trace(box_plot, row=2, col=1)
    fig.update_layout(title=f"Distribution of {col_name}", height=600)
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


def create_pie_chart(df: pd.DataFrame, col_name: str, base_path: Path):
    plot_name = f"{col_name}_dist_pie_chart"
    fig = px.pie(
        names=df[col_name], hole=0.3, title=f"Distribution of {col_name}"
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=800)
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


def create_bar_chart(
    df: pd.DataFrame, col_name: str, base_path: Path, top_n=10
):
    plot_name = f"{col_name}_top_frequent_bar_plot"
    dff = df[col_name].value_counts().head(top_n).reset_index()
    dff.columns = [col_name, "count"]
    fig = px.bar(dff, x=col_name, y="count", title=f"Top {top_n} {col_name}")
    fig.update_layout(height=600, xaxis_title=col_name, yaxis_title="count")
    fig.update_traces(textposition="outside")
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


def plot_time_series(
    df: pd.DataFrame, col_name: str, year: int, base_path: Path
):
    plot_name = f"{col_name}_time_series_plot_at_{year}"
    dff = df[df[col_name].dt.year == year]
    dff = (
        dff.groupby(col_name)
        .size()
        .reset_index(name="count")
        .sort_values(col_name)
    )
    fig = px.line(dff, x=col_name, y="count")
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


def plot_time_series_overlay(df: pd.DataFrame, col_name: str, base_path: Path):
    plot_name = f"yearly_comparsion_{col_name}"
    df = df.copy()
    df["year"] = df[col_name].dt.year
    df["day_of_year"] = df[col_name].dt.dayofyear
    dff = df.groupby(["year", "day_of_year"]).size().reset_index(name="count")
    title = f"Yearly Comparsion ({col_name})"
    fig = px.line(
        dff,
        x="day_of_year",
        y="count",
        color="year",
        title=title,
    )
    fig.update_layout(xaxis_title="Day of Year", yaxis_title="Bookings")
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


def generate_country_plot(df: pd.DataFrame, base_path: Path):
    plot_name = "Home_Country_Of_Guests"
    country_data = (
        df.groupby("country")["adults"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    country_data["adults"] = (
        country_data["adults"] / country_data["adults"].sum()
    ) * 100
    country_data.columns = ["country", "Guests in %"]
    guest_map = px.choropleth(
        country_data,
        locations="country",
        locationmode="ISO-3",
        color="Guests in %",
        hover_name="country",
        color_continuous_scale="Plasma",
        title="Home Country of Guests",
    )

    save_path = base_path / f"{plot_name}.jpg"
    guest_map.write_image(save_path)


def univariate_eda_plots(
    numerical_cols: list,
    categorical_cols: list,
    datetime_cols: list,
    df: pd.DataFrame,
):
    cfg = read_config()
    base_path = Path(cfg["reports"]["figures_path"]) / "univariate_plots/"
    base_path.mkdir(parents=True, exist_ok=True)
    for col in numerical_cols:
        create_histogram_with_box_plot(df, col, base_path)
    for col in categorical_cols:
        if col in ("arrival_date_week_number", "agent"):
            create_bar_chart(df, col, base_path)
        else:
            create_pie_chart(df, col, base_path)
    generate_country_plot(df, base_path)
    for col in datetime_cols:
        years = sorted(df[col].dt.year.unique())
        for year in years:
            plot_time_series(df, col, year, base_path)
        plot_time_series_overlay(df, col, base_path)


def plot_mean_numerical_by_cancellation(
    df: pd.DataFrame, col_name: str, base_path: Path
):
    plot_name = f"{col_name}_cancellation_bar"
    title = (
        f'Mean of {col_name.replace("_"," ").title()} by Cancellation Status'
    )
    labels = {
        "is_canceled": "Is Canceled",
        col_name: f'Mean  {col_name.replace("_"," ").title()}',
    }
    dff = df.groupby("is_canceled")[col_name].mean().reset_index()
    fig = px.bar(
        dff,
        x="is_canceled",
        y=col_name,
        title=title,
        labels=labels,
        text_auto=".2f",
    )
    fig.update_layout(xaxis={"categoryorder": "total ascending"})
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


# TODO: Fix the issues in this function
def plot_mean_categorical_by_cancellation(
    df: pd.DataFrame, col_name: str, base_path: Path
):
    plot_name = f"{col_name}_cancellation_bar"
    dff = (
        df.groupby(col_name)["is_canceled"]
        .value_counts(normalize=True)
        .reset_index()
    )
    # dff = dff.melt(
    #     id_vars=col_name, var_name="is_canceled", value_name="proportion"
    # )
    fig = px.bar(
        dff,
        x=col_name,
        y="proportion",
        color="is_canceled",
        barmode="group",
        text_auto=".2f",
        title=f"Distribution of is_canceled by {col_name}",
        labels={
            col_name: col_name.capitalize(),
            "proportion": "Proportion",
            "is_canceled": "Is Canceled",
        },
    )
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


def plot_years_overlay_canceled(
    df: pd.DataFrame, date_col: str, base_path: Path
):
    plot_name = f"{date_col}_cancellation_overlay"
    df = df[df["is_canceled"] == 1].copy()
    df["year"] = df[date_col].dt.year
    df["day_of_year"] = df[date_col].dt.dayofyear
    dff = df.groupby(["year", "day_of_year"]).size().reset_index(name="count")
    fig = px.line(
        dff,
        x="day_of_year",
        y="count",
        color="year",
        title="Yearly Comparsion (Overlay)",
    )
    fig.update_layout(xaxis_title="Day of Year", yaxis_title="Bookings")
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


def plot_time_series_canceled(
    df: pd.DataFrame, date_col: str, year: str, base_path: Path
):
    plot_name = f"{date_col}_cancellation_overlay"
    dff = df.copy()
    dff = dff[dff[date_col].dt.year == year]
    dff = (
        dff.groupby([date_col, "is_canceled"])
        .size()
        .reset_index(name="count")
        .sort_values(date_col)
    )
    dff["status"] = dff["is_canceled"].map({0: "Not Canceled", 1: "Canceled"})
    fig = px.line(dff, x=date_col, y="count", color="status")
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


def plot_all_days_trend_vs_canceled(
    df: pd.DataFrame, date_col: str, base_path: Path
):
    plot_name = f"{date_col}_days_trend_vs_canceled"
    dff = df.copy()
    dff["day_of_year"] = dff[date_col].dt.dayofyear

    dff = (
        dff.groupby(["day_of_year", "is_canceled"])
        .size()
        .reset_index(name="count")
        .sort_values("day_of_year")
    )
    dff["status"] = dff["is_canceled"].map({0: "Not Canceled", 1: "Canceled"})

    fig = px.line(dff, x="day_of_year", y="count", color="status")
    save_path = base_path / f"{plot_name}.jpg"


def correlation_matrix(
    df: pd.DataFrame, numerical_cols: list, base_path: Path
):
    plot_name = f"correlation_matrix_for_numerical_features"
    fig = px.imshow(
        df[numerical_cols].corr(numeric_only=True, method="spearman"),
        text_auto=".2f",
        width=1000,
        height=800,
    )
    save_path = base_path / f"{plot_name}.jpg"
    fig.write_image(save_path)


def bivariate_eda_plots(
    numerical_cols: list,
    categorical_cols: list,
    datetime_cols: list,
    df: pd.DataFrame,
):
    cfg = read_config()
    base_path = Path(cfg["reports"]["figures_path"]) / "bivariate_plots/"
    base_path.mkdir(parents=True, exist_ok=True)
    correlation_matrix(df, numerical_cols, base_path)
    for col in numerical_cols:
        plot_mean_numerical_by_cancellation(df, col, base_path)
    for col in categorical_cols:
        if col == "is_canceled":
            continue
        plot_mean_categorical_by_cancellation(df, col, base_path)
    for col in datetime_cols:
        years = sorted(df[col].dt.year.unique())
        for year in years:
            plot_time_series_canceled(df, col, year, base_path)
        plot_years_overlay_canceled(df, col, base_path)
        plot_all_days_trend_vs_canceled(df, col, base_path)


def create_eda_plots(type_normalized_df: pd.DataFrame):
    numerical_cols = type_normalized_df.select_dtypes(
        include=np.number
    ).columns.tolist()
    categorical_cols = type_normalized_df.select_dtypes(
        include="category"
    ).columns.tolist()
    datetime_cols = type_normalized_df.select_dtypes(
        include="datetime64[ns]"
    ).columns.tolist()
    univariate_eda_plots(
        numerical_cols, categorical_cols, datetime_cols, type_normalized_df
    )
    bivariate_eda_plots(
        numerical_cols, categorical_cols, datetime_cols, type_normalized_df
    )
