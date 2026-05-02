"""
EDA plotting functions using matplotlib/seaborn for reliable PNG image generation.
This replaces plotly+kaleido which has issues on Windows.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.utils.io import read_config

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'


def save_figure(fig, save_path: Path):
    """Safely save matplotlib figure."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches='tight', dpi=150)
        plt.close(fig)
        logger.debug(f"Successfully saved {save_path.name}")
    except Exception as e:
        logger.error(f"Error saving {save_path.name}: {e}")
        plt.close(fig)


def create_histogram_with_box_plot(df: pd.DataFrame, col_name: str, base_path: Path):
    """Create histogram and box plot for numerical column."""
    plot_name = f"{col_name}_dist_histogram_box_plot"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Histogram
    ax1.hist(df[col_name].dropna(), bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel(col_name.replace('_', ' ').title())
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of {col_name.replace("_", " ").title()}')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(df[col_name].dropna(), vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_xlabel(col_name.replace('_', ' ').title())
    ax2.set_title(f'Box Plot of {col_name.replace("_", " ").title()}')
    ax2.grid(True, alpha=0.3)
    
    save_path = base_path / f"{plot_name}.png"
    save_figure(fig, save_path)


def create_pie_chart(df: pd.DataFrame, col_name: str, base_path: Path):
    """Create pie chart for categorical column."""
    plot_name = f"{col_name}_dist_pie_chart"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    value_counts = df[col_name].value_counts()
    colors = sns.color_palette('husl', len(value_counts))
    
    wedges, texts, autotexts = ax.pie(value_counts, labels=value_counts.index,
                                       autopct='%1.1f%%', colors=colors,
                                       startangle=90, textprops={'fontsize': 10})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(f'Distribution of {col_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    
    save_path = base_path / f"{plot_name}.png"
    save_figure(fig, save_path)


def create_bar_chart(df: pd.DataFrame, col_name: str, base_path: Path, top_n=10):
    """Create bar chart for top N values."""
    plot_name = f"{col_name}_top_frequent_bar_plot"
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    value_counts = df[col_name].value_counts().head(top_n)
    
    bars = ax.bar(range(len(value_counts)), value_counts.values, 
                  color=sns.color_palette('husl', len(value_counts)), edgecolor='black')
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
    ax.set_xlabel(col_name.replace('_', ' ').title())
    ax.set_ylabel('Count')
    ax.set_title(f'Top {top_n} {col_name.replace("_", " ").title()}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    save_path = base_path / f"{plot_name}.png"
    save_figure(fig, save_path)


def plot_time_series(df: pd.DataFrame, col_name: str, year: int, base_path: Path):
    """Create time series plot for specific year."""
    plot_name = f"{col_name}_time_series_plot_at_{year}"
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    dff = df[df[col_name].dt.year == year].copy()
    dff = dff.groupby(col_name).size().reset_index(name="count").sort_values(col_name)
    
    ax.plot(dff[col_name], dff['count'], linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Bookings')
    ax.set_title(f'Booking Trends - {year}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    save_path = base_path / f"{plot_name}.png"
    save_figure(fig, save_path)


def plot_time_series_overlay(df: pd.DataFrame, col_name: str, base_path: Path):
    """Create overlayed time series plot for multiple years."""
    plot_name = f"yearly_comparsion_{col_name}"
    
    df_copy = df.copy()
    df_copy["year"] = df_copy[col_name].dt.year
    df_copy["day_of_year"] = df_copy[col_name].dt.dayofyear
    dff = df_copy.groupby(["year", "day_of_year"]).size().reset_index(name="count")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    years = sorted(dff['year'].unique())
    colors = sns.color_palette('husl', len(years))
    
    for i, year in enumerate(years):
        year_data = dff[dff['year'] == year]
        ax.plot(year_data['day_of_year'], year_data['count'], 
                label=str(year), linewidth=2, color=colors[i])
    
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Number of Bookings')
    ax.set_title(f'Yearly Comparison - {col_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = base_path / f"{plot_name}.png"
    save_figure(fig, save_path)


def generate_country_plot(df: pd.DataFrame, base_path: Path):
    """Create country distribution visualization."""
    plot_name = "Home_Country_Of_Guests"
    
    country_data = (
        df.groupby("country")["adults"].sum()
        .sort_values(ascending=False).head(20).reset_index()
    )
    country_data["percentage"] = (country_data["adults"] / country_data["adults"].sum()) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(country_data['country'], country_data['percentage'],
                   color=sns.color_palette('viridis', len(country_data)), edgecolor='black')
    ax.set_xlabel('Percentage of Guests (%)')
    ax.set_ylabel('Country')
    ax.set_title('Top 20 Home Countries of Guests', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%',
                ha='left', va='center', fontsize=8, fontweight='bold')
    
    save_path = base_path / f"{plot_name}.png"
    save_figure(fig, save_path)


def univariate_eda_plots(
    numerical_cols: list,
    categorical_cols: list,
    datetime_cols: list,
    df: pd.DataFrame,
):
    cfg = read_config()
    base_path = Path(cfg["visualization"]["reports_path"]) / "univariate_plots/"
    base_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {len(numerical_cols)} numerical column plots...")
    for i, col in enumerate(numerical_cols, 1):
        logger.info(f"[{i}/{len(numerical_cols)}] Creating histogram for {col}")
        create_histogram_with_box_plot(df, col, base_path)
    
    logger.info(f"Generating {len(categorical_cols)} categorical column plots...")
    for i, col in enumerate(categorical_cols, 1):
        logger.info(f"[{i}/{len(categorical_cols)}] Creating plot for {col}")
        if col in ("arrival_date_week_number", "agent"):
            create_bar_chart(df, col, base_path)
        else:
            create_pie_chart(df, col, base_path)
    
    logger.info("Generating country plot...")
    generate_country_plot(df, base_path)
    
    logger.info(f"Generating time series plots for {len(datetime_cols)} datetime columns...")
    for col in datetime_cols:
        years = sorted(df[col].dt.year.unique())
        for year in years:
            logger.info(f"Creating time series plot for {col} year {year}")
            plot_time_series(df, col, year, base_path)
        logger.info(f"Creating overlay plot for {col}")
        plot_time_series_overlay(df, col, base_path)


def plot_mean_numerical_by_cancellation(df: pd.DataFrame, col_name: str, base_path: Path):
    """Plot mean of numerical column by cancellation status."""
    plot_name = f"{col_name}_cancellation_bar"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dff = df.groupby("is_canceled")[col_name].mean().reset_index()
    dff['is_canceled'] = dff['is_canceled'].map({0: 'Not Canceled', 1: 'Canceled'})
    
    bars = ax.bar(dff['is_canceled'], dff[col_name],
                  color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cancellation Status')
    ax.set_ylabel(f'Mean {col_name.replace("_", " ").title()}')
    ax.set_title(f'Mean {col_name.replace("_", " ").title()} by Cancellation Status',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    save_path = base_path / f"{plot_name}.png"
    save_figure(fig, save_path)


def plot_mean_categorical_by_cancellation(df: pd.DataFrame, col_name: str, base_path: Path):
    """Plot cancellation rate by categorical variable."""
    plot_name = f"{col_name}_cancellation_bar"
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert is_canceled to numeric for aggregation
    df_temp = df.copy()
    df_temp['is_canceled_num'] = df_temp['is_canceled'].astype(int)
    dff = df_temp.groupby(col_name)["is_canceled_num"].mean().reset_index()
    dff = dff.rename(columns={'is_canceled_num': 'is_canceled'})
    dff = dff.sort_values("is_canceled", ascending=False)
    
    bars = ax.bar(range(len(dff)), dff['is_canceled'],
                  color=sns.color_palette('RdYlGn_r', len(dff)), edgecolor='black')
    ax.set_xticks(range(len(dff)))
    ax.set_xticklabels(dff[col_name], rotation=45, ha='right')
    ax.set_xlabel(col_name.replace('_', ' ').title())
    ax.set_ylabel('Cancellation Rate')
    ax.set_title(f'Cancellation Rate by {col_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    save_path = base_path / f"{plot_name}.png"
    save_figure(fig, save_path)


def correlation_matrix(df: pd.DataFrame, numerical_cols: list, base_path: Path):
    """Create correlation matrix heatmap."""
    plot_name = "correlation_matrix_for_numerical_features"
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    corr = df[numerical_cols].corr(method='spearman')
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix (Spearman)', fontsize=14, fontweight='bold')
    
    save_path = base_path / f"{plot_name}.png"
    save_figure(fig, save_path)


def bivariate_eda_plots(numerical_cols: list, categorical_cols: list, datetime_cols: list, df: pd.DataFrame):
    """Generate all bivariate plots."""
    cfg = read_config()
    base_path = Path(cfg["visualization"]["reports_path"]) / "bivariate_plots/"
    base_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating correlation matrix...")
    correlation_matrix(df, numerical_cols, base_path)
    
    logger.info(f"Generating {len(numerical_cols)} numerical vs cancellation plots...")
    for col in numerical_cols:
        logger.info(f"Creating cancellation plot for {col}")
        plot_mean_numerical_by_cancellation(df, col, base_path)
    
    # Filter out is_canceled from categorical columns for bivariate analysis
    categorical_cols_filtered = [col for col in categorical_cols if col != 'is_canceled']
    logger.info(f"Generating {len(categorical_cols_filtered)} categorical vs cancellation plots...")
    for col in categorical_cols_filtered:
        logger.info(f"Creating cancellation plot for {col}")
        plot_mean_categorical_by_cancellation(df, col, base_path)


def create_eda_plots(type_normalized_df: pd.DataFrame):
    """Main function to generate all EDA plots."""
    numerical_cols = type_normalized_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = type_normalized_df.select_dtypes(include="category").columns.tolist()
    datetime_cols = type_normalized_df.select_dtypes(include="datetime64[ns]").columns.tolist()
    
    univariate_eda_plots(numerical_cols, categorical_cols, datetime_cols, type_normalized_df)
    bivariate_eda_plots(numerical_cols, categorical_cols, datetime_cols, type_normalized_df)


