from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def generate_eda_plots(df: pd.DataFrame, output_dir: str | Path) -> None:
	"""Generate required EDA visualizations and save them as PNG files."""
	output = Path(output_dir)
	_ensure_dir(output)

	if "is_canceled" in df.columns:
		plt.figure(figsize=(8, 4))
		df["is_canceled"].value_counts().plot(kind="bar")
		plt.title("Cancellation Distribution")
		plt.tight_layout()
		plt.savefig(output / "cancellation_distribution.png")
		plt.close()

	if {"lead_time", "is_canceled"}.issubset(df.columns):
		plt.figure(figsize=(8, 4))
		df.boxplot(column="lead_time", by="is_canceled")
		plt.title("Lead Time vs Cancellation")
		plt.suptitle("")
		plt.tight_layout()
		plt.savefig(output / "lead_time_vs_cancellation.png")
		plt.close()

	if "adr" in df.columns:
		plt.figure(figsize=(8, 4))
		df["adr"].hist(bins=40)
		plt.title("ADR Distribution")
		plt.tight_layout()
		plt.savefig(output / "adr_distribution.png")
		plt.close()

	numeric_df = df.select_dtypes(include=["number"])
	if not numeric_df.empty:
		plt.figure(figsize=(10, 8))
		plt.imshow(numeric_df.corr(), cmap="coolwarm", aspect="auto")
		plt.colorbar()
		plt.xticks(range(len(numeric_df.columns)), numeric_df.columns, rotation=90)
		plt.yticks(range(len(numeric_df.columns)), numeric_df.columns)
		plt.title("Correlation Heatmap")
		plt.tight_layout()
		plt.savefig(output / "correlation_heatmap.png")
		plt.close()

	if "hotel" in df.columns and "is_canceled" in df.columns:
		plt.figure(figsize=(8, 4))
		pd.crosstab(df["hotel"], df["is_canceled"], normalize="index").plot(kind="bar", ax=plt.gca())
		plt.title("Cancellation Rate by Hotel")
		plt.tight_layout()
		plt.savefig(output / "hotel_categorical_analysis.png")
		plt.close()

