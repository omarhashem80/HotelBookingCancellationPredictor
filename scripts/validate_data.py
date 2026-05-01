from src.data.gx_validation import build_report, _build_suite
from src.utils.io import read_config
from src.config.logging import configure_logging

import great_expectations as gx
import pandas as pd
from loguru import logger


def run_hotel_validation(df: pd.DataFrame, report_path: str):
    logger.info("Running data validation. rows={}, cols={}", df.shape[0], df.shape[1])
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name="hotel_source")
    data_asset = data_source.add_dataframe_asset(name="hotel_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("hotel_batch")

    suite = _build_suite(context)

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(name="hotel_validation", data=batch_def, suite=suite)
    )

    results = validation_def.run(batch_parameters={"dataframe": df})
    report_text = build_report(results, df)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Saved validation report to {}", report_path)

    context.build_data_docs()
    context.open_data_docs()
    return results


if __name__ == "__main__":
    configure_logging()
    cfg = read_config()
    report_path = cfg["reports"]["validation_path"]
    df = pd.read_csv(
        cfg["data"]["merged_data_path"],
        parse_dates=["reservation_status_date"],
    )
    logger.info("Loaded merged data: rows={}, cols={}", df.shape[0], df.shape[1])
    run_hotel_validation(df, report_path)
