from src.data.gx_validation import build_report, _build_suite
from src.utils.io import read_config

import great_expectations as gx
import pandas as pd


def run_hotel_validation(df: pd.DataFrame, report_path: str):
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name="hotel_source")
    data_asset = data_source.add_dataframe_asset(name="hotel_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("hotel_batch")

    suite = _build_suite(context)

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="hotel_validation", data=batch_def, suite=suite
        )
    )

    results = validation_def.run(batch_parameters={"dataframe": df})
    report_text = build_report(results, df)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    context.build_data_docs()
    context.open_data_docs()
    return results


if __name__ == "__main__":
    cfg = read_config()
    report_path = cfg["validation"]["report_path"]
    df = pd.read_csv(
        cfg['data']['merged_data_path'],
        parse_dates=["reservation_status_date"],
    )
    print("{} rows, {} columns\n".format(df.shape[0], df.shape[1]))
    run_hotel_validation(df, report_path)
