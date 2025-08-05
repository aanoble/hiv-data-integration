"""Extract and process DHIS2 data for pathologies PEC and aggregated data PEC."""

from pathlib import Path

import polars as pl
from constants import (
    DICO_COLUMNS,
    DICO_EXPECTED_COLUMNS,
)
from openhexa.sdk import current_run, workspace
from openhexa.toolbox.dhis2 import DHIS2, dataframe, periods
from utils import fetch_dhis2_indicators, multi_replace, process_column


def extract_dhis2_pec_data(
    dhis2: DHIS2,
    periods_list: list[str],
    ou_ids: list[str],
    coc: pl.DataFrame,
    fp_ressources: str,
) -> pl.DataFrame:
    """Fetch and process DHIS2 data for PEC pathologies.

    Args:
        dhis2: DHIS2 connection object.
        periods_list: List of periods to fetch data for.
        ou_ids: List of organization unit IDs.
        coc: DataFrame containing category option combos.
        fp_ressources: Path to resource files.

    Returns:
        A Polars DataFrame containing processed DHIS2 data for PEC pathologies.
    """
    fp_data_element = Path(workspace.files_path) / f"{fp_ressources}/data_element_pec.parquet"
    if not fp_data_element.exists():
        msg_error = (
            f"Le fichier des éléments de données de la pathologie PEC `{fp_data_element}` "
            "n'existe pas."
        )
        current_run.log_error(msg_error)
        raise FileNotFoundError(f"File {fp_data_element} does not exist.")
    df_data_element_pec = pl.read_parquet(fp_data_element.as_posix())
    de_pec = (
        df_data_element_pec.filter(pl.col("type") == "data_element")
        .select("id")
        .unique()
        .to_series()
        .to_list()
    )
    current_run.log_info("⏳ Extraction et traitement des données pour la pathologie `PEC`...")
    df_pec = dataframe.extract_analytics(
        dhis2,
        periods=periods_list,
        data_elements=de_pec,
        org_units=ou_ids,
        org_unit_levels=[4],
    )
    df_pec = df_pec.join(coc, left_on="category_option_combo_id", right_on="id", how="left").rename(
        {"name": "coc_name"}
    )

    df_pec = (
        df_pec.join(
            df_data_element_pec.with_columns(
                pl.col("column")
                .map_elements(lambda s: DICO_COLUMNS["PEC"][s], return_dtype=pl.String)
                .alias("column_indicator")
            ).select(["id", "column_indicator"]),
            left_on="data_element_id",
            right_on="id",
            how="left",
        )
        .with_columns(
            pl.col("coc_name")
            .map_elements(
                lambda x: (
                    multi_replace(x).replace("Féminin", "").strip().strip(",") + "_F"
                    if "Féminin" in x
                    else multi_replace(x).replace("Masculin", "").strip().strip(",") + "_M"
                    if "Masculin" in x
                    else multi_replace(x)
                ),
                return_dtype=pl.String,
            )
            .alias("coc_name")
        )
        .with_columns((pl.col("column_indicator") + "_" + pl.col("coc_name")).alias("column_name"))
        .pivot(
            index=["organisation_unit_id", "period"],
            columns=["column_name"],
            values="value",
        )
    )
    df_pec = df_pec.rename(lambda x: x.replace(" ", ""))

    df_pec_ind = fetch_dhis2_indicators(
        dhis2=dhis2,
        periods_list=periods_list,
        indicators_uid=[
            "bZ113eh6zgZ",
            "xYrmHNcrCMK",
            "etFvcc6ABmx",
            "Y99o4wHv1zr",
            "t90eO6Qo5An",
            "rymKnAHRp45",
            "gqn8Ybqla8m",
            "f9rSMi5o1j4",
            "xP7UNxEE9Lc",
            "pmi6fk6MR9s",
            "jb6GKFZffJ7",
            "ZBoNPLk8hL6",
            "F23qEOERvP0",
            "bzEdlzRfDOV",
        ],
    )
    df_pec_ind = (
        df_pec_ind.with_columns(
            pl.lit("indicateur_4").alias("column_indicator"),
            pl.col("dx_name")
            .map_elements(process_column, return_dtype=pl.String)
            .alias("coc_name"),
        )
        .with_columns((pl.col("column_indicator") + "_" + pl.col("coc_name")).alias("column_name"))
        .rename({"dx": "data_element_id", "ou": "organisation_unit_id", "pe": "period"})
        .pivot(
            index=["organisation_unit_id", "period"],
            columns=["column_name"],
            values="value",
        )
    )
    df_concat = pl.concat([df_pec, df_pec_ind], how="diagonal_relaxed")

    df_concat = df_concat.with_columns(
        [
            pl.col(c).cast(pl.Float64)
            for c in df_concat.columns
            if c not in ["organisation_unit_id", "period"]
        ]
    )

    df_concat = (
        df_concat.group_by(["organisation_unit_id", "period"])  # .fill_null(0)
        .agg(
            [
                pl.when(pl.col(col).is_not_null().any()).then(pl.col(col).sum()).otherwise(None)
                # pl.sum(col)
                for col in df_concat.columns
                if col not in ["organisation_unit_id", "period"]
            ]
        )
        .sort(["organisation_unit_id", "period"])
    )

    df_concat = df_concat.with_columns(
        [
            pl.lit(None).alias(col)
            for col in DICO_EXPECTED_COLUMNS["PEC"]
            if col not in df_concat.columns
        ]
    )
    current_run.log_info(
        "✅ Extraction des données pour la pathologie `PEC` effectuée avec succès."
    )
    return df_concat.select(
        ["organisation_unit_id", "period"]
        + [col for col in DICO_EXPECTED_COLUMNS["PEC"] if col in df_concat.columns]
    ).with_columns(pl.col(pl.NUMERIC_DTYPES).round(0).cast(pl.Int64))


def extract_dhis2_pec_aggregated_data(
    dhis2: DHIS2, ou_ids: list[str], coc: pl.DataFrame, periods_list: list[str]
) -> pl.DataFrame:
    """Fetch and process DHIS2 data for indicators 11, 14, and PEC.

    Args:
        dhis2: DHIS2 connection object.
        ou_ids: List of organization unit IDs.
        coc: DataFrame containing category option combos.
        periods_list: List of periods to fetch data for.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the processed data for indicators 11, 14.
    """
    if not any(p.endswith(("03", "06", "09", "12")) for p in periods_list):
        return pl.DataFrame()

    current_run.log_info(
        "⏳ Extraction des données aggrégées sur 6 mois pour la pathologie `PEC`..."
    )

    df_ind = pl.DataFrame()
    year = int(periods_list[0][:4])
    periods_map = {
        "03": (f"{year - 1}10", f"{year}03"),
        "06": (f"{year}01", f"{year}06"),
        "09": (f"{year}04", f"{year}09"),
        "12": (f"{year}07", f"{year}12"),
    }
    data_elements = ["zxMJc6KPesz", "ykXAaZFC1bt"]
    collected_data = []
    valid_suffixes = {
        s for s in ("03", "06", "09", "12") if any(p.endswith(s) for p in periods_list)
    }

    for period_suffix in valid_suffixes:
        start, end = periods_map[period_suffix]
        month_range = periods.Month.from_string(start).range(periods.Month.from_string(end))
        periods_range = [month.period for month in month_range]
        data = dhis2.analytics.get(
            periods=periods_range,
            data_elements=data_elements,
            org_units=ou_ids,
        )
        df = (
            pl.DataFrame(data)
            .with_columns(pl.lit(f"{year}{period_suffix}").alias("period"))
            .select(["dx", "co", "ou", "period", "value"])
        )
        collected_data.append(df)

    if not collected_data:
        return pl.DataFrame()

    df_ind = (
        pl.concat(collected_data, how="diagonal_relaxed")
        .rename(
            {
                "dx": "column",
                "co": "category_option_combo_id",
                "ou": "organisation_unit_id",
            }
        )
        .join(coc, left_on="category_option_combo_id", right_on="id", how="left")
        .rename({"name": "coc_name"})
    )

    df_ind = (
        df_ind.with_columns(
            pl.col("column")
            .replace(
                {"zxMJc6KPesz": "indicateur_11", "ykXAaZFC1bt": "indicateur_14"},
                default=pl.first(),
            )
            .alias("column_indicator"),
            pl.col("coc_name").map_elements(
                lambda x: (
                    multi_replace(x).replace("Féminin", "").strip().strip(",") + "_F"
                    if "Féminin" in x
                    else multi_replace(x).replace("Masculin", "").strip().strip(",") + "_M"
                    if "Masculin" in x
                    else multi_replace(x)
                ),
                return_dtype=pl.String,
            ),
        )
        .with_columns(
            (pl.col("column_indicator") + "_" + pl.col("coc_name")).alias("column_name"),
            pl.col("value").cast(pl.Float64),
        )
        .pivot(
            index=["organisation_unit_id", "period"],
            columns=["column_name"],
            aggregate_function="sum",
            values="value",
        )
        .rename(lambda x: x.replace(" ", ""))
    )

    df_ind = df_ind.with_columns(
        [
            pl.col(c).cast(pl.Float64)
            for c in df_ind.columns
            if c not in ["organisation_unit_id", "period"]
        ]
    )
    df_ind = (
        df_ind.group_by(["organisation_unit_id", "period"])
        .agg(
            [
                pl.when(pl.col(col).is_not_null().any()).then(pl.col(col).sum()).otherwise(None)
                for col in df_ind.columns
                if col not in ["organisation_unit_id", "period"]
            ]
        )
        .sort(["organisation_unit_id", "period"])
    )
    current_run.log_info(
        "✅ Extraction des données aggrégées sur 6 mois pour la "
        "pathologie `PEC` effectuée avec succès."
    )
    return df_ind
