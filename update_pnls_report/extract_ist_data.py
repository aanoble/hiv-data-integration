"""Extract and process DHIS2 data for pathologies IST CD."""

from pathlib import Path

import polars as pl
from constants import (
    DICO_COLUMNS,
    DICO_EXPECTED_COLUMNS,
)
from openhexa.sdk import current_run, workspace
from openhexa.toolbox.dhis2 import DHIS2, dataframe
from utils import fetch_dhis2_indicators, multi_replace, process_column


def extract_dhis2_ist_data(
    dhis2: DHIS2,
    periods_list: list[str],
    ou_ids: list[str],
    coc: pl.DataFrame,
    fp_ressources: str,
) -> pl.DataFrame:
    """Fetch and process DHIS2 data for IST CD pathologies.

    Args:
        dhis2: DHIS2 connection object.
        periods_list: List of periods to fetch data for.
        ou_ids: List of organization unit IDs.
        coc: DataFrame containing category option combos.
        fp_ressources: Path to resource files.

    Returns:
        A Polars DataFrame containing processed DHIS2 data for IST CD pathologies.
    """
    fp_data_element = Path(workspace.files_path) / f"{fp_ressources}/data_element_ist.parquet"
    if not fp_data_element.exists():
        msg_error = (
            f"Le fichier des éléments de données de la pathologie IST CD `{fp_data_element}` "
            "n'existe pas."
        )
        current_run.log_error(msg_error)
        raise FileNotFoundError(f"File {fp_data_element} does not exist.")

    df_data_element_ist = pl.read_parquet(fp_data_element.as_posix())

    de_ist = (
        df_data_element_ist.filter(pl.col("type") == "data_element")
        .select("id")
        .unique()
        .to_series()
        .to_list()
    )
    current_run.log_info(
        "Extraction des données DHIS2 des différentes pathologies "
        f"aux périodes suivantes : {', '.join(periods_list)}."
    )
    current_run.log_info("⏳ Extraction et traitements des données pour la pathologie `IST CD`...")

    df_ist = dataframe.extract_analytics(
        dhis2,
        periods=periods_list,
        data_elements=de_ist,
        org_units=ou_ids,
        org_unit_levels=[4],
    )

    df_ist = df_ist.join(coc, left_on="category_option_combo_id", right_on="id", how="left").rename(
        {"name": "coc_name"}
    )
    df_ist = (
        df_ist.join(
            df_data_element_ist.with_columns(
                pl.col("column")
                .map_elements(lambda s: DICO_COLUMNS["IST"][s], return_dtype=pl.String)
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
    df_ist = df_ist.rename(lambda x: x.replace(" ", ""))
    df_ist_ind = fetch_dhis2_indicators(
        dhis2=dhis2,
        periods_list=periods_list,
        indicators_uid=[
            "r8NxcXBxRlw",
            "jm1J88Ye2dJ",
            "HW8DCBLBL2H",
            "euQzrRXmR0X",
            "eTUDNIwZKMm",
            "fTogxCAg65D",
            "ee7RhWGrsa4",
            "pIf1CmuVeQF",
            "rKGxxPDJ9Rz",
            "JubvGJM4hly",
        ],
    )
    df_ist_ind = (
        df_ist_ind.with_columns(
            pl.lit("indicateur_7").alias("column_indicator"),
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
    df_concat = pl.concat([df_ist, df_ist_ind], how="diagonal_relaxed")

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
            for col in DICO_EXPECTED_COLUMNS["IST"]
            if col not in df_concat.columns
        ]
    )
    current_run.log_info(
        "✅ Extraction des données pour la pathologie `IST CD` effectuée avec succès."
    )
    return df_concat.select(
        ["organisation_unit_id", "period"]
        + [col for col in DICO_EXPECTED_COLUMNS["IST"] if col in df_concat.columns]
    ).with_columns(pl.col(pl.NUMERIC_DTYPES).round(0).cast(pl.Int64))
