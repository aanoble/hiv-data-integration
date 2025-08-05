"""Extract and process DHIS2 data for Consultants."""

import polars as pl
from openhexa.sdk import current_run
from openhexa.toolbox.dhis2 import DHIS2, dataframe
from utils import multi_replace


def extract_dhis2_consultant_data(
    dhis2: DHIS2,
    periods_list: list[str],
    ou_ids: list[str],
    coc: pl.DataFrame,
) -> pl.DataFrame:
    """Fetch and process DHIS2 data for Consultants.

    Args:
        dhis2: DHIS2 connection object.
        periods_list: List of periods to fetch data for.
        ou_ids: List of organization unit IDs.
        coc: DataFrame containing category option combos.
        df_ptme: DataFrame containing PTME data.

    Returns:
        A Polars DataFrame containing processed DHIS2 data for consultants.
    """
    de_cons = [
        "YDot5SUphZV",
        "nUCqISZiVDM",
        "FFPL8EBoeMd",
        "Pi7rVfzadYK",
        "jr6m2WAXchp",
        "pNCQFe2Z0eS",
        "xyNw0Rz34hz",
        "s7agDWyjMWW",
        "TPe9uUCdila",
        "Y7ngiZ8a8t7",
        "eWOytfDFlG2",
        "ZXdCyi2f1Jg",
        "q6IrYbibICS",
        "HiCYvltnIp5",
    ]
    current_run.log_info("⏳ Extraction des éléments de données pour les `consultants`...")
    df_cons = dataframe.extract_analytics(
        dhis2,
        periods=periods_list,
        data_elements=de_cons,
        org_units=ou_ids,
        org_unit_levels=[4],
    )

    df_cons = df_cons.join(
        coc, left_on="category_option_combo_id", right_on="id", how="left"
    ).rename({"name": "coc_name"})
    current_run.log_info("✅ Extraction des données des `consultants` effectuée avec succès.")
    return (
        df_cons.with_columns(
            pl.lit("indicateur_3").alias("column_indicator"),
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
            .alias("coc_name"),
            pl.col("value").cast(pl.Float64),
        )
        .with_columns((pl.col("column_indicator") + "_" + pl.col("coc_name")).alias("column_name"))
        .pivot(
            index=["organisation_unit_id", "period"],
            columns=["column_name"],
            aggregate_function="sum",
            values="value",
        )
    ).with_columns(pl.col(pl.NUMERIC_DTYPES).round(0).cast(pl.Int64))
