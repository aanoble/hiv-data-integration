"""Extract and process DHIS2 data for pathologies PTME."""

from pathlib import Path

import numpy as np
import polars as pl
from constants import (
    DICO_COLUMNS,
    DICO_EXPECTED_COLUMNS,
)
from openhexa.sdk import current_run, workspace
from openhexa.toolbox.dhis2 import DHIS2, dataframe
from utils import fetch_dhis2_indicators, multi_replace, process_column


def extract_dhis2_ptme_data(
    dhis2: DHIS2,
    periods_list: list[str],
    ou_ids: list[str],
    coc: pl.DataFrame,
    fp_ressources: str,
) -> pl.DataFrame:
    """Fetch and process DHIS2 data for PTME pathologies.

    Args:
        dhis2: DHIS2 connection object.
        periods_list: List of periods to fetch data for.
        ou_ids: List of organization unit IDs.
        coc: DataFrame containing category option combos.
        fp_ressources: Path to resource files.
        df_pec: DataFrame containing PEC data.

    Returns:
        A Polars DataFrame containing processed DHIS2 data for PTME pathologies.
    """
    fp_data_element = Path(workspace.files_path) / f"{fp_ressources}/data_element_ptme.parquet"
    if not fp_data_element.exists():
        msg_error = (
            f"Le fichier des éléments de données de la pathologie PTME `{fp_data_element}` "
            "n'existe pas."
        )
        current_run.log_error(msg_error)
        raise FileNotFoundError(f"File {fp_data_element} does not exist.")
    df_data_element_ptme = pl.read_parquet(fp_data_element.as_posix())
    de_ptme = (
        df_data_element_ptme.filter(pl.col("type") == "data_element")
        .select("id")
        .unique()
        .to_series()
        .to_list()
    )

    de_ptme = np.unique([col.split(".")[0] for col in de_ptme]).tolist()
    current_run.log_info("⏳ Extraction et traitement des données pour la pathologie `PTME`...")
    df_ptme = dataframe.extract_analytics(
        dhis2,
        periods=periods_list,
        data_elements=de_ptme,
        org_units=ou_ids,
        org_unit_levels=[4],
    )
    # fetch_dhis2_data(
    #     dhis2=dhis2,
    #     periods_list=periods_list,
    #     ou_ids=ou_ids,
    #     data_elements_uid=de_ptme,
    # )
    df_ptme = (
        df_ptme.join(coc, left_on="category_option_combo_id", right_on="id", how="left").rename(
            {"name": "coc_name"}
        )
        # .drop("attribute_option_combo_id")
    )

    df_ptme = (
        df_ptme.with_columns(
            pl.when(
                (pl.col("category_option_combo_id") != "HllvX50cXC0")
                & (
                    ~pl.col("data_element_id").is_in(
                        ["zeh89uTtSwj", "kHvgHJBM73m", "kHvgHJBM73m", "OgikEnCNaTS"]
                    )
                )
            )
            .then(pl.col("data_element_id") + "." + pl.col("category_option_combo_id"))
            .otherwise(pl.col("data_element_id"))
            .alias("data_element_id_new")
        )
        .join(
            df_data_element_ptme.with_columns(
                pl.col("column")
                .map_elements(lambda s: DICO_COLUMNS["PTME"][s], return_dtype=pl.String)
                .alias("column_indicator")
            ).select(["id", "column_indicator"]),
            left_on="data_element_id_new",
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
        .with_columns(
            pl.col("value").cast(pl.Float64),
            (pl.col("column_indicator")).alias("column_name"),
        )
        .pivot(
            index=["organisation_unit_id", "period"],
            columns=["column_name"],
            values="value",
        )
    )
    df_ptme = df_ptme.rename(lambda x: x.replace(" ", ""))
    df_ptme_ind = fetch_dhis2_indicators(
        dhis2=dhis2,
        periods_list=periods_list,
        indicators_uid=(
            df_data_element_ptme.filter(pl.col("type") == "indicator")
            .select("id")
            .unique()
            .to_series()
            .to_list()
        ),
    )

    df_ptme_ind = (
        df_ptme_ind.with_columns(
            pl.col("dx")
            .map_elements(
                {
                    "mfDEvkT3f6g": "indicateur_3",
                    "B4rn1KsphYr": "indicateur_18",
                    "pruTz6nW3Pg": "indicateur_23",
                }.get,
                return_dtype=pl.String,
            )
            .alias("column_indicator"),
            pl.col("dx_name")
            .map_elements(process_column, return_dtype=pl.String)
            .alias("coc_name"),
        )
        .with_columns((pl.col("column_indicator")).alias("column_name"))
        .rename({"dx": "data_element_id", "ou": "organisation_unit_id", "pe": "period"})
        .pivot(
            index=["organisation_unit_id", "period"],
            columns=["column_name"],
            values="value",
        )
    )
    df_concat = pl.concat([df_ptme, df_ptme_ind], how="diagonal_relaxed")

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
            for col in DICO_EXPECTED_COLUMNS["PTME"]
            if col not in df_concat.columns
        ]
    )

    current_run.log_info(
        "✅ Extraction des données pour la pathologie `PTME` effectuée avec succès."
    )
    return df_concat.select(
        ["organisation_unit_id", "period"]
        + [col for col in DICO_EXPECTED_COLUMNS["PTME"] if col in df_concat.columns]
    ).with_columns(pl.col(pl.NUMERIC_DTYPES).round(0).cast(pl.Int64))
