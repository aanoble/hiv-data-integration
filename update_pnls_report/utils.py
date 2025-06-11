import contextlib
import functools
import json

# import operator
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl as pyxl

# import papermill as pm
import polars as pl
import requests
from constants import (
    COLUMN_NAME_GROUP_AGE,
    DICO_COLUMNS,
    DICO_EXPECTED_COLUMNS,
    MAP_AGE_GROUP,
)
from openhexa.sdk import current_run, workspace
from openhexa.toolbox.dhis2 import DHIS2, dataframe, periods
from openpyxl.utils.dataframe import dataframe_to_rows


def extract_naomi_api_data(
    year: int,
    fp_ressources: str,
    base_url: str | None = None,
    indicators_mapping: dict[str, str] | None = None,
    sex_mapping: dict[str, str] | None = None,
    age_mapping: dict[str, str] | None = None,
    max_workers: int = 5,
) -> pl.DataFrame:
    """Récupère les données Naomi de l'API et les combine dans un DataFrame optimisé.

    Args:
        year: Année de référence
        fp_ressources: Chemin d'accès aux fichiers ressources
        base_url: URL template avec placeholders {indicator}, {age_group}, {year}, {sex}
        indicators_mapping: Dictionnnaire de mapping des indicateurs à récupérer
        sex_mapping: Dictionnaire de mapping des sexes
        age_mapping: Dictionnaire de mapping des groupes d'âge
        max_workers: Nombre de requêtes parallèles

    Returns:
        DataFrame Polars avec les données consolidées
    """
    base_url = (
        "https://naomiviewerserver.azurewebsites.net/api/v1/areas?country=CIV&indicator={indicator}&ageGroup={age_group}&period={year}-4&sex={sex}&areaLevel=2"
        if base_url is None
        else base_url
    )

    indicators_mapping = (
        {
            "aware_plhiv_num": "indicateur_9",
            "plhiv": "indicateur_10",
        }
        if indicators_mapping is None
        else indicators_mapping
    )
    sex_mapping = {"male": "M", "female": "F"} if sex_mapping is None else sex_mapping
    age_mapping = (
        {
            "age_0_4_ans": "Y000_004",
            "age_05_09_ans": "Y005_009",
            "age_10_14_ans": "Y010_014",
            "age_15_19_ans": "Y015_019",
            "age_20_24_ans": "Y020_024",
            "age_25_49_ans": "Y025_049",
            "age_50_ans_et_plus": "Y050_999",
        }
        if age_mapping is None
        else age_mapping
    )
    current_run.log_info(f"⏳ Extraction des données NAOMI pour l'année `{year}`...")
    params = [
        (ind, sex, age, age_code)
        for ind in indicators_mapping
        for sex in sex_mapping
        for age, age_code in age_mapping.items()
    ]

    def process_request(ind: str, sex: str, age: str, age_code: str) -> pl.DataFrame | None:
        try:
            url = base_url.format(indicator=ind, age_group=age_code, year=year, sex=sex)
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data or not data[0].get("subareas"):
                return None

            subareas = [row["subareas"] for row in data[0]["subareas"]]
            flat_data = [item for sublist in subareas for item in sublist]

            return pl.DataFrame(flat_data).with_columns(
                pl.lit(f"{sex_mapping[sex]}_{age}").alias("coc_name"),
                pl.lit(indicators_mapping[ind]).alias("indicator"),
            )

        except Exception as e:
            current_run.log_error(f"Error fetching {ind}|{sex}|{age}: {e!s}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(functools.partial(process_request, *p)) for p in params]
        results = [f.result() for f in futures if f.result() is not None]

    df_naomi = (
        (
            pl.concat(results, how="diagonal_relaxed")
            .with_columns(
                pl.col("name").str.to_uppercase().alias("name"),
                pl.lit(f"{year}12").alias("period"),
                (pl.col("indicator") + "_" + pl.col("coc_name")).alias("column_name"),
            )
            .pivot(
                index=["code", "name", "period"],
                columns=["column_name"],
                values="mean",
            )
        )
        if results
        else pl.DataFrame()
    )

    if df_naomi.is_empty():
        current_run.log_error(
            f"Aucune donnée retrouvée depuis NAOMI pour l'année `{year}`, "
            "il se peut qu'elle soit en cours de validation."
        )
        raise ValueError("No data fetched from NAOMI API.")

    fp_path = Path(workspace.files_path) / f"{fp_ressources}/district_mapping_naomi_dhis2.json"
    if not fp_path.exists():
        current_run.log_error(
            f"Le fichier de mapping des districts NAOMI & DHIS2 `{fp_path}` n'existe pas."
        )
        raise FileNotFoundError(f"File {fp_path} does not exist.")

    current_run.log_info(
        f"✅ Extraction des données de NAOMI pour l'année `{year}` terminée avec succès."
    )

    return (
        df_naomi.join(
            pl.DataFrame(
                data=list(json.load(Path.open(fp_path.as_posix(), "r", encoding="utf-8")).items()),
                schema=["code", "organisation_unit_id"],
                orient="row",
            ),
            how="left",
            on="code",
        )
        .select(
            ["code", "organisation_unit_id", "period"]
            + [pl.exclude(["code", "organisation_unit_id", "period", "name"])]
        )
        .with_columns(pl.col(pl.NUMERIC_DTYPES).round(0).cast(pl.Int64))
    )


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
    # fetch_dhis2_data(
    #     dhis2=dhis2,
    #     periods_list=periods_list,
    #     ou_ids=ou_ids,
    #     data_elements_uid=de_ist,
    # )
    df_ist = (
        df_ist.join(coc, left_on="category_option_combo_id", right_on="id", how="left").rename(
            {"name": "coc_name"}
        )
        # .drop("attribute_option_combo_id")
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

    # fetch_dhis2_data(
    #     dhis2=dhis2,
    #     periods_list=periods_list,
    #     ou_ids=ou_ids,
    #     data_elements_uid=de_pec,
    # )
    df_pec = (
        df_pec.join(coc, left_on="category_option_combo_id", right_on="id", how="left").rename(
            {"name": "coc_name"}
        )
        # .drop("attribute_option_combo_id")
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
    # fetch_dhis2_data(
    #     dhis2=dhis2, periods_list=periods_list, ou_ids=ou_ids, data_elements_uid=de_cons
    # )
    df_cons = (
        df_cons.join(coc, left_on="category_option_combo_id", right_on="id", how="left").rename(
            {"name": "coc_name"}
        )
        # .drop("attribute_option_combo_id")
    )
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


def fetch_dhis2_data(
    dhis2: DHIS2,
    periods_list: list[str],
    ou_ids: list[str],
    data_elements_uid: list[str],
) -> pl.DataFrame:
    """Fetch data from DHIS2 for the specified year and trimester.

    Returns:
        A Polars DataFrame containing the fetched data.
    """

    def fetch_data_element(data_element: str) -> pl.DataFrame | None:
        """Fetch a specific indicator from DHIS2 for the given periods.

        Args:
            data_element: The UID of the dataelement to fetch.

        Returns:
            A Polars DataFrame containing the fetched data elements, or None if an error occurs.
        """
        try:
            return dataframe.extract_data_elements(
                dhis2,
                data_element,
                start_date=datetime.fromisoformat(f"{periods_list[0]}01"),
                end_date=datetime.fromisoformat(f"{periods_list[-1]}01"),
                org_units=ou_ids,
                include_children=True,
            )
        except Exception:
            try:
                time.sleep(5)
                return dataframe.extract_data_elements(
                    dhis2,
                    data_element,
                    start_date=datetime.fromisoformat(f"{periods_list[0]}01"),
                    end_date=datetime.fromisoformat(f"{periods_list[-1]}01"),
                    org_units=ou_ids,
                    include_children=True,
                )
            except Exception as e:
                current_run.log_critical(
                    "Erreur survenue lors de la récupération des données "
                    f"l'élément de données {data_element}: {e!s}"
                )
                return None

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(fetch_data_element, data_element) for data_element in data_elements_uid
        ]
        results = [future.result() for future in futures if future.result() is not None]

    return pl.concat(results)


def fetch_dhis2_indicators(
    dhis2: DHIS2,
    periods_list: list[str],
    indicators_uid: list[str],
) -> pl.DataFrame:
    """Fetch indicators from DHIS2 for the specified year and trimester.

    Returns:
        A Polars DataFrame containing the fetched indicators.
    """
    # def fetch_indicator(indicator: str) -> pl.DataFrame | None:
    #     """Fetch a specific indicator from DHIS2 for the given periods.

    #     Args:
    #         indicator: The UID of the indicator to fetch.

    #     Returns:
    #         A Polars DataFrame containing the fetched indicator data, or None if an error occurs.
    #     """
    #     try:
    #         return dhis2.analytics.get(
    #             indicators=[indicator],
    #             periods=periods.get_range(start=periods_list[0], end=periods_list[-1]),
    #             org_unit_levels=[4],
    #             include_cocs=False,
    #         )
    #     except Exception:
    #         try:
    #             time.sleep(5)
    #             return dhis2.analytics.get(
    #                 indicators=[indicator],
    #                 periods=periods.get_range(start=periods_list[0], end=periods_list[-1]),
    #                 org_unit_levels=[4],
    #                 include_cocs=False,
    #             )
    #         except Exception as e:
    #             current_run.log_critical(
    #                 "Erreur survenue lors de la récupération des données de "
    #                 f"l'indicateur {indicator}: {e!s}"
    #             )
    #             return None

    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     futures = [executor.submit(fetch_indicator, ind) for ind in indicators_uid]
    #     results = [future.result() for future in futures if future.result() is not None]

    results = dhis2.analytics.get(
        indicators=indicators_uid,
        periods=periods.get_range(start=periods_list[0], end=periods_list[-1]),
        org_unit_levels=[4],
        include_cocs=False,
    )
    if results:
        df_data = pl.DataFrame(results)  # functools.reduce(operator.iadd, results, []))
        if not df_data.is_empty():
            for method in [
                dhis2.meta.add_org_unit_name_column,
                dhis2.meta.add_dx_name_column,
                dhis2.meta.add_coc_name_column,
            ]:
                with contextlib.suppress(Exception):
                    df_data = method(df_data)
    else:
        df_data = pl.DataFrame()
    return df_data


def filter_consistent_data_by_rules(
    df_data: pl.DataFrame,
    organisation_units: pl.DataFrame,
    include_inconsistent_data: bool,
    dico_rules: dict[str, str],
    workbook: pyxl.Workbook,
    pathologie: str,
) -> tuple[pl.DataFrame, pyxl.Workbook]:
    """Extract consistent data from the given DataFrame based on defined rules.

    Args:
        df_data: A Polars DataFrame containing the data to be filtered.
        organisation_units: A DataFrame containing organization unit information.
        include_inconsistent_data: A boolean indicating whether to include inconsistent data.
        dico_rules: A dictionary containing rules for consistency checks.
        workbook: A Workbook object for accessing the rules.
        pathologie: A string representing the pathology name.

    Returns:
        A tuple containing a Polars DataFrame with consistent data and the updated workbook.
    """
    if include_inconsistent_data:
        return df_data

    current_run.log_info(
        f"🛠️ Filtre et exportation des données incohérentes à partir de la matrice de cohérence "
        f"des données pour la pathologie `{pathologie}`..."
    )
    df_data_rules = df_data.to_pandas().copy()

    # df_data_rules_info = (
    #     df_data.to_pandas().copy()
    # )

    df_data_copy = df_data.to_pandas().copy().fillna(0)

    column_list = [
        col for col in df_data_rules.columns if col not in ["organisation_unit_id", "period"]
    ]

    for col in column_list:
        dico_rules_col = {
            k: v
            for k, v in dico_rules.items()
            if pyxl.utils.get_column_letter(list(df_data_copy.columns).index(col) + 3)
            in re.findall(r"\b[A-Z]{1,2}\b", v[0])
        }
        df_data_rules[col] = df_data_copy.apply(
            lambda row: get_dataframe_color_rules(row, col, dico_rules_col, df_data_rules),  # noqa: B023
            axis=1,
        )

    columns_to_check = df_data_rules.columns.difference(["organisation_unit_id", "period"])

    df_filtered = df_data_rules[df_data_rules[columns_to_check].notna().any(axis=1)]

    df_sheet = df_data.join(
        pl.DataFrame(df_filtered).select(["organisation_unit_id", "period"]),
        on=["organisation_unit_id", "period"],
        how="inner",
    )

    df_sheet = add_organisation_units(df_sheet, organisation_units)
    src_ws = workbook[pathologie]
    index_start = 6 if pathologie != "PTME" else 4
    for start, row in enumerate(
        dataframe_to_rows(df_sheet.to_pandas(), header=False, index=False), start=index_start
    ):
        for column, value in enumerate(row, start=1):
            src_ws.cell(row=start, column=column, value=value)

    df_filtered = df_data_rules[df_data_rules[columns_to_check].isna().all(axis=1)]
    return {
        "data": df_data.join(
            pl.DataFrame(df_filtered).select(["organisation_unit_id", "period"]),
            on=["organisation_unit_id", "period"],
            how="inner",
        ),
        "workbook": workbook,
    }


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
    if not any(p.endswith(("06", "12")) for p in periods_list):
        return pl.DataFrame()

    current_run.log_info(
        "⏳ Extraction des données aggrégées sur 6 mois pour la pathologie `PEC`..."
    )

    df_ind = pl.DataFrame()
    year = periods_list[0][:4]
    periods_map = {"06": f"{year}S1", "12": f"{year}S2"}
    data_elements = ["zxMJc6KPesz", "ykXAaZFC1bt"]
    collected_data = []

    for period_suffix, six_month in periods_map.items():
        if any(p.endswith(period_suffix) for p in periods_list):
            for de in data_elements:
                try:
                    data = dhis2.analytics.get(
                        periods=[periods.SixMonth(six_month)],
                        data_elements=[de],
                        org_units=ou_ids,
                    )
                    df = pl.DataFrame(data).select(["dx", "co", "ou", "pe", "value"])
                    collected_data.append(df)
                except Exception as e:
                    current_run.log_critical(
                        f"Erreur survenue lors de la récupération des données de  `{de}`: {e!s}"
                    )
                    continue

    if not collected_data:
        return pl.DataFrame()

    df_ind = (
        pl.concat(collected_data, how="diagonal_relaxed")
        .rename(
            {
                "dx": "column",
                "co": "category_option_combo_id",
                "ou": "organisation_unit_id",
                "pe": "period",
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
        .with_columns((pl.col("column_indicator") + "_" + pl.col("coc_name")).alias("column_name"))
        .pivot(
            index=["organisation_unit_id", "period"],
            columns=["column_name"],
            values="value",
        )
    )
    df_ind = df_ind.rename(lambda x: x.replace(" ", "")).with_columns(
        pl.col("period").replace({f"{year}S1": f"{year}06", f"{year}S2": f"{year}12"})
    )
    df_ind = df_ind.with_columns(
        [
            pl.col(c).cast(pl.Float64)
            for c in df_ind.columns
            if c not in ["organisation_unit_id", "period"]
        ]
    )
    df_ind = (
        df_ind.group_by(["organisation_unit_id", "period"])  # .fill_null(0)
        .agg(
            [
                pl.when(pl.col(col).is_not_null().any()).then(pl.col(col).sum()).otherwise(None)
                # pl.sum(col)
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


def transform_for_pnls_reporting(df: pl.DataFrame, map_indicators: dict[str, str]) -> pl.DataFrame:
    """Transforme le dataframe en un format standardisé.

    Args:
        df: DataFrame Polars contenant les données à transformer.
        map_indicators: Dictionnaire de mapping des indicateurs.

    Returns:
        DataFrame Polars transformé avec les colonnes standardisées.
    """
    if df.is_empty():
        return pl.DataFrame()

    df_final = pl.DataFrame()
    for col, ind in map_indicators.items():
        df_final = pl.concat(
            [
                df_final,
                df.select(
                    ["organisation_unit_id", "period"]
                    + [_ for _ in df.columns if _.startswith(col)]
                )
                .with_columns(pl.lit(ind).alias("indicateur"))
                .rename(
                    lambda col: standardize_column(col)
                    if col not in ["organisation_unit_id", "period", "indicateur"]
                    else col
                )
                .select(
                    ["organisation_unit_id", "period", "indicateur"]
                    + [pl.exclude(["organisation_unit_id", "period", "indicateur"])]
                ),
            ],
            how="diagonal_relaxed",
        )
    return df_final


def multi_replace(
    s: str, column_names_age_groups: dict[str, str] | None = COLUMN_NAME_GROUP_AGE
) -> str:
    """Replace multiple substrings in a string based on a mapping.

    Args:
        s: The input string to process.
        column_names_age_groups: A dictionary mapping substrings to their replacements.

    Returns:
        The modified string with replacements applied.
    """
    for old, new in column_names_age_groups.items():
        s = s.replace(old, new)
    return s.strip()


def process_column(
    s: str, column_names_age_groups: dict[str, str] | None = COLUMN_NAME_GROUP_AGE
) -> str | None:
    """Process a column name to map age groups and gender.

    Args:
        s: The input string to process.
        column_names_age_groups: A dictionary mapping age group substrings to their replacements.

    Returns:
        A string representing the processed column name with age group and gender, or None
        if no match is found.
    """
    for k, v in column_names_age_groups.items():
        if k in s:
            if "Féminin" in s:
                return f"{v}_F"
            if "Masculin" in s:
                return f"{v}_M"
            return v
    return None


def remplacement(formula, dataframe) -> str:  # noqa: ANN001
    """Cette fonciton permet de changer les colonnes au format lettre au nom de colonnes présent dans le dataframe
    On retranche de -3, car des colonnes ont été supprimer dans le dataframe par rapport au dataframe initial.
    Par exemple la colonne du premier indicateur qui est E est à la 5 ième position tandis que dans notre data frame il est a la 2ième position.
    """  # noqa: D205, DOC201, E501
    return f"row.{dataframe.columns[pyxl.utils.column_index_from_string(formula.group(0)) - 3]}"


def get_dataframe_color_rules(row, col, dico_rules_col, dataframe):  # noqa: ANN001, ANN201
    """Recherche de l'arrière plan du de la cellule sur évaluation des formules."""  # noqa: DOC201
    color_return = []
    for formula, color, priority in dico_rules_col.values():
        to_evaluate = re.sub(
            r"\b[A-Z]{1,2}\b",
            functools.partial(remplacement, dataframe=dataframe),
            formula,
        )
        # print(to_evaluate)
        if eval(to_evaluate):
            color_return.append([color, priority])
    try:
        return max(color_return, key=lambda item: item[1])[0]
    except (IndexError, ValueError):
        return np.nan


def standardize_column(column: str, map_age_group: dict[str, str] = MAP_AGE_GROUP) -> str:
    """Standardize a column name based on age group and gender.

    Args:
        column : The column name to standardize.
        map_age_group: A dictionary mapping age group substrings to their replacements.
        Defaults to COLUMN_NAME_GROUP_AGE.

    Returns:
        The standardized column name with gender and age group prefixes.
    """
    for key, val in map_age_group.items():
        if val in column:
            if "F" in column:
                return f"F_{key}"
            if "M" in column:
                return f"M_{key}"
            return f"nosex_{key}"
    return "nosex_noage"


def add_organisation_units(df: pl.DataFrame, organisation_units: pl.DataFrame) -> pl.DataFrame:
    """Add organisation unit names to the DataFrame.

    Args:
        df: The DataFrame to which organisation unit names will be added.
        organisation_units: A DataFrame containing organisation unit information.

    Returns:
        The DataFrame with organisation unit names added.
    """
    df = df.join(
        organisation_units.filter(pl.col("level") == 4)
        .drop(["geometry", "level"])
        .rename({"name": "etablissement_sanitaire"}),
        left_on="organisation_unit_id",
        right_on="id",
    ).with_columns(pl.col("path").str.split("/").alias("path"))

    df = df.with_columns(
        pl.col("path").list.get(2).alias("lvl_2_uid"),
        pl.col("path").list.get(3).alias("lvl_3_uid"),
    ).drop("path")

    df = (
        df.join(
            organisation_units.filter(organisation_units["level"] == 3)[["id", "name"]].rename(
                {"name": "district"}
            ),
            left_on="lvl_3_uid",
            right_on="id",
            how="left",
        ).join(
            organisation_units.filter(organisation_units["level"] == 2)[["id", "name"]].rename(
                {"name": "region"}
            ),
            left_on="lvl_2_uid",
            right_on="id",
            how="left",
        )
    ).drop(["lvl_2_uid", "lvl_3_uid"])
    return (
        df.drop("organisation_unit_id")
        .select(
            ["region", "district", "etablissement_sanitaire", "period"]
            + [pl.exclude(["region", "district", "etablissement_sanitaire", "period"])]
        )
        .with_columns(
            pl.col("period")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime("ns"), "%Y%m")
            .cast(pl.Date)
            .alias("period")
        )
    ).sort(["period", "region", "district"])


def export_file(df: pl.DataFrame, fp_historical_data: str, annee_extraction: int):
    """Export the DataFrame to a Parquet file.

    Args:
        df: The DataFrame to export.
        fp_historical_data: The path to the historical data directory.
        annee_extraction: The year of data extraction.
    """
    dst_dir = Path(workspace.files_path) / f"{fp_historical_data}/{annee_extraction}_OH"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for period in df.select("periode").sort("periode").unique().to_series().to_list():
        file_name = f"{period}.parquet"
        df.filter(pl.col("periode") == period).write_parquet(dst_dir / file_name)
        current_run.add_file_output((dst_dir / file_name).as_posix())

    current_run.log_info(
        "Le fichier de données consolidées a été exporté avec succès dans "
        f"le repertoire `{dst_dir.as_posix()}`"
    )
