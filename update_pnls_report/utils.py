import contextlib
import functools
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl as pyxl
import polars as pl
from constants import (
    COLUMN_NAME_GROUP_AGE,
    MAP_AGE_GROUP,
)
from fuzzywuzzy import fuzz, process
from openhexa.sdk import current_run, workspace
from openhexa.toolbox.dhis2 import DHIS2, dataframe, periods
from openpyxl.utils.dataframe import dataframe_to_rows


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
    start, end = periods_list[0], periods_list[-1]
    month_range = periods.Month.from_string(start).range(periods.Month.from_string(end))
    periods_range = [month.period for month in month_range]
    results = dhis2.analytics.get(
        indicators=indicators_uid,
        periods=periods_range,
        org_unit_levels=[4],
        include_cocs=False,
    )
    if results:
        df_data = pl.DataFrame(results)
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
    current_run.log_info(
        f"Consolidation de la matrice de cohérence des données pour la pathologie `{pathologie}`..."
    )
    df_data_rules = df_data.to_pandas().copy()

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

    if not include_inconsistent_data:
        current_run.log_info(
            f"🛠️ Filtre des données incohérentes à partir de la matrice de cohérence "
            f"des données pour la pathologie `{pathologie}`..."
        )
        df_data = df_data.join(
            pl.DataFrame(df_filtered).select(["organisation_unit_id", "period"]),
            on=["organisation_unit_id", "period"],
            how="inner",
        )
    return {
        "data": df_data,
        "workbook": workbook,
    }


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
        file_name = f"{period}.csv"
        df.filter(pl.col("periode") == period).write_csv(dst_dir / file_name)
        current_run.add_file_output((dst_dir / file_name).as_posix())

    current_run.log_info(
        "Le fichier de données consolidées a été exporté avec succès dans "
        f"le repertoire `{dst_dir.as_posix()}`"
    )


def generate_org_unit_uuid(col: str, namespace: uuid.UUID = uuid.NAMESPACE_DNS) -> str:
    """Generate a UUID for an organization unit based on its name and a namespace.

    Args:
        col (str): The name or identifier of the organization unit.
        namespace (uuid.UUID, optional): The namespace to use for UUID generation. 
                    Defaults to uuid.NAMESPACE_DNS.

    Returns:
        str: The generated UUID as a string without hyphens.
    """
    return str(uuid.uuid5(namespace, col)).replace("-", "")


def find_best_match(element: str, values: list[str], threshold: int = 95) -> int | None:
    """Finds the best match for a given element within a list of values using fuzzy matching.

    This function searches for the exact position of an element in a list of values. If the exact
    element is not found, it uses fuzzy matching to find the closest match based on a specified
    threshold.

    Args:
        element (str): The element to search for in the list of values.
        values (list of str): The list of values to search within.
        threshold (int, optional): The minimum score for a fuzzy match to be considered valid.
            Defaults to 95.

    Returns:
        int or None: The 1-based index of the best match in the list of values, or None if no
            match is found.
    """
    try:
        if element in values:
            return values.index(element) + 1
        best_match = process.extractOne(element, values, scorer=fuzz.token_set_ratio)
        if best_match[1] >= threshold:
            return values.index(best_match[0]) + 1
        return None
    except Exception:
        return None


def find_best_match_org_unit_chu(
    col: str,
    org_unit_list: list,
    organisation_units: pl.DataFrame,
    threshold: int = 90,
) -> str | None:
    """Find the best matching organization unit path for a given name using fuzzy matching.

    Args:
        col (str): The name to match.
        org_unit_list (list): List of organization unit names.
        organisation_units (pl.DataFrame): DataFrame containing organization unit information.
        threshold (int, optional): Minimum score for a match to be considered valid. Defaults to 90.

    Returns:
        str or None: The path of the best matching organization unit, or None if no match is found.
    """
    try:
        best_match = process.extractOne(col, org_unit_list, scorer=fuzz.token_set_ratio)
        if best_match[1] >= threshold:
            return organisation_units.filter(pl.col("name") == best_match[0])["path"][0]
    except Exception:
        return None
