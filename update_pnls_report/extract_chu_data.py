import functools
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from constants import SHEET_RENAMES
from openhexa.sdk import current_run, workspace
from utils import (
    find_best_match,
    generate_org_unit_uuid,
    match_org_unit_chu,
    match_org_unit_with_data,
    multi_replace,
    rename_or_drop_column_if_found,
    transform_for_pnls_reporting,
)


def extract_data_from_excel_file(
    file_path: Path, fp_ressources: Path, organisation_units: pl.DataFrame, periods_list: list[str]
) -> pl.DataFrame:
    """Extract data from an Excel file.

    Args:
        file_path: The path to the Excel file.
        fp_ressources: The path to the resources directory.
        organisation_units: A Polars DataFrame containing existing organisation units.
        periods_list: A list of periods to filter the data.

    Returns:
        A Polars DataFrame containing the extracted data.
    """
    sheetnames = pd.ExcelFile(file_path).sheet_names
    current_run.log_info(
        f"📄 Extraction des données à partir du fichier Excel des CHU `{file_path.name}`..."
    )
    df_cd = extract_data_from_sheet(file_path, "CD", sheetnames)
    df_pec = extract_data_from_sheet(file_path, "PEC", sheetnames)
    df_ptme = extract_data_from_sheet(file_path, "PTME", sheetnames)
    current_run.log_info("✅ Extraction des données des CHU réussie.")

    current_run.log_info(
        "🔄 Standardisation des unités d'organisation des CHU des données extraites..."
    )
    df_org_unit_current = standardize_org_units(
        df_cd, df_pec, df_ptme, organisation_units, fp_ressources
    )

    df_cd = merge_with_chu_org_units(df_cd, df_org_unit_current)
    df_pec = merge_with_chu_org_units(df_pec, df_org_unit_current)
    df_ptme = merge_with_chu_org_units(df_ptme, df_org_unit_current)

    current_run.log_info(
        "⏳ Extraction et consolidation des données agrégées pour la pathologie PEC..."
    )
    df_pec_agg = fetch_chu_pec_aggregates(df_pec, fp_ressources, periods_list)
    current_run.log_info("✅ Consolidation des données agrégées pour la pathologie PEC réussie.")

    current_run.log_info(
        "🔄 Consolidation des données des CHU extraites pour les pathologies CD, PEC et PTME "
        "suivant le format attendu pour le rapport VIH pédiatrique..."
    )
    df_cd = transform_for_pnls_reporting(
        df_cd.select(
            ["organisation_unit_id", "periode"]
            + [col for col in df_cd.columns if col.startswith(("indicateur_11", "indicateur_12"))],
        ).rename({"periode": "period"}),
        map_indicators={"indicateur_11_": 1, "indicateur_12_": 2},
    )

    df_pec = transform_for_pnls_reporting(
        df_pec.select(
            ["organisation_unit_id", "periode"]
            + [
                col
                for col in df_pec.columns
                if col.startswith(
                    (
                        "indicateur_10",
                        "indicateur_11",
                        "indicateur_8",
                        "indicateur_9",
                        "indicateur_17",
                        "indicateur_18",
                        "indicateur_1",
                    )
                )
            ]
        ).rename({"periode": "period"}),
        map_indicators={
            "indicateur_10_": 5,
            "indicateur_11_": 6,
            "indicateur_8_": 7,
            "indicateur_9_": 8,
            "indicateur_17_": 12,
            "indicateur_18_": 13,
            "indicateur_1_": 16,
        },
    )

    df_pec_agg = transform_for_pnls_reporting(
        df=df_pec_agg.rename({"periode": "period"}),
        map_indicators={
            "indicateur_11": 11,
            "indicateur_14": 14,
        },
    )

    df_ptme = transform_for_pnls_reporting(
        df_ptme.select(
            ["organisation_unit_id", "periode"]
            + [
                col
                for col in df_ptme.columns
                if col.startswith(
                    (
                        "indicateur_31",
                        "indicateur_12",
                    )
                )
            ]
        ).rename({"periode": "period"}),
        map_indicators={
            "indicateur_31": 4,
            "indicateur_12": 15,
        },
    )

    df_final = pl.concat(
        [
            df_cd,
            df_pec,
            df_pec_agg,
            df_ptme,
        ],
        how="diagonal_relaxed",
    )

    df_final = (
        df_final.join(
            organisation_units.filter(pl.col("level").is_in([4, 3])).select(["id", "path"]),
            left_on="organisation_unit_id",
            right_on="id",
            how="left",
        )
        .with_columns(
            pl.col("path")
            .str.replace_all("/", "_")
            .str.replace_all(r"_ZD44Asc0bAk_", "", literal=True)
            .alias("organisation_unit_id"),
            pl.col("period")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime("ns"), "%Y%m")
            .cast(pl.Date)
            .alias("period"),
        )
        .drop("path")
        .rename({"period": "periode", "indicateur": "Indicateur", "organisation_unit_id": "idsite"})
    )
    return df_final.with_columns(
        [
            pl.lit(None).alias(col)
            for col in ["M_<15 ans", "M_>15 ans", "F_<15 ans", "F_>15 ans"]
            if col not in df_final.columns
        ]
    )


def extract_data_from_sheet(
    file_path: Path, sheet_name: str, sheetnames: list[str]
) -> pl.DataFrame:
    """Extract data from a specific sheet in an Excel file.

    Args:
        file_path: The path to the Excel file.
        sheet_name: The name of the sheet to extract data from.
        sheetnames: A list of available sheet names in the Excel file.

    Returns:
        A Polars DataFrame containing the extracted data.
    """
    if (sheet_name_index := find_best_match(sheet_name, sheetnames)) is None:
        current_run.log_error(
            f"La feuille '{sheet_name}' n'est pas présente dans le fichier Excel. "
            f"Feuilles disponibles : {sheetnames}"
        )
        raise ValueError(f"La feuille '{sheet_name}' n'est pas présente dans le fichier Excel.")

    df = pl.read_excel(file_path, sheet_name=sheetnames[sheet_name_index])
    df = df.rename(lambda x: x.strip())

    # Gestion des colonnes spéciales
    df, _ = rename_or_drop_column_if_found(df, "regions", ["Région", "région"], drop=True)
    df, district_here = rename_or_drop_column_if_found(
        df, "districts_sanitaires", ["Districts", "districts"]
    )
    df, _ = rename_or_drop_column_if_found(
        df, "formations_sanitaires", ["Formations sanitaires", "Etablissements"]
    )
    df, _ = rename_or_drop_column_if_found(df, "service", ["Service", "service"], drop=True)
    df, _ = rename_or_drop_column_if_found(df, "periode", ["Mois"])

    # Renommage spécifique par type de feuille
    if sheet_name in SHEET_RENAMES:
        for orig, new in SHEET_RENAMES[sheet_name].items():
            if (idx := find_best_match(orig, df.columns)) is not None:
                df = df.rename({df.columns[idx]: new})

    # Nettoyage de données
    for col in df.columns:
        if (
            col not in {"districts_sanitaires", "formations_sanitaires", "periode"}
            and col not in df.select(pl.selectors.numeric()).columns
        ):
            df = df.with_columns(pl.col(col).map_elements(multi_replace, return_dtype=pl.String))

        elif col in {"districts_sanitaires", "formations_sanitaires"}:
            df = df.with_columns(pl.col(col).str.strip_chars())

    if sheet_name != "PTME":
        start_idx = 3 if district_here else 2
        list_columns_indexing = df.columns[start_idx:]
        for col in list_columns_indexing:
            val = next(row[col] for row in df.iter_rows(named=True))
            if val:
                df = df.rename(
                    {
                        col: functools.reduce(
                            lambda a, b: a if ("__UNNAMED" not in a) and ("__UNNAMED" in b) else b,
                            list_columns_indexing[: list_columns_indexing.index(col) + 1],
                        )
                        + "_"
                        + val
                    }
                )
        list_columns_indexing = df.columns[start_idx:]
        for col in list_columns_indexing:
            val = [row[col] for row in df.iter_rows(named=True)][1]
            if val:
                df = df.rename(
                    {
                        col: functools.reduce(
                            lambda a, b: a if ("__UNNAMED" not in a) and ("__UNNAMED" in b) else b,
                            list_columns_indexing[: list_columns_indexing.index(col) + 1],
                        )
                        + "_"
                        + val
                    }
                )

    # Sélection finale des colonnes
    keep_cols = [
        col
        for col in df.columns
        if col in {"districts_sanitaires", "formations_sanitaires", "periode"}
        or col.startswith("indicateur_")
    ]

    return df.select(keep_cols)[2:] if sheet_name != "PTME" else df.select(keep_cols)


def standardize_org_units(
    df_cd: pl.DataFrame,
    df_pec: pl.DataFrame,
    df_ptme: pl.DataFrame,
    organisation_units: pl.DataFrame,
    fp_ressources: Path,
    ou_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Standardize organisation units in the DataFrame.

    Args:
        df_cd: DataFrame containing CD data.
        df_pec: DataFrame containing PEC data.
        df_ptme: DataFrame containing PTME data.
        organisation_units: Polars DataFrame containing existing organisation units.
        fp_ressources: Path to the resources directory.
        ou_columns: Optional list of columns to standardize.
                    Defaults to ["districts_sanitaires", "formations_sanitaires"].

    Returns:
        A Polars DataFrame with standardized organisation units.
    """
    if ou_columns is None:
        ou_columns = ["districts_sanitaires", "formations_sanitaires"]

    fp_org_unit_chu = Path(workspace.files_path) / f"{fp_ressources}/org_unit_chu.parquet"
    if not fp_org_unit_chu.exists():
        msg_error = (
            f"Le fichier des unités d'organisation des CHU `{fp_org_unit_chu}` n'existe pas."
        )
        current_run.log_error(msg_error)
        raise FileNotFoundError(f"File {fp_org_unit_chu} does not exist.")

    df_org_unit_chu = pl.read_parquet(fp_org_unit_chu)

    df_org_unit_current = pl.concat(
        [
            df_cd.select([col for col in df_cd.columns if col in ou_columns]),
            df_pec.select([col for col in df_pec.columns if col in ou_columns]),
            df_ptme.select([col for col in df_ptme.columns if col in ou_columns]),
        ],
        how="diagonal_relaxed",
    ).unique()

    df_org_unit_current = df_org_unit_current.with_columns(
        pl.col("formations_sanitaires")
        .map_elements(
            lambda col: match_org_unit_with_data(
                col, df_org_unit_chu["formations_sanitaires"].unique().to_list()
            ),
            return_dtype=pl.String,
        )
        .alias("formations_sanitaires_new"),
    )

    if df_org_unit_current["formations_sanitaires_new"].is_null().any():
        # S'il y a un élément nulle l'idée est de mettre à jour le fichier des unités d'organisation
        df_org_unit_chu = pl.concat(
            [
                df_org_unit_chu,
                df_org_unit_current.filter(pl.col("formations_sanitaires_new").is_null()).select(
                    [col for col in df_org_unit_current.columns if col in ou_columns]
                ),
            ],
            how="diagonal_relaxed",
        )

        df_org_unit_chu = df_org_unit_chu.with_columns(
            pl.when(pl.col("organisation_unit_id").is_null())
            .then(
                pl.col("formations_sanitaires").map_elements(
                    lambda col: match_org_unit_chu(
                        col,
                        organisation_units.filter(pl.col("level") == 4)["name"],
                        organisation_units,
                    ),
                    return_dtype=pl.String,
                )
            )
            .otherwise(pl.col("organisation_unit_id"))
            .alias("organisation_unit_id"),
        )

        df_org_unit_chu = df_org_unit_chu.with_columns(
            pl.when(pl.col("organisation_unit_id").is_null())
            .then(
                pl.col("formations_sanitaires").map_elements(
                    lambda col: match_org_unit_chu(
                        col,
                        organisation_units.filter(pl.col("level") == 3)["name"],
                        organisation_units,
                    ),
                    return_dtype=pl.String,
                )
            )
            .otherwise(pl.lit(None))
            .alias("districts_sanitaires_id_new"),
        )

        # Génération de l'ID d'unité d'organisation
        # si l'ID d'unité d'organisation est toujours null
        df_org_unit_chu = df_org_unit_chu.with_columns(
            pl.struct(
                ["organisation_unit_id", "formations_sanitaires", "districts_sanitaires_id_new"]
            )
            .map_elements(
                lambda row: row["organisation_unit_id"]
                if row["organisation_unit_id"] is not None
                else (
                    f"{row['districts_sanitaires_id_new']}/"
                    f"{generate_org_unit_uuid(row['formations_sanitaires'])}"
                )
            )
            .alias("organisation_unit_id")
        )
    # Exportation du dataframe des unités d'organisation CHU
    df_org_unit_chu = df_org_unit_chu.select(["organisation_unit_id"] + ou_columns)
    df_org_unit_chu.write_parquet(fp_org_unit_chu)

    df_org_unit_current = df_org_unit_current.with_columns(
        pl.when(pl.col("formations_sanitaires_new").is_not_null())
        .then(
            pl.col("formations_sanitaires").map_elements(
                lambda col: match_org_unit_with_data(
                    col, df_org_unit_chu["formations_sanitaires"].unique().to_list()
                ),
                return_dtype=pl.String,
            )
        )
        .otherwise(pl.lit("formations_sanitaires_new"))
        .alias("formations_sanitaires_new"),
    )

    keep_cols = (
        ["districts_sanitaires"] if "districts_sanitaires" in df_org_unit_current.columns else []
    )

    df_org_unit_current = df_org_unit_current.join(
        df_org_unit_chu.select(["organisation_unit_id"] + keep_cols + ["formations_sanitaires"]),
        left_on=keep_cols + ["formations_sanitaires_new"],
        right_on=keep_cols + ["formations_sanitaires"],
        how="inner",
    )

    return df_org_unit_current.select(
        ["organisation_unit_id"] + keep_cols + ["formations_sanitaires"]
    )


def merge_with_chu_org_units(data: pl.DataFrame, df_org_unit: pl.DataFrame) -> pl.DataFrame:
    """Format data with organisation unit CHU.

    Args:
        data: A Polars DataFrame containing the data to format.
        df_org_unit: A Polars DataFrame containing organisation unit information.

    Returns:
        A Polars DataFrame with formatted data including organisation unit IDs and periods.

    """
    list_columns = [col for col in data.columns if col.startswith("indicateur_")]
    keep_cols = (
        ["districts_sanitaires"]
        if "districts_sanitaires" in df_org_unit.columns and "districts_sanitaires" in data.columns
        else []
    )
    data = (
        data.join(
            df_org_unit,
            on=keep_cols + ["formations_sanitaires"],
            how="inner",
        )
        .select(["organisation_unit_id", "periode"] + list_columns)
        .with_columns(pl.col("periode").cast(pl.Date).alias("periode"))
    )

    def convert_value(val: str) -> float:
        try:
            if val in ["", "0"]:
                return 0
            return float(val)
        except Exception:
            return 0

    list_columns = [
        col for col in list_columns if col not in data.select(pl.selectors.numeric()).columns
    ]

    data = data.with_columns(
        [
            pl.col(c)
            .cast(pl.Utf8)
            .str.replace_all('"', "")
            .str.strip_chars()
            .map_elements(
                convert_value,
                return_dtype=pl.Float64,
            )
            .alias(c)
            for c in list_columns
        ]
    )

    return data.with_columns(pl.col(pl.NUMERIC_DTYPES).round(0).cast(pl.Int64))


def fetch_chu_pec_aggregates(
    df_pec: pl.DataFrame,
    fp_ressources: Path,
    periods_list: list[str],
) -> pl.DataFrame:
    """Get historical data for PEC.

    Args:
        df_pec: A Polars DataFrame containing the current data.
        fp_ressources: Path to the resources directory.
        periods_list: A list of periods to filter the historical data.

    Returns:
        A Polars DataFrame with historical PEC data merged with the current data.
    """
    fp_historique_pec = (
        Path(workspace.files_path) / f"{fp_ressources}/historique_data_pec_chu.parquet"
    )
    if not fp_historique_pec.exists():
        msg_error = f"Le fichier des historiques de PEC des CHU `{fp_historique_pec}` n'existe pas."
        current_run.log_error(msg_error)
        raise FileNotFoundError(f"File {fp_historique_pec} does not exist.")
    df_historique_pec = pl.read_parquet(fp_historique_pec)

    df_historique_pec = pl.concat([df_pec, df_historique_pec], how="diagonal_relaxed").sort(
        by=["organisation_unit_id", "periode"],
    )
    # Mise à jour des données historiques suivant les nouvelles données
    df_historique_pec.write_parquet(fp_historique_pec)

    year = int(periods_list[0][:4])
    periods_map = {
        "03": (f"{year - 1}10", f"{year}03"),
        "06": (f"{year}01", f"{year}06"),
        "09": (f"{year}04", f"{year}09"),
        "12": (f"{year}07", f"{year}12"),
    }
    collected_data = []
    valid_suffixes = {
        s for s in ("03", "06", "09", "12") if any(p.endswith(s) for p in periods_list)
    }

    for period_suffix in valid_suffixes:
        start, end = [datetime.strptime(col + "01", "%Y%m%d") for col in periods_map[period_suffix]]
        df = (
            df_historique_pec.select(
                ["organisation_unit_id", "periode"]
                + [
                    col
                    for col in df_historique_pec.columns
                    if col.startswith(("indicateur_11_", "indicateur_14_"))
                ]
            )
            .filter((pl.col("periode") >= start) & (pl.col("periode") <= end))
            .with_columns(pl.lit(f"{year}{period_suffix}").alias("periode"))
        )

        collected_data.append(df)

    if not collected_data:
        return pl.DataFrame()

    df_pec_agg = pl.concat(collected_data, how="diagonal_relaxed")

    return df_pec_agg.with_columns(
        pl.col("periode").cast(pl.Utf8).str.strptime(pl.Datetime("ns"), "%Y%m").cast(pl.Date)
    )
