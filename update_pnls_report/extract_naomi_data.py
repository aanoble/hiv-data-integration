"""Extract naomi data from the API and process it into a Polars DataFrame."""

import functools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import requests
from openhexa.sdk import current_run, workspace


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
