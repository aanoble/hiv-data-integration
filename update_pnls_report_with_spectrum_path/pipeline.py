"""Template for newly generated pipelines."""

from datetime import datetime
from pathlib import Path

import openpyxl as pyxl
import papermill as pm
import polars as pl
from constants import (
    DICO_RULES_IST,
    DICO_RULES_PEC,
    DICO_RULES_PTME,
)
from openhexa.sdk import DHIS2Connection, current_run, parameter, pipeline, workspace
from openhexa.toolbox.dhis2 import DHIS2
from utils import (
    export_file,
    extract_dhis2_consultant_data,
    extract_dhis2_ist_data,
    extract_dhis2_pec_aggregated_data,
    extract_dhis2_pec_data,
    extract_dhis2_ptme_data,
    extract_spectrum_data,
    filter_consistent_data_by_rules,
    transform_for_pnls_reporting,
)


@pipeline("update_pnls_report_with_spectrum_path")
@parameter(
    "dhis2_connection",
    type=DHIS2Connection,
    name="DHIS2 connection",
    required=True,
    default="sig-sante-civ-prod",
)
@parameter(
    "fp_spectrum",
    type=str,
    name="Fichier de données Spectrum",
    help=(
        "Fichier de données Spectrum préalablement standardisé et chargé dans le dossier "
        "`data/données spectrum/` à partir de l'interface de chargement des fichiers."
    ),
    required=True,
)
@parameter(
    "trimestres",
    type=str,
    name="Trimestre d'extraction des données DHIS2",
    required=True,
    multiple=True,
    choices=[
        "T1 (Janvier - Mars)",
        "T2 (Avril - Juin)",
        "T3 (Juillet - Septembre)",
        "T4 (Octobre - Décembre)",
    ],
)
@parameter("annee_extraction", type=int, name="Année d'extraction", required=True)
@parameter(
    "include_inconsistent_data",
    type=bool,
    name="Inclure les données incohérentes ?",
    help=(
        "Ce paramètre permet à l'utilisateur d'inclure les données incohérentes celles, "
        "c'est-à-dire celles ne respectant pas un ensemble de règles de cohérence définies dans "
        "la matrice de cohérence des données. ‼️(En règle générale, il est préférable de ne pas "
        "les inclure)."
    ),
    required=False,
    default=False,
)
def update_pnls_report_with_spectrum_path(
    dhis2_connection: DHIS2Connection,
    fp_spectrum: str,
    trimestres: list[str],
    annee_extraction: int,
    include_inconsistent_data: bool = False,
    fp_historical_data: str = "data/données historiques/",
    fp_ressources: str = "data/matrice de cohérence/ressources/",
    fp_matrix: str = "data/matrice de cohérence/Matrice de Cohérence BASE NEW.xlsx",
):
    """Write your pipeline orchestration here.

    Pipeline functions should only call tasks and should never perform IO operations
    or expensive computations.
    """
    periods_list = generate_extraction_periods(year=annee_extraction, trimestres=trimestres)

    dhis2 = DHIS2(
        connection=dhis2_connection,
        # cache_dir=Path(workspace.files_path, ".cache"),
    )

    df_final = consolidate_dhis2_and_naomi_data(
        dhis2=dhis2,
        fp_spectrum=fp_spectrum,
        annee_extraction=annee_extraction,
        periods_list=periods_list,
        fp_ressources=fp_ressources,
        fp_matrix=fp_matrix,
        include_inconsistent_data=include_inconsistent_data,
    )

    run_notebook_update_pnls_report_with_spectrum_path(
        df=df_final, fp_historical_data=fp_historical_data, annee_extraction=annee_extraction
    )


# @update_pnls_report_with_spectrum_path.task
def consolidate_dhis2_and_naomi_data(
    dhis2: DHIS2,
    fp_spectrum: str,
    annee_extraction: int,
    periods_list: list[str],
    fp_ressources: str,
    fp_matrix: str,
    include_inconsistent_data: bool,
) -> pl.DataFrame:
    """Fetch data from DHIS2.

    Args:
        dhis2: The DHIS2 instance.
        fp_spectrum: The path to the Spectrum data file.
        annee_extraction: The year of data extraction.
        periods_list: A list of periods for which to fetch the data.
        fp_ressources: The path to the resources directory.
        fp_matrix: The path to the matrix directory.
        include_inconsistent_data: A boolean indicating whether to include inconsistent data.

    Returns:
        A Polars DataFrame containing the fetched data.
    """
    df_naomi = extract_spectrum_data(fp_spectrum=fp_spectrum)

    coc = pl.DataFrame(dhis2.meta.category_option_combos())
    organisation_units = pl.DataFrame(dhis2.meta.organisation_units())
    ou_ids = (
        organisation_units.filter(pl.col("level") == 4).select("id").unique().to_series().to_list()
    )

    workbook = pyxl.load_workbook(Path(workspace.files_path, fp_matrix))

    df_ist = extract_dhis2_ist_data(
        dhis2=dhis2,
        periods_list=periods_list,
        ou_ids=ou_ids,
        coc=coc,
        fp_ressources=fp_ressources,
    )
    dico_ist = filter_consistent_data_by_rules(
        df_data=df_ist,
        organisation_units=organisation_units,
        include_inconsistent_data=include_inconsistent_data,
        dico_rules=DICO_RULES_IST,
        workbook=workbook,
        pathologie="IST CD",
    )
    df_ist = dico_ist["data"]
    workbook = dico_ist["workbook"]

    df_pec = extract_dhis2_pec_data(
        dhis2=dhis2,
        periods_list=periods_list,
        ou_ids=ou_ids,
        coc=coc,
        fp_ressources=fp_ressources,
    )
    dico_pec = filter_consistent_data_by_rules(
        df_data=df_pec,
        organisation_units=organisation_units,
        include_inconsistent_data=include_inconsistent_data,
        dico_rules=DICO_RULES_PEC,
        workbook=workbook,
        pathologie="PEC",
    )
    df_pec, workbook = dico_pec["data"], dico_pec["workbook"]
    df_pec_agg = extract_dhis2_pec_aggregated_data(
        dhis2=dhis2,
        ou_ids=df_pec.select("organisation_unit_id").unique().to_series().to_list(),
        coc=coc,
        periods_list=periods_list,
    )

    df_ptme = extract_dhis2_ptme_data(
        dhis2=dhis2,
        periods_list=periods_list,
        ou_ids=ou_ids,
        coc=coc,
        fp_ressources=fp_ressources,
    )
    dico_ptme = filter_consistent_data_by_rules(
        df_data=df_ptme,
        organisation_units=organisation_units,
        include_inconsistent_data=include_inconsistent_data,
        dico_rules=DICO_RULES_PTME,
        workbook=workbook,
        pathologie="PTME",
    )
    df_ptme, workbook = dico_ptme["data"], dico_ptme["workbook"]

    df_cons = extract_dhis2_consultant_data(
        dhis2=dhis2,
        periods_list=periods_list,
        ou_ids=ou_ids,
        coc=coc,
    )

    msg_info = (
        "Consolidation des données pour l'alimentation du rapport PBI VIH pédiatrique "
        "à partir des données résultantes..."
    )
    current_run.log_info(msg_info)
    df_naomi = transform_for_pnls_reporting(
        df=df_naomi.drop("code"),
        map_indicators={"indicateur_9": 9, "indicateur_10": 10},
    )
    # Les données de NAOMI sont extraits chaque trimestre
    df_naomi = (
        df_naomi.lazy()
        .join(
            pl.DataFrame(
                {"suffix": [p[-2:] for p in periods_list if p.endswith(("03", "06", "09", "12"))]}
            ).lazy(),
            how="cross",
        )
        .with_columns(
            pl.col("period").str.replace(
                f"{annee_extraction}12", f"{annee_extraction}" + pl.col("suffix")
            )
        )
        .drop("suffix")
        .collect()
    )

    df_ist = transform_for_pnls_reporting(
        df=df_ist.select(
            ["organisation_unit_id", "period"]
            + [col for col in df_ist.columns if col.startswith(("indicateur_11", "indicateur_12"))]
        ),
        map_indicators={"indicateur_11_": 1, "indicateur_12_": 2},
    )

    df_pec = transform_for_pnls_reporting(
        df=df_pec.select(
            ["organisation_unit_id", "period"]
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
        ),
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
        df=df_pec_agg,
        map_indicators={
            "indicateur_11": 11,
            "indicateur_14": 14,
        },
    )

    df_ptme = transform_for_pnls_reporting(
        df=df_ptme.select(
            ["organisation_unit_id", "period"]
            + [
                col
                for col in df_ptme.columns
                if col.startswith(
                    (
                        "indicateur_31",
                        "indicateur_12",
                        "indicateur_17",
                    )
                )
            ]
        ),
        map_indicators={
            "indicateur_31": 4,
            "indicateur_12": 15,
        },
    )

    df_cons = transform_for_pnls_reporting(df=df_cons, map_indicators={"indicateur_3": 3})

    df_final = pl.concat(
        [
            df_ist,
            df_pec,
            df_pec_agg,
            df_ptme,
            df_cons,
            df_naomi,
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
    df_final = df_final.with_columns(
        [
            pl.lit(None).alias(col)
            for col in ["M_<15 ans", "M_>15 ans", "F_<15 ans", "F_>15 ans"]
            if col not in df_final.columns
        ]
    )
    current_run.log_info("✅ Consolidation des données effectuée avec succès.")
    dest_path = Path(workspace.files_path) / "data/matrice de cohérence/output_pipelines/"
    dest_path.mkdir(parents=True, exist_ok=True)
    file_output = (
        dest_path
        / f"matrice_de_coherence_{'_'.join([p[-2:] for p in periods_list])}_{annee_extraction}.xlsx"
    )
    workbook.save(file_output)
    current_run.add_file_output(file_output.as_posix())
    current_run.log_debug(f"{len(df_final)} lignes de données consolidées.")
    current_run.log_debug(f"{df_final.columns}")
    return df_final


# @update_pnls_report_with_spectrum_path.task
def generate_extraction_periods(
    year: int,
    trimestres: list[str],
) -> list[str]:
    """Process the extraction periods based on the selected trimester and year.

    Returns:
        A list of strings representing the processed periods.
    """
    periods_list = []
    for trimestre in trimestres:
        if trimestre == "T1 (Janvier - Mars)":
            periods_list.extend([f"{year}01", f"{year}02", f"{year}03"])
        elif trimestre == "T2 (Avril - Juin)":
            periods_list.extend([f"{year}04", f"{year}05", f"{year}06"])
        elif trimestre == "T3 (Juillet - Septembre)":
            periods_list.extend([f"{year}07", f"{year}08", f"{year}09"])
        elif trimestre == "T4 (Octobre - Décembre)":
            periods_list.extend([f"{year}10", f"{year}11", f"{year}12"])
        else:
            msg_error = (
                f"Invalid trimester selection: {trimestre}. "
                "Please select one of the following: `T1 (Janvier - Mars)`, `T2 (Avril - Juin)`, "
                "`T3 (Juillet - Septembre)`, or `T4 (Octobre - Décembre)`."
            )
            current_run.log_error(msg_error)
            raise ValueError(msg_error)
    return periods_list


# @update_pnls_report_with_spectrum_path.task
def run_notebook_update_pnls_report_with_spectrum_path(
    df: pl.DataFrame, fp_historical_data: str, annee_extraction: int
) -> None:
    """Run a Jupyter notebook for data extraction.

    Args:
    df: The DataFrame to export.
    fp_historical_data: The path to the historical data directory.
    annee_extraction: The year of data extraction.
    """
    export_file(df, fp_historical_data=fp_historical_data, annee_extraction=annee_extraction)

    current_run.log_info(
        "Exécution du notebook de rafraîchissement des données du rapport PVVIH pédiatrique."
    )
    timestamp = datetime.now().strftime("%Y-%m-%d")
    input_path = Path(workspace.files_path, "pipeline_automation/main_program.ipynb")
    output_path = Path(
        workspace.files_path, f"pipeline_automation/execution/output_main_program_{timestamp}.ipynb"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pm.execute_notebook(
        input_path=input_path.as_posix(),
        output_path=output_path.as_posix(),
    )
    current_run.log_info("✅ Le pipeline a été exécuté avec succès. ✅")


if __name__ == "__main__":
    update_pnls_report_with_spectrum_path()
