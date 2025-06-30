"""Template for newly generated pipelines."""

from openhexa.sdk import workspace, current_run, pipeline
import papermill as pm


@pipeline("refresh-data", name="Pipeline de Rafraichissement des donnees du TDB PBI")
def refresh_data():
    """Write your pipeline orchestration here.

    Pipeline functions should only call tasks and should never perform IO operations or expensive computations.
    """
    run_notebook()


@refresh_data.task
def run_notebook():
    """Put some data processing code here."""
    current_run.log_info("Start to run jupyter notebook")
    pm.execute_notebook(
        workspace.files_path + "/pipeline_automation/main_program.ipynb",
        workspace.files_path + "/pipeline_automation/output_main_program.ipynb",
    )

    current_run.log_info("Done!")


if __name__ == "__main__":
    refresh_data()
