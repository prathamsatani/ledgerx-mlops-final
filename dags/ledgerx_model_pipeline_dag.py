from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "ledgerx",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="ledgerx_model_pipeline",
    default_args=default_args,
    description="ML model pipeline for LedgerX",
    schedule_interval=None,
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["ledgerx", "model", "pipeline"],
) as dag:

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/src/model/train_model.py",
    )

    tune_model = BashOperator(
        task_id="tune_model",
        bash_command="python /opt/airflow/src/model/tune_model.py",
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="python /opt/airflow/src/model/register_model.py",
    )

    run_model_tests = BashOperator(
        task_id="run_model_tests",
        bash_command=(
            "pytest -v --disable-warnings /opt/airflow/tests/model "
            "> /opt/airflow/reports/model_test_report.txt || true"
        ),
    )

    train_model >> tune_model >> run_model_tests >> register_model
