from __future__ import annotations

import mlflow
from typing import Dict, Any


def start_run(experiment: str, run_name: str = None):
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


def log_params(params: Dict[str, Any]):
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: int = None):
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str):
    mlflow.log_artifact(path)


def register_model(model_uri: str, name: str):
    return mlflow.register_model(model_uri, name)
