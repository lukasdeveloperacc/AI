from common.base import BaseExperimentLogger

import torch
import mlflow.pytorch
import logging
import mlflow


class PythonLogger(BaseExperimentLogger):
    def __init__(self, name="experiment", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)

    def log_param(self, key, value):
        self.logger.info(f"[param] {key} = {value}")

    def log_metric(self, key, value, step=None):
        step_info = f" @ step {step}" if step is not None else ""
        self.logger.info(f"[metric] {key} = {value}{step_info}")

    def log_artifact(self, path):
        self.logger.info(f"[artifact] saved: {path}")

    def log_model(self, model, is_state_dict: bool = False):
        self.logger.info(f"[artifact] saved model: {type(model)}")

    def end(self):
        self.logger.info("Experiment finished")


class MLflowLogger(BaseExperimentLogger):
    def __init__(self, run_name="default", tracking_uri: str = "http://localhost:5000", **params):
        try:
            if mlflow.active_run():
                logging.info(f"[{self.__class__.__name__}] Active run : {mlflow.active_run()}")
                self.end()
                logging.info(f"[{self.__class__.__name__}] Finish actived run : {mlflow.active_run()}")

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name=run_name)
            mlflow.start_run(run_name=run_name)
            logging.info(f"[{self.__class__.__name__}] Start run")
            mlflow.log_params(params)

        except Exception as e:
            logging.error(f"[{self.__class__.__name__}] {e}")
            self.end()

    def log_param(self, key, value):
        try:
            mlflow.log_param(key, value)

        except Exception as e:
            logging.error(e)
            self.end()

    def log_metric(self, key, value, step=None):
        try:
            mlflow.log_metric(key, value, step=step)

        except Exception as e:
            logging.error(e)
            self.end()

    def log_artifact(self, path):
        try:
            mlflow.log_artifact(path)
            logging.info(f"[{self.__class__.__name__}] log artifact : {path}")

        except Exception as e:
            logging.error(e)
            self.end()

    def log_model(self, model: torch.nn.Module, is_state_dict: bool = False):
        try:
            if is_state_dict:
                mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="state_dict")
            else:
                mlflow.pytorch.log_model(model, artifact_path="model")

            logging.info(f"[{self.__class__.__name__}] Log model, Is state dict : {is_state_dict}")

        except Exception as e:
            logging.error(e)
            self.end()

    def end(self):
        try:
            mlflow.end_run()

        except Exception as e:
            logging.error(e)
