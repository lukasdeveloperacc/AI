from pydantic import BaseModel
from common.base import BaseDataset, BaseExperimentLogger
from common.loggers import PythonLogger
from common.utils import ConfigUtil

import torch


class UnetTrainerConfig(BaseModel):
    dataset: BaseDataset | list[BaseDataset] | None = None
    network: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    loss_fn: torch.nn.Module | None = None
    scheduler: torch.nn.Module | None = None
    logger: BaseExperimentLogger | None = None
    device: str = "cuda:0"
    batch_size: int = 32
    epochs: int = 1
    including_test: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    pretrained_checkpoint_path: str = ""
    is_export_onnx: bool = False
    is_export_tensorrt: bool = False

    class Config:
        arbitrary_types_allowed = True


class TrainConfig(UnetTrainerConfig):
    device: str = "cuda:0"
    epochs: int = 1
    including_test: bool = False
    checkpoint_dir: str = "outputs/checkpoints"
    pretrained_checkpoint_path: str = ""


class TestConfig(UnetTrainerConfig):
    device: str = "cuda:0"
    checkpoint_dir: str = "outputs/checkpoints"
    pretrained_checkpoint_path: str = "outputs/checkpoints/epoch_1.pt"
    is_export_tensorrt: bool = False

ConfigUtil.create_from_config_to_yaml_when_importing()
