from segmentation.unet.config.dataset.config import BaseDatasetConfig, MVTecDatasetConfig, AnotherDatasetConfig
from segmentation.unet.config.network.config import UnetConfig
from segmentation.unet.config.optimizer.config import AdamConfig
from segmentation.unet.config.loss.config import CrossEntropyLossConfig
from segmentation.unet.config.trainer.config import TrainConfig as trainerTrainConfig
from segmentation.unet.config.trainer.config import TestConfig as trainerTestConfig

from typing import Optional
from pydantic import BaseModel


class TrainConfig(BaseModel):
    mode: str = "train"
    dataset: dict[str, BaseDatasetConfig | MVTecDatasetConfig | AnotherDatasetConfig] | MVTecDatasetConfig = (
        MVTecDatasetConfig()
    )
    network: dict[str, UnetConfig] | UnetConfig = UnetConfig()
    optimizer: dict[str, AdamConfig] | AdamConfig = AdamConfig()
    loss: dict[str, Optional[CrossEntropyLossConfig]] | CrossEntropyLossConfig = CrossEntropyLossConfig()
    scheduler: dict | None = None
    trainer: dict[str, trainerTrainConfig] | trainerTrainConfig = trainerTrainConfig()

    class Config:
        arbitrary_types_allowed = True


class TestConfig(BaseModel):
    mode: str = "test"
    dataset: dict[str, BaseDatasetConfig | MVTecDatasetConfig | AnotherDatasetConfig] | MVTecDatasetConfig = (
        MVTecDatasetConfig()
    )
    network: dict[str, UnetConfig] | UnetConfig = UnetConfig()
    trainer: dict[str, trainerTestConfig] | trainerTestConfig = trainerTestConfig()
    loss: dict[str, Optional[CrossEntropyLossConfig]] | CrossEntropyLossConfig = CrossEntropyLossConfig()
    is_export_onnx: bool = True
    is_export_tensorrt: bool = True

    class Config:
        arbitrary_types_allowed = True
