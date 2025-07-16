from pydantic import BaseModel
from common.datasets import MVTecDataset


class BaseDatasetConfig(BaseModel):
    pass


class MVTecDatasetConfig(BaseDatasetConfig):
    root_dir: str = "datasets/mvtec"
    data_extensions: list[str] = ["jpg", "png"]
    resize: int = 256


class AnotherDatasetConfig(BaseModel):
    param1: int = 1
    param2: str = "aa"
