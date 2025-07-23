from pydantic import BaseModel
from common.utils import ConfigUtil

class BaseDatasetConfig(BaseModel):
    pass


class MVTecDatasetConfig(BaseDatasetConfig):
    root_dir: str = "datasets/mvtec"
    data_extensions: list[str] = ["jpg", "png"]
    resize: int = 256


class AnotherDatasetConfig(BaseDatasetConfig):
    param1: int = 1
    param2: str = "aa"

class ComposeDatasetConfig(BaseDatasetConfig):
    MVTecDataset: MVTecDatasetConfig = MVTecDatasetConfig()
    AnotherDataset: AnotherDatasetConfig = AnotherDatasetConfig()

if __name__ == "__main__":
    ConfigUtil.create_from_config_to_yaml(ComposeDatasetConfig())
