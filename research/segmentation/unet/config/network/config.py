from pydantic import BaseModel
from common.utils import ConfigUtil

class UnetConfig(BaseModel):
    in_channels: int = 3
    conv_channels: list[int] = [64, 128, 256, 512]

ConfigUtil.create_from_config_to_yaml_when_importing()
