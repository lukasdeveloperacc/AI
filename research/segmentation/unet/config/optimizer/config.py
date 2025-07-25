from pydantic import BaseModel
from common.utils import ConfigUtil


class AdamConfig(BaseModel):
    lr: float = 1e-3

ConfigUtil.create_from_config_to_yaml_when_importing()
