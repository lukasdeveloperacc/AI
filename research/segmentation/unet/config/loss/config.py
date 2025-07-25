from pydantic import BaseModel
from common.utils import ConfigUtil


class CrossEntropyLossConfig(BaseModel):
    pass

ConfigUtil.create_from_config_to_yaml_when_importing()
