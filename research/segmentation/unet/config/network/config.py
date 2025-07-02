from pydantic import BaseModel


class UnetConfig(BaseModel):
    in_channels: int = 3
    conv_channels: list[int] = [64, 128, 256, 512]
