from pydantic import BaseModel


class AdamConfig(BaseModel):
    lr: float = 1e-3
