from pydantic import Field
from langserve import CustomUserType


class SummaryChainInput(CustomUserType):
    base64_file: str = Field(..., description="", extra={"widget": {"type": "base64file"}})
    language: str = Field(default="korean", description="")
