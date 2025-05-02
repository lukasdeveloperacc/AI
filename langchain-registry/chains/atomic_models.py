from pydantic import BaseModel, Field

from typing import TypeVar
from chains.preprocess_chains import DocumentPreprocessOutput

T = TypeVar("T", bound=BaseModel)


class MapChainInput(DocumentPreprocessOutput):
    pass


class MapChainOutput(BaseModel):
    doc_summaries: list[str] = Field(..., description="")


class ReduceChainInput(MapChainOutput):
    language: str = Field(default="korea", description="")

    @classmethod
    def from_output(cls, output: T):
        return cls(**output.model_dump())


class ReduceChainOutput(BaseModel):
    summary: str
