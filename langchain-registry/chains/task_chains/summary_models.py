from pydantic import Field

from chains.preprocess_models import DocumentPreprocessInput
from chains.composed_models import MapReduceChainOutput


class SummaryChainInput(DocumentPreprocessInput):
    language: str | None = Field(default="korean", description="")


class SummaryChainOutput(MapReduceChainOutput):
    pass
