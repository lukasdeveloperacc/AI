from langchain.schema import Document
from pydantic import BaseModel, Field


class DocumentPreprocessInput(BaseModel):
    base64_str: str | None = Field(default=None, extra={"widget": {"type": "base64file"}})
    pdf_path: str | None = Field(default=None, description="")


class DocumentPreprocessOutput(BaseModel):
    docs: list[Document] | None = Field(default=None, description="")
