# from langchain.schema import Document
from pydantic import BaseModel, Field


class DocumentPreprocessInput(BaseModel):
    base64_file: str | None = Field(default=None, description="")
    pdf_path: str | None = Field(default=None, description="")


class DocumentPreprocessOutput(BaseModel):
    docs: list | None = Field(default=None, description="")

    class Config:
        arbitrary_types_allowed = True
