from pydantic import BaseModel, Field

from typing import TypeVar

T = TypeVar("T", bound=BaseModel)


class DocumentPreprocessInput(BaseModel):
    base64_file: str | None = Field(default=None, description="")
    pdf_path: str | None = Field(default=None, description="")

    @classmethod
    def from_output(cls, output: T):
        return cls(**output.model_dump())


class DocumentPreprocessOutput(BaseModel):
    docs: list | None = Field(default=None, description="")
