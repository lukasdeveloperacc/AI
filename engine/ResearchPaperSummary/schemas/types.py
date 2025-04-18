from langserve import CustomUserType
from pydantic import Field


class FileProcessingRequest(CustomUserType):
    base64_file: str = Field(..., extra={"widget": {"type": "base64file"}})
    language: str = Field("korean", description="Output language")


class ChatReuqest(CustomUserType):
    question: str = Field(..., description="Question to ask the model")
