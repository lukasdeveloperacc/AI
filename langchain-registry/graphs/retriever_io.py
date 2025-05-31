from pydantic import BaseModel


class RetrieverInput(BaseModel):
    file_path: str
