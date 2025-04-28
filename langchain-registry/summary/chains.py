from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langsmith import Client

from typing import Any
from dotenv import load_dotenv

import os, logging

load_dotenv()


class MapChain(Runnable):
    def __init__(self, prompt_key: str, llm: BaseLanguageModel, output_paresr: BaseOutputParser):
        super().__init__()

        client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self._prompt: PromptTemplate = client.pull_prompt(prompt_key)
        self._prompt.pretty_print()
        self._llm = llm
        self._output_parser = output_paresr

    def invoke(self, input: dict[str, Any], config: RunnableConfig | None = None):
        doc = input.get("doc")
        if doc is None:
            raise ValueError(f"doc isn't exist in the input : {input}")

        return {"doc": doc} | self._prompt | self._llm | self._output_parser

    def batch(self, inputs: dict[str, Any], config: RunnableConfig | None = None):
        docs = inputs.get("docs")

        return [self.invoke(doc, config) for doc in docs]


if __name__ == "__main__":
    from langchain_openai import OpenAI
    from langchain_core.output_parsers import StrOutputParser

    logging.basicConfig(level=logging.INFO)
    chain = MapChain("map_prompt", OpenAI(api_key=os.getenv("OPENAI_API_KEY")), StrOutputParser())
