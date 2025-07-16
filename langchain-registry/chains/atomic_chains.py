from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langsmith import Client

from chains.atomic_models import MapChainInput, MapChainOutput, ReduceChainInput, ReduceChainOutput

from dotenv import load_dotenv

import os, logging

load_dotenv()


class MapChain(Runnable[MapChainInput, MapChainOutput]):
    def __init__(self, prompt_key: str, llm: BaseLanguageModel, output_paresr: BaseOutputParser):
        super().__init__()

        client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self._prompt: PromptTemplate = client.pull_prompt(prompt_key)
        self._prompt.pretty_print()
        self._llm = llm
        self._output_parser = output_paresr

    def invoke(self, _input: MapChainInput, config: RunnableConfig | None = None) -> MapChainOutput:
        chain = self._prompt | self._llm | self._output_parser
        doc_summaries = chain.batch(_input.docs, config)
        logging.info(f"docs summaies : {type(doc_summaries)}, {len(doc_summaries)}")
        logging.info(f"docs element type  : {type(doc_summaries[0])}")

        return MapChainOutput(doc_summaries=doc_summaries)


class ReduceChain(Runnable[ReduceChainInput, ReduceChainOutput]):
    def __init__(self, prompt_key: str, llm: BaseLanguageModel, output_paresr: BaseOutputParser):
        super().__init__()

        client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        self._prompt: PromptTemplate = client.pull_prompt(prompt_key)
        self._prompt.pretty_print()
        self._llm = llm
        self._output_parser = output_paresr
        self._chain = self._prompt | self._llm | self._output_parser

    def invoke(self, _input: ReduceChainInput, config: RunnableConfig | None = None) -> ReduceChainOutput:
        logging.info(f"Config : {config} , {type(config)}")
        if config.get("configurable"):
            _input.language = config.get("configurable")["language"]

        summary = self._chain.invoke(_input.model_dump(), config)

        return ReduceChainOutput(summary=summary)
