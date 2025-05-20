from chains.preprocess_chains import DocumentPreprocessor
from chains.composed_chains import MapReduceChain
from chains.constants import SummaryMode
from chains.task_chains.summary_models import SummaryChainInput

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from typing import Any, Iterator, AsyncIterable
from dotenv import load_dotenv

import os
import logging

load_dotenv()


class ResearchPaperSummaryChain(Runnable[SummaryChainInput, str]):
    def __init__(self, mode: str = SummaryMode.MAP_REDUCE.value, preprocess_args: dict[str, Any] = {}):
        super().__init__()

        self._preprocess_chain: Runnable = self._make_preprocess_chain(**preprocess_args)
        if mode == SummaryMode.MAP_REDUCE.value:
            self._sub_chain: Runnable = self._make_map_reduce_chain()

        self._chain = self._preprocess_chain | self._sub_chain

    def _make_map_reduce_chain(self) -> MapReduceChain:
        from chains.atomic_chains import MapChain, ReduceChain

        map_chain = MapChain(
            "map_prompt", ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0), StrOutputParser()
        )
        reduce_chain = ReduceChain(
            "reduce_prompt", ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0), StrOutputParser()
        )

        return MapReduceChain(map_chain, reduce_chain)

    def _make_preprocess_chain(self, chunk_size: int, chunk_overlap: int) -> DocumentPreprocessor:
        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        return DocumentPreprocessor(loader_cls=PyMuPDFLoader, loader_args={}, splitter=splitter)

    def invoke(self, _input: SummaryChainInput, config: RunnableConfig | None = None) -> str:
        logging.info(f"{__class__.__name__} Invoke")
        _config = RunnableConfig(**config) if config is not None else RunnableConfig()
        _config.update({"configurable": {"language": _input.language}})
        output = self._chain.invoke(_input, config=_config)

        return output.summary

    def stream(self, _input: SummaryChainInput, config: RunnableConfig | None = None) -> Iterator[str]:
        try:
            logging.info(f"{__class__.__name__} Stream")
            _config = RunnableConfig(**config) if config is not None else RunnableConfig()
            _config.update({"configurable": {"language": _input.language}})
            for chunk in self._chain.stream(_input, config=config):
                yield chunk.summary

        except Exception as e:
            raise Exception(f"Fail to stream chain : {e}")

    async def astream(self, _input: SummaryChainInput, config: RunnableConfig | None = None) -> AsyncIterable[str]:
        try:
            logging.info(f"{__class__.__name__} AStream")
            _config = RunnableConfig(**config) if config is not None else RunnableConfig()
            _config.update({"configurable": {"language": _input.language}})
            async for chunk in self._chain.astream(_input, config=config):
                yield chunk.summary

        except Exception as e:
            raise Exception(f"Fail to stream chain : {e}")
