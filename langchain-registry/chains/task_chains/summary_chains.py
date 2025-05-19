from chains.preprocess_chains import DocumentPreprocessor
from chains.composed_chains import MapReduceChain
from chains.constants import SummaryMode
from chains.task_chains.summary_models import SummaryChainInput

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from typing import Any, Iterator
from dotenv import load_dotenv

import os

load_dotenv()


class ResearchPaperSummaryChain(Runnable):
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
        _config = RunnableConfig(**config) if config is not None else RunnableConfig()
        _config.update({"configurable": {"language": _input.language}})
        output = self._chain.invoke(_input, config=_config)

        return output.summary

    def stream(self, _input: SummaryChainInput, config: RunnableConfig | None = None) -> Iterator[str]:
        try:
            _config = RunnableConfig(**config) if config is not None else RunnableConfig()
            _config.update({"configurable": {"language": _input.language}})
            for chunk in self._chain.stream(_input, config=config):
                yield chunk.summary

        except Exception as e:
            raise Exception(f"Fail to stream chain : {e}")

    def batch(self, _input: SummaryChainInput, config: RunnableConfig | None = None) -> list[str]:
        _config = RunnableConfig(**config) if config is not None else RunnableConfig()
        _config.update({"configurable": {"language": _input.language}})
        outputs = self._chain.batch(_input, config=_config)

        return [output.summary for output in outputs]


if __name__ == "__main__":
    import logging, base64

    logging.basicConfig(level=logging.INFO)

    chain = ResearchPaperSummaryChain(preprocess_args={"chunk_size": 1000, "chunk_overlap": 50})
    with open("chains/1706.03762v7.pdf", "rb") as f:
        b = f.read()
        b = base64.b64encode(b)
        b = b.decode("utf-8")
        _input = SummaryChainInput(base64_file=b)

        result = chain.invoke(
            _input,
            config=RunnableConfig(configurable={"language": _input.language}),
        )

        logging.info(f"{result}")
