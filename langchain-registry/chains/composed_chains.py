from langchain_core.runnables import Runnable, RunnableLambda, RunnableConfig

from chains.composed_models import MapReduceChainInput, MapReduceChainOutput
from chains.atomic_chains import MapChain, ReduceChain
from chains.atomic_models import ReduceChainInput

from dotenv import load_dotenv
from typing import AsyncIterator, Iterator

import logging

load_dotenv()


class MapReduceChain(Runnable[MapReduceChainInput, MapReduceChainOutput]):
    def __init__(self, map_chain: MapChain, reduce_chain: ReduceChain):
        super().__init__()

        self._map_chain = map_chain
        self._reduce_chain = reduce_chain
        self._chain = self._map_chain | RunnableLambda(ReduceChainInput.from_output) | self._reduce_chain

    def invoke(self, _input: MapReduceChainInput, config: RunnableConfig | None = None) -> MapReduceChainOutput:
        logging.info(f"{__class__.__name__} Invoke")
        output = self._chain.invoke(_input, config=config)

        return MapReduceChainOutput(summary=output.summary)

    def stream(
        self, _input: MapReduceChainInput, config: RunnableConfig | None = None
    ) -> Iterator[MapReduceChainOutput]:
        try:
            logging.info(f"{__class__.__name__} Stream")
            for chunk in self._chain.stream(_input, config=config):
                yield MapReduceChainOutput(summary=chunk.summary)

        except Exception as e:
            raise Exception(f"Fail to stream chain : {e}")

    async def astream(
        self, _input: MapReduceChainInput, config: RunnableConfig | None = None
    ) -> AsyncIterator[MapReduceChainOutput]:
        try:
            logging.info(f"{__class__.__name__} AStream")
            async for chunk in self._chain.astream(_input, config=config):
                yield MapReduceChainOutput(summary=chunk.summary)

        except Exception as e:
            raise Exception(f"Fail to stream chain : {e}")
