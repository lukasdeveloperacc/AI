from langchain_core.runnables import Runnable, RunnableLambda, RunnableConfig

from chains.composed_models import MapReduceChainInput, MapReduceChainOutput
from chains.atomic_chains import MapChain, ReduceChain
from chains.atomic_models import ReduceChainInput

from dotenv import load_dotenv

load_dotenv()


class MapReduceChain(Runnable[MapReduceChainInput, MapReduceChainOutput]):
    def __init__(self, map_chain: MapChain, reduce_chain: ReduceChain):
        super().__init__()

        self._map_chain = map_chain
        self._reduce_chain = reduce_chain
        self._chain = self._map_chain | RunnableLambda(ReduceChainInput.from_output) | self._reduce_chain

    def invoke(self, _input: MapReduceChainInput, config: RunnableConfig | None = None) -> MapReduceChainOutput:
        output = self._chain.invoke(_input, config=config)

        return MapReduceChainOutput(summary=output.summary)
