from chains.atomic_models import MapChainInput, ReduceChainInput, ReduceChainOutput


class MapReduceChainInput(MapChainInput, ReduceChainInput):
    pass


class MapReduceChainOutput(ReduceChainOutput):
    pass
