"""Dataset types for CoreRec.

This file was empty, so `corerec.data` advertised itself in ``corerec.__all__``
and then exported nothing -- `from corerec import data; data.RecommendationDataset`
raised AttributeError even though the class was sitting in data/datasets.py.

Imports are guarded because the streaming and multimodal readers pull optional
dependencies; a missing extra leaves that name unavailable rather than breaking
`import corerec.data` for everyone.
"""

__all__ = []


def _export(module_name, *names):
    import importlib

    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ImportError:  # optional dependency missing
        return
    for name in names:
        obj = getattr(module, name, None)
        if obj is not None:
            globals()[name] = obj
            __all__.append(name)


_export("data", "BaseDataset", "ContextualDataset", "GraphDataset")
_export("datasets", "RecommendationDataset", "SequentialRecommendationDataset", "DSSMDataset")
_export("multimodal_dataset", "MultimodalRecommendationDataset", "MultimodalSequentialDataset")
_export("streaming_dataloader", "StreamingDataset", "StreamingDataLoader")
_export("see", "SeeReader")
