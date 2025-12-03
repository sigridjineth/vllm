import sys
from unittest.mock import MagicMock

sys.path.append("/Users/sigridjineth/Desktop/work/vllm")

# Mock dependencies
sys.modules["transformers"] = MagicMock()
sys.modules["transformers.configuration_utils"] = MagicMock()
sys.modules["transformers.models"] = MagicMock()
sys.modules["transformers.models.auto"] = MagicMock()
sys.modules["transformers.models.auto.image_processing_auto"] = MagicMock()
sys.modules["transformers.models.auto.modeling_auto"] = MagicMock()
sys.modules["transformers.models.roberta"] = MagicMock()
sys.modules["transformers.processing_utils"] = MagicMock()
sys.modules["transformers.utils"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["zmq"] = MagicMock()
sys.modules["zmq.asyncio"] = MagicMock()
sys.modules["cbor2"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["huggingface_hub.utils"] = MagicMock()
sys.modules["msgspec"] = MagicMock()
sys.modules["cloudpickle"] = MagicMock()
sys.modules["blake3"] = MagicMock()


# Mock msgspec.Struct to be a simple class so PoolingParams works
class MockStruct:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __init_subclass__(cls, **kwargs):
        pass


sys.modules["msgspec"].Struct = MockStruct

from vllm.entrypoints.openai.protocol import EmbeddingCompletionRequest

# Pass is_document as an extra field
req = EmbeddingCompletionRequest(model="test", input="test", is_document=True)

params = req.to_pooling_params()
# Check if is_document is in the request object (as extra field)
print(f"is_document in request keys: {'is_document' in req.model_dump()}")
print(f"is_document in params: {params.is_document}")
