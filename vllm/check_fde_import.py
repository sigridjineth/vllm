# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys

# Add parent directory to path
sys.path.append("/Users/sigridjineth/Desktop/work/vllm")

from unittest.mock import MagicMock

sys.modules["psutil"] = MagicMock()
sys.modules["zmq"] = MagicMock()
sys.modules["zmq.asyncio"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["transformers.configuration_utils"] = MagicMock()
sys.modules["transformers.models"] = MagicMock()
sys.modules["transformers.models.auto"] = MagicMock()
sys.modules["transformers.models.mllama"] = MagicMock()
sys.modules["transformers.models.auto.image_processing_auto"] = MagicMock()
sys.modules["transformers.models.auto.modeling_auto"] = MagicMock()
sys.modules["transformers.models.roberta"] = MagicMock()
sys.modules["transformers.processing_utils"] = MagicMock()
sys.modules["transformers.utils"] = MagicMock()
sys.modules["transformers.utils.chat_template_utils"] = MagicMock()
sys.modules["transformers.modeling_rope_utils"] = MagicMock()
sys.modules["cbor2"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["huggingface_hub.utils"] = MagicMock()
sys.modules["cloudpickle"] = MagicMock()
sys.modules["msgspec"] = MagicMock()
sys.modules["blake3"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["cpuinfo"] = MagicMock()

class MockUploadFile:
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema
        return core_schema.any_schema()
sys.modules["fastapi"].UploadFile = MockUploadFile

# Mock msgspec.Struct to be a simple class
class MockStruct:
    def __init__(self, **kwargs):
        pass
    def __init_subclass__(cls, **kwargs):
        pass
sys.modules["msgspec"].Struct = MockStruct

# Mock transformers.PretrainedConfig
class MockPretrainedConfig:
    def __init__(self, **kwargs):
        pass
sys.modules["transformers"].PretrainedConfig = MockPretrainedConfig
sys.modules["transformers.configuration_utils"].PretrainedConfig = MockPretrainedConfig

try:
    print("Importing FDEPooler...")
    from vllm.model_executor.layers.fde import FDEPooler
    print(f"FDEPooler type: {type(FDEPooler)}")
    print(f"FDEPooler is class: {isinstance(FDEPooler, type)}")
    
    print("Importing BgeM3FDE...")
    from vllm.model_executor.models.bge_m3_fde import BgeM3FDE
    print("Imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
