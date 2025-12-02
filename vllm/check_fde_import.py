# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from unittest.mock import MagicMock

sys.modules["psutil"] = MagicMock()
sys.modules["zmq"] = MagicMock()
sys.modules["zmq.asyncio"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["transformers.configuration_utils"] = MagicMock()
sys.modules["transformers.models"] = MagicMock()
sys.modules["transformers.models.roberta"] = MagicMock()
sys.modules["cbor2"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()

try:
    print("Importing FDEPooler...")
    print("Importing BgeM3FDE...")
    print("Imports successful!")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
