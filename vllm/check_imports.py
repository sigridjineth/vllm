import sys
import importlib
from pathlib import Path

# Add vllm root to path if needed (simulating running from root)
sys.path.append("/Users/sigridjineth/Desktop/work/vllm")

try:
    import vllm.model_executor.models.registry as registry
    print("Successfully imported registry")
    
    # Check if we can import 'llama' directly
    try:
        import llama
        print("Successfully imported llama directly")
    except ImportError:
        print("Failed to import llama directly")
        
    # Check if we can import 'vllm.model_executor.models.llama'
    try:
        import vllm.model_executor.models.llama
        print("Successfully imported vllm.model_executor.models.llama")
    except ImportError:
        print("Failed to import vllm.model_executor.models.llama")

    # Check registry content
    print(f"Registry keys sample: {list(registry._VLLM_MODELS.keys())[:5]}")
    
    # Check what happens if we use the registry to load
    # We need to instantiate _LazyRegisteredModel manually since it's internal
    from vllm.model_executor.models.registry import _LazyRegisteredModel
    
    model = _LazyRegisteredModel("llama", "LlamaForCausalLM")
    try:
        cls = model.load_model_cls()
        print(f"Successfully loaded class: {cls}")
    except Exception as e:
        print(f"Failed to load class via registry: {e}")
        
    # Try with full path
    model_full = _LazyRegisteredModel("vllm.model_executor.models.llama", "LlamaForCausalLM")
    try:
        cls = model_full.load_model_cls()
        print(f"Successfully loaded class with full path: {cls}")
    except Exception as e:
        print(f"Failed to load class via registry with full path: {e}")

except Exception as e:
    print(f"General error: {e}")
