import argparse
import asyncio
import json
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.pooling_params import PoolingParams

# --- Mock ANN Index (Replace with Faiss/DiskANN) ---
class MockANNIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vector: np.ndarray, meta: Dict[str, Any]):
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dim {vector.shape[0]} != index dim {self.dim}")
        self.vectors.append(vector)
        self.metadata.append(meta)

    def search(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if not self.vectors:
            return []
        
        # Simple cosine similarity
        # query: (d,)
        # db: (N, d)
        db = np.stack(self.vectors)
        
        # Normalize
        q_norm = query / (np.linalg.norm(query) + 1e-9)
        db_norm = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)
        
        scores = np.dot(db_norm, q_norm)
        
        # Top-k
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "score": float(scores[idx]),
                "metadata": self.metadata[idx]
            })
        return results

# --- FastAPI App ---
app = FastAPI(title="vLLM FDE RAG Middleware")

class AppState:
    engine: Optional[AsyncLLMEngine] = None
    ann_index: Optional[MockANNIndex] = None

state = AppState()

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class IndexRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    # Initialize vLLM Engine
    # Note: In production, pass args via command line or config
    parser = argparse.ArgumentParser()
    AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    
    # Force embedding mode if not set (though model type should handle it)
    # args.model = "BAAI/bge-m3" # Example, user should pass this
    
    engine_args = AsyncEngineArgs.from_cli_args(args)
    state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Initialize ANN Index (dim depends on FDE config, e.g., 1024)
    # We might need to get this from model config after load, 
    # but for now assume we know it or lazy init.
    # Let's lazy init on first embedding if needed, or hardcode for demo.
    state.ann_index = MockANNIndex(dim=1024) # Default BGE-M3 FDE dim?
    
    print("vLLM Engine and ANN Index initialized.")

async def get_embedding(text: str) -> np.ndarray:
    """Generate FDE embedding using vLLM."""
    if state.engine is None:
        raise RuntimeError("Engine not initialized")

    # Create a unique request ID
    request_id = f"req_{hash(text)}"
    
    # Pooling params
    pooling_params = PoolingParams()
    
    # Generate
    # vLLM's encode() returns an AsyncGenerator
    results_generator = state.engine.encode(
        request_id=request_id,
        prompt=text,
        pooling_params=pooling_params
    )
    
    # Get final result
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
        
    # Extract vector
    # PoolerOutput is typically a list of floats or list of list of floats
    # FDE returns a single vector per prompt
    if final_output.outputs:
        # data is PoolingOutput
        # embedding is in outputs[0].embedding
        embedding = final_output.outputs[0].embedding
        return np.array(embedding, dtype=np.float32)
    else:
        raise RuntimeError("No embedding returned")

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """
    1. Embed query using vLLM FDE.
    2. Search ANN index.
    3. Return ranked results.
    """
    try:
        # 1. Embed
        embedding = await get_embedding(req.query)
        
        # 2. Search
        # Check dim
        if state.ann_index.dim != embedding.shape[0]:
             # Re-init index if dim mismatch (lazy fix for demo)
             print(f"Resizing index from {state.ann_index.dim} to {embedding.shape[0]}")
             state.ann_index = MockANNIndex(dim=embedding.shape[0])

        results = state.ann_index.search(embedding, k=req.k)
        
        return {"query": req.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_endpoint(req: IndexRequest):
    """
    1. Embed text using vLLM FDE.
    2. Add to ANN index.
    """
    try:
        embedding = await get_embedding(req.text)
        
        # Check dim
        if state.ann_index.dim != embedding.shape[0]:
             print(f"Resizing index from {state.ann_index.dim} to {embedding.shape[0]}")
             state.ann_index = MockANNIndex(dim=embedding.shape[0])

        idx = len(state.ann_index.vectors)
        state.ann_index.add(embedding, req.metadata)
        
        return {"id": idx, "status": "indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_ready": state.engine is not None,
        "index_size": len(state.ann_index.vectors) if state.ann_index else 0,
    }


if __name__ == "__main__":
    import uvicorn
    # Example usage:
    # python rag_middleware.py --model /path/to/bge-m3-fde --trust-remote-code
    uvicorn.run(app, host="0.0.0.0", port=8000)
