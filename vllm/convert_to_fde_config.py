import argparse
import json
import os
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Convert BGE-M3 model config to enable FDE in vLLM")
    parser.add_argument("--model", type=str, required=True, help="Source model (HF Hub ID or local path), e.g., BAAI/bge-m3")
    parser.add_argument("--output", type=str, required=True, help="Output directory for the FDE-enabled model")
    parser.add_argument("--ksim", type=int, default=6, help="FDE ksim parameter")
    parser.add_argument("--d_proj", type=int, default=32, help="FDE d_proj parameter")
    parser.add_argument("--R_reps", type=int, default=10, help="FDE R_reps parameter")
    parser.add_argument("--fill_empty_clusters", action="store_true", help="Enable empty cluster filling (default: False)")
    
    args = parser.parse_args()
    
    print(f"Downloading/Copying model from {args.model} to {args.output}...")
    # Download everything except safetensors/bin if we want to save space? 
    # No, vLLM needs weights. We should download/copy everything.
    # Using ignore_patterns to avoid downloading huge files if user just wants config? 
    # No, user needs to serve it.
    
    try:
        snapshot_download(repo_id=args.model, local_dir=args.output, local_dir_use_symlinks=False)
    except Exception as e:
        # If it's a local path, snapshot_download might fail or we should just copy.
        if os.path.isdir(args.model):
            import shutil
            shutil.copytree(args.model, args.output, dirs_exist_ok=True)
        else:
            raise e

    config_path = os.path.join(args.output, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Modify Architecture
    print("Modifying config.json...")
    config["architectures"] = ["BgeM3FDEModel"]
    
    # Add FDE Config
    config["fde_config"] = {
        "ksim": args.ksim,
        "d_proj": args.d_proj,
        "R_reps": args.R_reps,
        "d_final": (1 << args.ksim) * args.d_proj * args.R_reps, # Auto-calc d_final
        "fill_empty_clusters": args.fill_empty_clusters,
        "seed": 42
    }
    
    # Save
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print("Generating FDE parameters...")
    try:
        import torch
        # We need to import FDEPooler and FDEConfig to generate params correctly
        # Assuming vllm is installed in editable mode or available in path
        from vllm.model_executor.layers.fde import FDEPooler, FDEConfig
        
        fde_config = FDEConfig(
            ksim=args.ksim,
            d_proj=args.d_proj,
            R_reps=args.R_reps,
            d_final=config["fde_config"]["d_final"],
            fill_empty_clusters=args.fill_empty_clusters,
            seed=42,
            use_mixed_precision=False # Save as float32 for compatibility
        )
        
        # Initialize pooler to generate params
        # We need hidden_size from config
        hidden_size = config["hidden_size"]
        pooler = FDEPooler(d=hidden_size, config=fde_config)
        
        # Extract params
        state_dict = {}
        state_dict["G"] = pooler.params.G
        if pooler.params.S is not None:
            state_dict["S"] = pooler.params.S
        if pooler.final_proj is not None:
            state_dict["W"] = pooler.final_proj.W
            
        # Save to fde_params.pt
        params_path = os.path.join(args.output, "fde_params.pt")
        torch.save(state_dict, params_path)
        print(f"FDE parameters saved to {params_path}")
        
    except ImportError:
        print("Warning: Could not import vllm to generate FDE params. You will see a warning when loading the model.")
    except Exception as e:
        print(f"Warning: Failed to generate FDE params: {e}")

    print(f"Success! FDE-enabled model saved to {args.output}")
    print(f"Run vLLM with: vllm serve {args.output} --trust-remote-code")

if __name__ == "__main__":
    main()
