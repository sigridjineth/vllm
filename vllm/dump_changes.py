import os

files_to_dump = [
    "vllm/check_fde_import.py", "vllm/convert_to_fde_config.py",
    "vllm/model_executor/layers/fde.py",
    "vllm/model_executor/models/bge_m3_fde.py",
    "vllm/model_executor/models/registry.py",
    "vllm/model_executor/models/test_fde.py", "vllm/pooling_params.py",
    "vllm/rag_middleware.py", "vllm/standalone_test_fde.py",
    "vllm/entrypoints/openai/protocol.py", "vllm/test_fde_logic_standalone.py",
    "vllm/Dockerfile.fde"
]

output_file = "fde_branch_dump.txt"
base_dir = "/Users/sigridjineth/Desktop/work/vllm"

with open(output_file, "w") as outfile:
    for rel_path in files_to_dump:
        abs_path = os.path.join(base_dir, rel_path)
        if os.path.exists(abs_path):
            outfile.write(f"--- START OF FILE: {rel_path} ---\n")
            try:
                with open(abs_path) as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"Error reading file: {e}\n")
            outfile.write(f"\n--- END OF FILE: {rel_path} ---\n\n")
        else:
            outfile.write(f"--- FILE NOT FOUND: {rel_path} ---\n\n")

print(f"Dumped {len(files_to_dump)} files to {output_file}")
