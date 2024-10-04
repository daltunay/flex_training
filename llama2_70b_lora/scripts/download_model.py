# Adapted from:
# https://github.com/mlcommons/training_results_v4.0/blob/main/Oracle/benchmarks/llama2_70b_lora/implementations/BM.GPU.H100.8/scripts/download_model.py

import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description="Download model using Hugging Face Hub")
parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Local directory to download the model to",
)

args = parser.parse_args()

snapshot_download(
    "regisss/llama2-70b-fused-qkv-mlperf",
    local_dir=args.model_dir,
    local_dir_use_symlinks=False,
)
