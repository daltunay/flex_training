# Adapted from:
# https://github.com/mlcommons/training_results_v4.0/blob/main/Oracle/benchmarks/llama2_70b_lora/implementations/BM.GPU.H100.8/scripts/download_dataset.py

import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description="Download dataset using Hugging Face Hub")
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Local directory to download the dataset to",
)

args = parser.parse_args()

snapshot_download(
    "regisss/scrolls_gov_report_preprocessed_mlperf_2",
    local_dir=args.data_dir,
    local_dir_use_symlinks=False,
    repo_type="dataset",
)
