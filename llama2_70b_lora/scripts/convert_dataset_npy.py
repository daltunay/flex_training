import argparse
import numpy as np
import pandas as pd


def convert(data_dir, split):
    df = pd.read_parquet(f"{data_dir}/{split}-00000-of-00001.parquet")
    transformed_data = df.apply(
        lambda row: {
            "input_ids": row["input_ids"],
            "loss_mask": [int(x != -100) for x in row["labels"]],
            "seq_start_id": [0],
        },
        axis=1,
    ).tolist()
    np.save(f"{data_dir}/{split}", transformed_data)


parser = argparse.ArgumentParser(description="Convert dataset script")
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory of the data files",
)

args = parser.parse_args()

convert(args.data_dir, "train")
convert(args.data_dir, "validation")
