import argparse
import os
import pickle
from typing import List

import torch


DENSE_FEATURE_COLS = [
    "O_TEMP",
    "D_TEMP",
    "O_PRCP",
    "D_PRCP",
    "O_WSPD",
    "D_WSPD",
    "FLIGHTS",
]

SPARSE_FEATURE_COLS = [
    "MONTH",
    "DAY_OF_WEEK",
    "CRS_ARR_TIME_HOUR",
    "CRS_DEP_TIME_HOUR",
    "ORIGIN_INDEX",
    "DEST_INDEX",
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
]


def load_sparse_tensor(path: str) -> torch.Tensor:
    dataset = torch.load(path)
    if not hasattr(dataset, "tensors") or len(dataset.tensors) < 2:
        raise ValueError(f"Unexpected dataset format: {path}")
    return dataset.tensors[1]


def compute_feat_nums(paths: List[str]) -> List[int]:
    sparse_all = []
    for path in paths:
        sparse = load_sparse_tensor(path)
        sparse_all.append(sparse.reshape(-1, sparse.shape[-1]))
    sparse_all = torch.cat(sparse_all, dim=0)
    min_vals = sparse_all.min(dim=0).values
    if (min_vals < 0).any():
        raise ValueError("Sparse features contain negative values; check your chain data.")
    max_vals = sparse_all.max(dim=0).values
    return (max_vals + 1).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate new_feature_columns.pkl for chain models.")
    parser.add_argument("--data-dir", default="data/Aeolus/Flight_chain/chain_data_2024")
    parser.add_argument("--year", default="2024")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--valid-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--embed-dim", type=int, default=7)
    parser.add_argument("--output", default="new_feature_columns.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_path and args.valid_path and args.test_path:
        paths = [args.train_path, args.valid_path, args.test_path]
    else:
        paths = [
            os.path.join(args.data_dir, f"flight_chain_train_{args.year}.pt"),
            os.path.join(args.data_dir, f"flight_chain_val_{args.year}.pt"),
            os.path.join(args.data_dir, f"flight_chain_test_{args.year}.pt"),
        ]
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")

    feat_nums = compute_feat_nums(paths)
    if len(feat_nums) != len(SPARSE_FEATURE_COLS):
        raise ValueError("Sparse feature count mismatch; check SPARSE_FEATURE_COLS.")

    sparse_feature_cols = [
        {"feat_name": name, "feat_num": int(num), "embed_dim": int(args.embed_dim)}
        for name, num in zip(SPARSE_FEATURE_COLS, feat_nums)
    ]
    feature_columns = (DENSE_FEATURE_COLS, sparse_feature_cols)

    with open(args.output, "wb") as f:
        pickle.dump(feature_columns, f)

    print(f"Saved: {args.output}")
    print(feature_columns)


if __name__ == "__main__":
    main()
