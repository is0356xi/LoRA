#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download and check Dolly-15k dataset (MLPerf準拠).
- HuggingFace Hubからデータを取得
- データ件数、カラム、欠損値をチェック
- サンプルを数件表示
"""

import os
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/raw_dolly")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Downloading databricks-dolly-15k ===")
    dataset = load_dataset("databricks/databricks-dolly-15k")

    print("=== Dataset Info ===")
    print(dataset)

    # 欠損値チェック
    print("=== Null Value Check ===")
    for col in dataset["train"].column_names:
        null_count = sum(x is None or x == "" for x in dataset["train"][col])
        print(f"{col}: {null_count} null/empty")

    # サンプル表示
    print("=== Sample Data ===")
    for i in range(3):
        sample = dataset["train"][i]
        print(f"[{i}] instruction: {sample['instruction']}")
        print(f"    context: {sample['context']}")
        print(f"    response: {sample['response']}")
        print("-" * 50)

    # 保存
    print("=== Saving raw dataset ===")
    dataset.save_to_disk(args.output_dir)
    print(f"Done. Saved at {args.output_dir}")


if __name__ == "__main__":
    main()
