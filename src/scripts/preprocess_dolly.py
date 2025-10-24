#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dolly-15k dataset preprocessing script (MLPerf準拠).
- instruction, context, response を忠実に利用
- train/test に分割
- tokenizerで符号化し、入力部分を -100 にして損失計算から除外
"""

import os
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer


def preprocess_function(example, tokenizer, max_length=2048):
    # 入力: instruction と context を結合
    if example["context"]:
        prompt = example["instruction"].strip() + "\n" + example["context"].strip()
    else:
        prompt = example["instruction"].strip()

    response = example["response"].strip()

    # 入力と出力をまとめる
    full_input = prompt + "\n" + response

    # tokenizer処理（入力＋出力）
    tokenized = tokenizer(full_input, truncation=True, max_length=max_length)

    # prompt部分だけを再度tokenizeして長さを取得
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    prompt_len = len(prompt_ids)

    labels = tokenized["input_ids"].copy()
    # prompt部分を -100 にして損失計算から除外
    labels[:prompt_len] = [-100] * prompt_len

    tokenized["labels"] = labels
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--input_dir", type=str, default="./data/raw_dolly")
    parser.add_argument("--output_dir", type=str, default="./data/processed_dolly")
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Loading raw dataset ===")
    dataset = load_from_disk(args.input_dir)

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, token=token
    )

    print("=== Splitting train/test ===")
    dataset = dataset["train"].train_test_split(test_size=args.test_size, seed=42)

    print("=== Tokenizing ===")
    processed = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=False,
        remove_columns=dataset["train"].column_names,
    )

    print("=== Saving to disk ===")
    processed.save_to_disk(args.output_dir)
    print(f"Done. Saved at {args.output_dir}")


if __name__ == "__main__":
    main()
