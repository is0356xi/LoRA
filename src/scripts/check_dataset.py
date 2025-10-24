#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check preprocessed Dolly dataset.
- サンプルの input_ids / labels / attention_mask を確認
- prompt 部分が -100 になっているか確認
- attention_mask の有効長とパディング確認
"""

import argparse
import os
from datasets import load_from_disk
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output_dir", type=str, default="/ssd/dolly15k/processed")
    args = parser.parse_args()

    dataset = load_from_disk(args.output_dir)
    train_data = dataset["train"]

    print("=== Dataset loaded ===")
    print(train_data)

    # 3個目のサンプルを確認
    sample = train_data[2]

    input_ids = sample["input_ids"]
    labels = sample["labels"]
    attention_mask = sample["attention_mask"]

    masked_count = sum(1 for x in labels if x == -100)
    total = len(labels)
    active_tokens = sum(attention_mask)  # 1 の数 = 実際の有効トークン数

    print("\n=== Third train example (tokenized) ===")
    print("Input IDs:", input_ids[:30])
    print("Labels   :", labels[:30])
    print("Attention mask:", attention_mask[:30])
    print(f"Masked count (-100): {masked_count} / {total}")
    print(f"Active tokens (sum of attention_mask): {active_tokens}")
    print(f"Sequence length (len): {len(attention_mask)}")

    # decodeして確認（前半は入力、後半は応答）
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, token=token
    )
    decoded_input = tokenizer.decode(
        [i for i in input_ids if i != tokenizer.pad_token_id]
    )
    print("\nDecoded text:\n", decoded_input)


if __name__ == "__main__":
    main()
