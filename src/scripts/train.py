import argparse
import os
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig
import torch
from tqdm import tqdm


def main():
    # --- 1. 設定: 実験のパラメータを定義 ---
    parser = argparse.ArgumentParser(description="LoRA Finetuning")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="pretrained model name or path",
    )
    parser.add_argument("--dataset_dir", type=str, default="./data/processed_dolly")
    parser.add_argument("--output_dir", type=str, default="./models/lora_finetuned")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Adjust based on your GPU memory",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="epochs for training"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank which controls the adaptation capacity",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha which scales the LoRA updates",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout which helps regularization",
    )
    # --- テスト用の追加引数 ---
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50,
        help="Maximum number of training steps for quick testing",
    )
    args = parser.parse_args()

    # acceleratorの初期化
    accelerator = Accelerator()

    # データセットの読み込み
    processed_dataset = load_from_disk(args.dataset_dir)
    print("Dataset loaded:", processed_dataset)

    # trainデータセット
    train_dataset = processed_dataset["train"]

    # トークナイザの読み込み
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, token=token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # モデルの読み込み
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=token,
        dtype=torch.bfloat16,
    )

    # LoRAの設定
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # modelにLoRAを適用
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # データコラレータの定義
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, pad_to_multiple_of=8, label_pad_token_id=-100
    )

    # データローダの定義
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )

    # オプティマイザの定義
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # ステップ数の計算
    num_training_steps = (
        args.max_train_steps
        if args.max_train_steps > 0
        else args.num_train_epochs * len(train_dataloader)
    )

    # 学習率スケジューラの定義
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Acceleratorで環境をラッピング
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    ### 学習ループ ###
    progress_bar = tqdm(
        range(num_training_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    # モデルをtrainモードに設定
    model.train()

    for epoch in range(args.num_train_epochs):
        for batch in train_dataloader:
            output = model(**batch)
            loss = output.loss

            # lossをprogress_barに表示
            progress_bar.set_description(f"loss: {loss.item():.4f}")

            # lossの逆伝搬
            accelerator.backward(loss)

            # パラメータの更新
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            completed_steps += 1

            if args.max_train_steps > 0 and completed_steps >= num_training_steps:
                break

        if args.max_train_steps > 0 and completed_steps >= num_training_steps:
            break


if __name__ == "__main__":
    main()
