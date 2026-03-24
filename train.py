import argparse
import os
import math

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

import config


def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def make_dataset(lines, tokenizer, block_size):
    all_ids = []
    for line in lines:
        ids = tokenizer.encode(line, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        all_ids.extend(ids)

    chunks = [
        all_ids[i : i + block_size]
        for i in range(0, len(all_ids) - block_size + 1, block_size)
    ]
    return Dataset.from_dict({"input_ids": chunks})


def add_syntax_tokens(tokenizer, model):
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": config.SYNTACTIC_SPECIAL_TOKENS}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            embed = model.get_input_embeddings()
            mean_vec = embed.weight[:-num_added].mean(dim=0)
            embed.weight[-num_added:] = mean_vec.unsqueeze(0).expand(num_added, -1)
        print(f"Added {num_added} special tokens.")
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["raw", "tagged"], required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--block_size", type=int, default=config.BLOCK_SIZE)
    args = parser.parse_args()

    if args.mode == "raw":
        train_file = config.RAW_TRAIN_FILE
        val_file = config.RAW_VAL_FILE
        output_dir = args.output_dir or config.RAW_OUTPUT_DIR
        tagged = False
    else:
        train_file = config.TAGGED_TRAIN_FILE
        val_file = config.TAGGED_VAL_FILE
        output_dir = args.output_dir or config.TAGGED_OUTPUT_DIR
        tagged = True

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if tagged:
        tokenizer, model = add_syntax_tokens(tokenizer, model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("Tokenising data...")
    train_dataset = make_dataset(load_lines(train_file), tokenizer, args.block_size)
    val_dataset = make_dataset(load_lines(val_file), tokenizer, args.block_size)
    print(f"Train blocks: {len(train_dataset)}, Val blocks: {len(val_dataset)}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=args.lr,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_results = trainer.evaluate()
    ppl = math.exp(eval_results["eval_loss"])
    print(f"Validation perplexity ({args.mode}): {ppl:.2f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
