import argparse
import math

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from peft import LoraConfig, TaskType, get_peft_model

import config


def make_dataset(path, tokenizer):
    with open(path, encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    ids = []
    for line in lines:
        ids += tokenizer.encode(line, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
    chunks = [ids[i:i+config.BLOCK_SIZE] for i in range(0, len(ids) - config.BLOCK_SIZE + 1, config.BLOCK_SIZE)]
    return Dataset.from_dict({'input_ids': chunks})


def add_syntax_tokens(tokenizer, model):
    n = tokenizer.add_special_tokens({'additional_special_tokens': config.SYNTACTIC_SPECIAL_TOKENS})
    if n > 0:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb = model.get_input_embeddings()
            emb.weight[-n:] = emb.weight[:-n].mean(dim=0)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['raw', 'tagged'], required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    args = parser.parse_args()

    tagged = args.mode == 'tagged'
    if tagged:
        train_file = config.TAGGED_TRAIN_FILE
        val_file = config.TAGGED_VAL_FILE
        output_dir = args.output_dir or config.TAGGED_OUTPUT_DIR
    else:
        train_file = config.RAW_TRAIN_FILE
        val_file = config.RAW_VAL_FILE
        output_dir = args.output_dir or config.RAW_OUTPUT_DIR

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if tagged:
        tokenizer, model = add_syntax_tokens(tokenizer, model)

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias='none',
    ))
    model.print_trainable_parameters()

    train_ds = make_dataset(train_file, tokenizer)
    val_ds = make_dataset(val_file, tokenizer)
    print(f'Train: {len(train_ds)} blocks, Val: {len(val_ds)} blocks')

    targs = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_steps=50,
        report_to='none',
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    ppl = math.exp(trainer.evaluate()['eval_loss'])
    print(f'Val perplexity: {ppl:.2f}')
    print(f'Saved to {output_dir}')


if __name__ == '__main__':
    main()
