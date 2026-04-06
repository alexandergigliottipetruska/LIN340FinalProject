import math, sys

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model

import config


def make_dataset(path, tokenizer):
    """
    Reads the text file and encodes every line into token IDs. These are then concatenated into a sequence, 
    separated by EOS (End of Sentence tokens), and split into fixed length chunks, which are used for training.
    """
    ids = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            # Tokenize the line, do not add EOS tokens automatically. 
            ids += tokenizer.encode(line.strip(), add_special_tokens=False)
            # Add EOS tokens after each sentence, so the model learns token boundaries. 
            ids.append(tokenizer.eos_token_id)

    # Slice token list into chunks of BLOCK_SIZE tokens with no overlap. Leftover tokens
    # that do not belong to a full chunk are dropped. 
    chunks = [ids[i:i+config.BLOCK_SIZE] for i in range(0, len(ids) - config.BLOCK_SIZE + 1, config.BLOCK_SIZE)]

    # HuggingFace Trainer requires a Dataset object with 'input_ids' column. 
    return Dataset.from_dict({'input_ids': chunks})


def add_syntax_tokens(tokenizer, model):
    """
    Add bracket tokens to tokenizer vocab and add
    """
    # Add the bracket tokens to the tokenizer vocabulary.
    # add_special_tokens returns the number of tokens added. 
    n = tokenizer.add_special_tokens({'additional_special_tokens': config.SYNTACTIC_SPECIAL_TOKENS})
    if n > 0:
        # Resize embedding matrix to account for new tokens.
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb = model.get_input_embeddings()
            # New token embeddings initialized as mean of all existing token embeddings.
            # The reason this is done is that it is a better starting point than just pure random noise. 
            emb.weight[-n:] = emb.weight[:-n].mean(dim=0)
    return tokenizer, model


def main():
    # Read mode from command line: either 'raw' or 'tagged'
    # Command line supplies mode either 'raw' or 'tagged' depending on the kind of model for training.
    tagged = sys.argv[1] == 'tagged'

    # Get data files and output directory corresponding to selected mode. 
    train_file = config.TAGGED_TRAIN_FILE if tagged else config.RAW_TRAIN_FILE
    val_file = config.TAGGED_VAL_FILE if tagged else config.RAW_VAL_FILE
    output_dir = config.TAGGED_OUTPUT_DIR if tagged else config.RAW_OUTPUT_DIR

    # Load pretrained Italian GPT2 tokenizer. 
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    # Has no pad token by default, so use the EOS as a substitute
    tokenizer.pad_token = tokenizer.eos_token

    # Float16 on GPU for faster training and lower memory, while CPU uses float32
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=dtype)

    # For tagged model, extend vocab with bracket tokens so model can learn sentence boundaries. 
    if tagged:
        tokenizer, model = add_syntax_tokens(tokenizer, model)

    # LoRA config (Low Rank Adaptation), used to train small low rank matrices inserted into the attention layer.
    # Chosen instead of updating all the weights during fine-tuning due to computation limits on colab. 
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES, bias='none',
    )
    model = get_peft_model(model, lora)
    # Check how many parameters are being trained. 
    model.print_trainable_parameters()

    # Preprocessed text files used to build train and validation datasets
    train_ds = make_dataset(train_file, tokenizer)
    val_ds = make_dataset(val_file, tokenizer)
    print(f'Train: {len(train_ds)} blocks, Val: {len(val_ds)} blocks') # get length of each. 

    # Training hyperparameters set
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE, gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, warmup_steps=config.WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        # Save and evaluate per epoch, one time, and keep best model checkpoint
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        logging_steps=50, report_to='none',
    )

    # DataCollectorForLanguageModeling deals with shifting labels for causal LM training, while min=False specifies it is causal
    # (next token prediction), as opposed to masked langauge modelling. 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    # Save LoRA adapter weights and tokenizer.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Compute Perplexity on the validation dataset. 
    ppl = math.exp(trainer.evaluate()['eval_loss'])
    print(f'Val perplexity: {ppl:.2f}')
    print(f'Saved to {output_dir}')


if __name__ == '__main__':
    main()
