# Pretrained Italian GPT2 model, base for fine tuning. 
BASE_MODEL = "GroNLP/gpt2-small-italian"

# Special tokens added to vocab to mark phrase boundaries in tagged corpus. Start and end tokens for learning syntactic structures.
# NP = noun phrase, VP = verb phrase, PP = prepositional phrase, AP = adjective phrase, and ADVP = adverb phrase.
SYNTACTIC_SPECIAL_TOKENS = [
    "[NP_START]", "[NP_END]",
    "[VP_START]", "[VP_END]",
    "[PP_START]", "[PP_END]",
    "[AP_START]", "[AP_END]",
    "[ADVP_START]", "[ADVP_END]",
]

# Preprocessed text files paths, for each split and corpus version. 
# "raw" = tokens with no syntactic tags, raw text
# "tagged" = same sentences, but with constituency bracket tokens inserted. 
RAW_TRAIN_FILE = "data/raw_train.txt"
RAW_VAL_FILE = "data/raw_val.txt"
RAW_TEST_FILE = "data/raw_test.txt"
TAGGED_TRAIN_FILE = "data/tagged_train.txt"
TAGGED_VAL_FILE = "data/tagged_val.txt"
TAGGED_TEST_FILE = "data/tagged_test.txt"

EVAL_PROMPTS_FILE = "data/eval_prompts.txt"

# 1024 token context window for GPT2, 256 speedens training and reduces memrory usage. 
BLOCK_SIZE = 256

BATCH_SIZE = 8
# Due to grad_accum, effective batch size is 32 as 4 * 8
GRAD_ACCUM = 4

LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# LoRA rank, controls capacity of low-rank adapter.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# c_attn = combined QKV projection in attention module, c_proj = output projection, layers where lora is applied to.
LORA_TARGET_MODULES = ["c_attn", "c_proj"]

RAW_OUTPUT_DIR = "checkpoints/raw"
TAGGED_OUTPUT_DIR = "checkpoints/tagged"

EVAL_MAX_NEW_TOKENS = 100
EVAL_NUM_BEAMS = 4
# Values > 1 reduce repetition, otherwise model leads to degenerate solutions where it just loops phrases. 
REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM_SIZE = 3
