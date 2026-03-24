BASE_MODEL = "GroNLP/gpt2-small-italian"

SYNTACTIC_SPECIAL_TOKENS = [
    "[NP_START]", "[NP_END]",
    "[VP_START]", "[VP_END]",
    "[PP_START]", "[PP_END]",
    "[AP_START]", "[AP_END]",
    "[ADVP_START]", "[ADVP_END]",
]

RAW_TRAIN_FILE = "data/raw_train.txt"
RAW_VAL_FILE = "data/raw_val.txt"
RAW_TEST_FILE = "data/raw_test.txt"
TAGGED_TRAIN_FILE = "data/tagged_train.txt"
TAGGED_VAL_FILE = "data/tagged_val.txt"
TAGGED_TEST_FILE = "data/tagged_test.txt"
EVAL_PROMPTS_FILE = "data/eval_prompts.txt"

BLOCK_SIZE = 256
BATCH_SIZE = 8
GRAD_ACCUM = 4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj"]

RAW_OUTPUT_DIR = "checkpoints/raw"
TAGGED_OUTPUT_DIR = "checkpoints/tagged"
EVAL_MAX_NEW_TOKENS = 100
EVAL_NUM_BEAMS = 4
REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM_SIZE = 3
