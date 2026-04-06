# Pretrained Italian GPT-2 model from HuggingFace, used as the base for fine-tuning
BASE_MODEL = "GroNLP/gpt2-small-italian"

# Special tokens added to the tokenizer vocabulary to mark phrase boundaries in the tagged corpus.
# Each phrase type gets a START and END token so the model can learn syntactic structure.
# NP = noun phrase, VP = verb phrase, PP = prepositional phrase,
# AP = adjective phrase, ADVP = adverb phrase
SYNTACTIC_SPECIAL_TOKENS = [
    "[NP_START]", "[NP_END]",
    "[VP_START]", "[VP_END]",
    "[PP_START]", "[PP_END]",
    "[AP_START]", "[AP_END]",
    "[ADVP_START]", "[ADVP_END]",
]

# Paths to the preprocessed text files for each split and corpus version.
# "raw" = surface-form tokens with no syntactic tags
# "tagged" = same sentences but with constituency bracket tokens inserted
RAW_TRAIN_FILE = "data/raw_train.txt"
RAW_VAL_FILE = "data/raw_val.txt"
RAW_TEST_FILE = "data/raw_test.txt"
TAGGED_TRAIN_FILE = "data/tagged_train.txt"
TAGGED_VAL_FILE = "data/tagged_val.txt"
TAGGED_TEST_FILE = "data/tagged_test.txt"

# Prompt/reference pairs used during evaluation — one prompt per line, next line is reference
EVAL_PROMPTS_FILE = "data/eval_prompts.txt"

# Number of tokens per training chunk — GPT-2 has a 1024 token context window,
# we use 256 to keep training fast and reduce memory usage
BLOCK_SIZE = 256

# Training batch size per GPU — effective batch size = BATCH_SIZE * GRAD_ACCUM = 32
BATCH_SIZE = 8

# Gradient accumulation: simulate a larger batch by accumulating gradients over 4 steps
# before updating weights (saves GPU memory)
GRAD_ACCUM = 4

# AdamW learning rate — 3e-4 is a standard starting point for LoRA fine-tuning
LEARNING_RATE = 3e-4

# Total training epochs
NUM_EPOCHS = 10

# Number of warmup steps — linearly increases LR from 0 to LEARNING_RATE over first 100 steps
WARMUP_STEPS = 100

# L2 regularization to prevent overfitting
WEIGHT_DECAY = 0.01

# LoRA rank — controls how many parameters are added; higher rank = more capacity but slower
LORA_R = 16

# LoRA scaling factor — typically set to 2 * LORA_R
LORA_ALPHA = 32

# Dropout applied to LoRA layers during training
LORA_DROPOUT = 0.05

# Which attention weight matrices to apply LoRA to.
# c_attn = combined QKV projection, c_proj = output projection (GPT-2 naming convention)
LORA_TARGET_MODULES = ["c_attn", "c_proj"]

# Where to save model checkpoints after training
RAW_OUTPUT_DIR = "checkpoints/raw"
TAGGED_OUTPUT_DIR = "checkpoints/tagged"

# Maximum number of new tokens to generate during evaluation
EVAL_MAX_NEW_TOKENS = 100

# Number of beams for beam search — more beams = better quality but slower
EVAL_NUM_BEAMS = 4

# Penalises the model for repeating the same tokens — values > 1 reduce repetition
REPETITION_PENALTY = 1.3

# Prevents any n-gram of this size from appearing more than once in the output
NO_REPEAT_NGRAM_SIZE = 3
