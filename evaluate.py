import json, math, os

import torch
import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

import config


def load_model(checkpoint_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load saved tokenizer and checkpoint 
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Load GPT2 model in float16 for GPU or float32 for CPU, due to storage and speed.
    dtype = torch.float16 if device == 'cuda' else torch.float32
    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, dtype=dtype)

    # Tagged model has extra tokens in vocab, so embedding needs to be resized to match that of training.
    base.resize_token_embeddings(len(tokenizer))

    # Load LoRA adapter with base model, set to eval. 
    model = PeftModel.from_pretrained(base, checkpoint_dir).eval().to(device)
    return model, tokenizer


def perplexity(model, tokenizer, text_file):
    device = next(model.parameters()).device

    # Tokenize test file as one long sequence. 
    with open(text_file, encoding='utf-8') as f:
        ids = tokenizer.encode(f.read(), add_special_tokens=False)

    loss = 0
    n = 0
    with torch.no_grad():
        # Slide a window of BLOCK_SIZE tokens across the sequence
        # Sliding window of block size tokens accross sequence. 
        for i in range(0, len(ids) - config.BLOCK_SIZE, config.BLOCK_SIZE):
            # Convert slice to tensor.
            chunk = torch.tensor(ids[i:i+config.BLOCK_SIZE]).unsqueeze(0).to(device)
            # If labels = chunk, model computes cross-entropy loss internally.
            loss += model(chunk, labels=chunk).loss.item()
            n += 1

    # Perplexity is exponent of average cross-entropy loss over all chunks. 
    return math.exp(loss / n)


class BlockBrackets(LogitsProcessor):
    # During beam search (inference), zeros out bracket tokens to prevent them from appearing in 
    # text generation output, as it was trained on text that contained them.
    def __init__(self, suppress_ids):
        self.suppress_ids = suppress_ids

    def __call__(self, input_ids, scores):
        # Logit for each bracket token set to -inf so it has probability 0. 
        scores[:, self.suppress_ids] = float('-inf')
        return scores


def generate(model, tokenizer, prompts, suppress_tokens=None):
    device = next(model.parameters()).device

    # Builds list for LogitsProcessor, empty for raw mode and for tagged having BlockBrackets.
    processor = LogitsProcessorList()
    if suppress_tokens:
        # Token strings converted to vocab ids.
        ids = [tokenizer.convert_tokens_to_ids(t) for t in suppress_tokens]
        # Drop unknown tokens (unk) that are out of vocab. 
        ids = [i for i in ids if i != tokenizer.unk_token_id]
        processor.append(BlockBrackets(ids))

    results = []
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize prompt 
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            out = model.generate(
                **inputs,
                # Amount of new tokens to generate beyond prompt
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS, num_beams=config.EVAL_NUM_BEAMS, early_stopping=True,
                pad_token_id=tokenizer.eos_token_id, logits_processor=processor,
                # Discourages repetition, otherwise model loops phrases and collapses into a degenerate solution.
                repetition_penalty=config.REPETITION_PENALTY, no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            )
            # Decode newly generated tokens and skip prompt ones. 
            results.append(tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
    return results


def evaluate_model(mode, prompts, references):
    # Use correct checkpoint and test file for mode. 
    checkpoint = config.RAW_OUTPUT_DIR if mode == 'raw' else config.TAGGED_OUTPUT_DIR
    test_file = config.RAW_TEST_FILE if mode == 'raw' else config.TAGGED_TEST_FILE

    print(f'\n--- {mode.upper()} ---')
    model, tokenizer = load_model(checkpoint)

    # Perplexity on test set
    ppl = perplexity(model, tokenizer, test_file)
    print(f'Perplexity: {ppl:.2f}')


    # Tagged model should not have bracket tokens appearing, output needs to be clean. Suppresses them.
    suppress = config.SYNTACTIC_SPECIAL_TOKENS if mode == 'tagged' else None # if raw, no suppression of course.
    hyps = generate(model, tokenizer, prompts, suppress_tokens=suppress)

    # BLEU to measure n-gram overlap, -1 to -4 to check unigram to 4-gram overlap.
    # Smoothing to prevent a score of 0 when there are no matches. 
    smoother = SmoothingFunction().method1
    # BLEU expects references as list of lists, multiple are allowed per sentence.
    ref_toks = [[ref.split()] for ref in references]
    hyp_toks = [hyp.split() for hyp in hyps]
    scores = {'mode': mode, 'perplexity': ppl}
    for n, w in enumerate([(1,0,0,0), (.5,.5,0,0), (1/3,1/3,1/3,0), (.25,.25,.25,.25)], 1):
        scores[f'bleu_{n}'] = corpus_bleu(ref_toks, hyp_toks, weights=w, smoothing_function=smoother)

    # ROUGE measures recall of reference n-grams in hypothesis, -1 unigrams, -2 bigrams, and -L longest commen subsequence.
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    r1 = r2 = rL = 0
    for ref, hyp in zip(references, hyps):
        s = rouge.score(ref, hyp)
        r1 += s['rouge1'].fmeasure
        r2 += s['rouge2'].fmeasure
        rL += s['rougeL'].fmeasure
    # Average accross all prompt and reference pairs.
    scores['rouge1'] = r1 / len(references)
    scores['rouge2'] = r2 / len(references)
    scores['rougeL'] = rL / len(references)

    # BERTScore uses pretrained Italian BERT model for semantic similarity between hypothesis and reference, 
    P, R, F = bert_score_fn(hyps, references,
        lang='it', model_type='dbmdz/bert-base-italian-cased', num_layers=9, verbose=False)
    scores['bertscore_p'] = P.mean().item()   # precision
    scores['bertscore_r'] = R.mean().item()   # recall
    scores['bertscore_f1'] = F.mean().item()  # F1

    print(f'bleu: {scores["bleu_1"]:.4f} / {scores["bleu_2"]:.4f} / {scores["bleu_3"]:.4f} / {scores["bleu_4"]:.4f}')
    print(f'rouge: {scores["rouge1"]:.4f} / {scores["rouge2"]:.4f} / {scores["rougeL"]:.4f}')
    print(f'bertscore f1: {scores["bertscore_f1"]:.4f}')
    return scores, hyps


def main():
    # NLTK tokenizer needed for BLEU computation.
    nltk.download('punkt', quiet=True)

    # Odd lines prompts, even references. 
    with open(config.EVAL_PROMPTS_FILE, encoding='utf-8') as f:
        lines = [l.rstrip() for l in f if l.strip()]
    prompts = lines[::2]
    references = lines[1::2]

    # Eval for both modes and results
    results = []
    for mode in ['raw', 'tagged']:
        scores, hyps = evaluate_model(mode, prompts, references)
        # Store hypothesis with metrics 
        scores['hypotheses'] = hyps
        results.append(scores)

    # Comparison of all metrics.
    metrics = ['perplexity', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
               'rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    raw, tagged = results
    print()
    for m in metrics:
        print(f'{m}: raw={raw[m]:.4f}, tagged={tagged[m]:.4f}')

    # Save results to JSON for notebook to load them for plotting. 
    os.makedirs('results', exist_ok=True)
    with open('results/eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\nSaved to results/eval_results.json')


if __name__ == '__main__':
    main()
