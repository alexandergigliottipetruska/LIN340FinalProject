import argparse
import json
import math
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from peft import PeftModel

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

import config


def load_model(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    base.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def perplexity(model, tokenizer, text_file):
    device = next(model.parameters()).device
    with open(text_file, encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text, add_special_tokens=False)
    block_size = config.BLOCK_SIZE
    loss, n_tok = 0.0, 0

    with torch.no_grad():
        for i in range(0, len(ids) - block_size, block_size):
            chunk = torch.tensor(ids[i : i + block_size]).unsqueeze(0).to(device)
            loss += model(chunk, labels=chunk).loss.item() * block_size
            n_tok += block_size

    return math.exp(loss / n_tok)


class BlockBrackets(LogitsProcessor):
    def __init__(self, suppress_ids):
        self.suppress_ids = suppress_ids

    def __call__(self, input_ids, scores):
        scores[:, self.suppress_ids] = float("-inf")
        return scores


def generate(model, tokenizer, prompts, suppress_tokens=None):
    device = next(model.parameters()).device

    logits_processor = LogitsProcessorList()
    if suppress_tokens:
        ids = [tokenizer.convert_tokens_to_ids(t) for t in suppress_tokens]
        ids = [i for i in ids if i != tokenizer.unk_token_id]
        if ids:
            logits_processor.append(BlockBrackets(ids))

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                num_beams=config.EVAL_NUM_BEAMS,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processor,
                repetition_penalty=config.REPETITION_PENALTY,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return results


def bleu_scores(references, hypotheses):
    refs = [[r.split()] for r in references]
    hyps = [h.split() for h in hypotheses]
    smoother = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f"bleu_{n}"] = corpus_bleu(refs, hyps, weights=weights, smoothing_function=smoother)
    return scores


def rouge_scores(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref, hyp in zip(references, hypotheses):
        s = scorer.score(ref, hyp)
        for k in agg:
            agg[k] += s[k].fmeasure
    return {k: v / len(references) for k, v in agg.items()}


def bertscore(references, hypotheses):
    P, R, F = bert_score_fn(
        hypotheses, references,
        lang="it",
        model_type="dbmdz/bert-base-italian-cased",
        num_layers=9,
        verbose=False,
    )
    return {"bertscore_p": P.mean().item(), "bertscore_r": R.mean().item(), "bertscore_f1": F.mean().item()}


def evaluate_model(mode, prompts, references):
    checkpoint = config.RAW_OUTPUT_DIR if mode == "raw" else config.TAGGED_OUTPUT_DIR
    test_file = config.RAW_TEST_FILE if mode == "raw" else config.TAGGED_TEST_FILE

    print(f"\n--- {mode.upper()} ---")
    model, tokenizer = load_model(checkpoint)

    ppl = perplexity(model, tokenizer, test_file)
    print(f"Perplexity: {ppl:.2f}")

    suppress = config.SYNTACTIC_SPECIAL_TOKENS if mode == "tagged" else None
    hypotheses = generate(model, tokenizer, prompts, suppress_tokens=suppress)

    bleu = bleu_scores(references, hypotheses)
    rouge = rouge_scores(references, hypotheses)
    bs = bertscore(references, hypotheses)

    for k, v in {**bleu, **rouge, **bs}.items():
        print(f"{k}: {v:.4f}")

    return {"mode": mode, "perplexity": ppl, **bleu, **rouge, **bs}, hypotheses


def main():
    nltk.download("punkt", quiet=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["raw", "tagged", "both"], default="both")
    parser.add_argument("--prompts_file", default=config.EVAL_PROMPTS_FILE)
    args = parser.parse_args()

    with open(args.prompts_file, encoding="utf-8") as f:
        lines = [l.rstrip() for l in f if l.strip() and not l.startswith("#")]

    if len(lines) % 2 != 0:
        raise ValueError("expected even number of lines (prompt/reference pairs)")

    prompts = lines[0::2]
    references = lines[1::2]

    modes = ["raw", "tagged"] if args.mode == "both" else [args.mode]
    results = []
    outputs = {}

    for mode in modes:
        r, hyps = evaluate_model(mode, prompts, references)
        results.append(r)
        outputs[mode] = hyps

    metrics = ["perplexity", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
               "rouge1", "rouge2", "rougeL", "bertscore_f1"]
    print(f"\n{'Metric':<22}" + "".join(f"{r['mode'].upper():>12}" for r in results))
    for m in metrics:
        print(f"{m:<22}" + "".join(f"{r[m]:>12.4f}" for r in results))

    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "hypotheses": outputs}, f, ensure_ascii=False, indent=2)
    print("\nSaved to results/eval_results.json")


if __name__ == "__main__":
    main()
