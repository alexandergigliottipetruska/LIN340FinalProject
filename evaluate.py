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
    with open(text_file, encoding='utf-8') as f:
        text = f.read()
    ids = tokenizer.encode(text, add_special_tokens=False)
    blocks = range(0, len(ids) - config.BLOCK_SIZE, config.BLOCK_SIZE)
    loss = 0
    with torch.no_grad():
        for i in blocks:
            chunk = torch.tensor(ids[i:i+config.BLOCK_SIZE]).unsqueeze(0).to(device)
            loss += model(chunk, labels=chunk).loss.item()
    return math.exp(loss / len(blocks))


class BlockBrackets(LogitsProcessor):
    def __init__(self, suppress_ids):
        self.suppress_ids = suppress_ids

    def __call__(self, input_ids, scores):
        scores[:, self.suppress_ids] = float('-inf')
        return scores


def generate(model, tokenizer, prompts, suppress_tokens=None):
    device = next(model.parameters()).device
    processor = LogitsProcessorList()
    if suppress_tokens:
        ids = [tokenizer.convert_tokens_to_ids(t) for t in suppress_tokens]
        ids = [i for i in ids if i != tokenizer.unk_token_id]
        if ids:
            processor.append(BlockBrackets(ids))
    results = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
                num_beams=config.EVAL_NUM_BEAMS,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=processor,
                repetition_penalty=config.REPETITION_PENALTY,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            )
            results.append(tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
    return results


def evaluate_model(mode, prompts, references):
    checkpoint = config.RAW_OUTPUT_DIR if mode == 'raw' else config.TAGGED_OUTPUT_DIR
    test_file = config.RAW_TEST_FILE if mode == 'raw' else config.TAGGED_TEST_FILE

    print(f'\n--- {mode.upper()} ---')
    model, tokenizer = load_model(checkpoint)

    ppl = perplexity(model, tokenizer, test_file)
    print(f'Perplexity: {ppl:.2f}')

    suppress = config.SYNTACTIC_SPECIAL_TOKENS if mode == 'tagged' else None
    hyps = generate(model, tokenizer, prompts, suppress_tokens=suppress)

    # BLEU
    smoother = SmoothingFunction().method1
    r = [[ref.split()] for ref in references]
    h = [hyp.split() for hyp in hyps]
    scores = {'mode': mode, 'perplexity': ppl}
    for n, w in enumerate([(1,0,0,0), (.5,.5,0,0), (1/3,1/3,1/3,0), (.25,.25,.25,.25)], 1):
        scores[f'bleu_{n}'] = corpus_bleu(r, h, weights=w, smoothing_function=smoother)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    r1, r2, rL = 0., 0., 0.
    for ref, hyp in zip(references, hyps):
        s = scorer.score(ref, hyp)
        r1 += s['rouge1'].fmeasure
        r2 += s['rouge2'].fmeasure
        rL += s['rougeL'].fmeasure
    n = len(references)
    scores['rouge1'] = r1 / n
    scores['rouge2'] = r2 / n
    scores['rougeL'] = rL / n

    # BERTScore
    P, R, F = bert_score_fn(hyps, references, lang='it',
                             model_type='dbmdz/bert-base-italian-cased',
                             num_layers=9, verbose=False)
    scores['bertscore_p'] = P.mean().item()
    scores['bertscore_r'] = R.mean().item()
    scores['bertscore_f1'] = F.mean().item()

    for k, v in scores.items():
        if k != 'mode':
            print(f'{k}: {v:.4f}')

    return scores, hyps


def main():
    nltk.download('punkt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['raw', 'tagged', 'both'], default='both')
    parser.add_argument('--prompts_file', default=config.EVAL_PROMPTS_FILE)
    args = parser.parse_args()

    with open(args.prompts_file, encoding='utf-8') as f:
        lines = [l.rstrip() for l in f if l.strip()]

    prompts = lines[0::2]
    references = lines[1::2]
    modes = ['raw', 'tagged'] if args.mode == 'both' else [args.mode]

    results = []
    for mode in modes:
        r, hyps = evaluate_model(mode, prompts, references)
        r['hypotheses'] = hyps
        results.append(r)

    metrics = ['perplexity', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
               'rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    print()
    print(f"{'Metric':<22}" + '  '.join(r['mode'].upper() for r in results))
    print('-' * 50)
    for m in metrics:
        print(f'{m:<22}' + '  '.join(f'{r[m]:.4f}' for r in results))

    os.makedirs('results', exist_ok=True)
    with open('results/eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\nSaved to results/eval_results.json')


if __name__ == '__main__':
    main()
