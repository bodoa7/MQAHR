import os
import json
import time
import argparse
import numpy as np
import regex as re
import multiprocessing
import random
from typing import Union, Dict, List
from pathlib import Path
from vllm import LLM, SamplingParams
from modelscope import AutoTokenizer

SEEDS = [0] 


def parse_args():
    parser = argparse.ArgumentParser(description="Medical hallucination detection (vLLM batch, 5 seeds, numeric output)")
    parser.add_argument("--model_path", type=str )
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--gpu_ids", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--only_hallucination", type=str, default="yes", choices=["yes", "no"])
    parser.add_argument("--save_interval", type=int, )
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--repetition_penalty", type=float)
    return parser.parse_args()


def extract_hallucination_result(text):

    matches = re.findall(r'\\boxed[\{\[]\s*([01])\s*[\}\]]', text)
    if not matches:
    
        matches = re.findall(r'\b([01])\b', text)
    if matches:
        return matches[-1] 
    return None


def extract_ground_truth(item: dict) -> Union[str, None]:
    return "1"


def calculate_metrics(y_true, y_pred):
    valid_labels = {"0", "1"}
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in valid_labels and p in valid_labels]
    if not valid_pairs:
        return {"total_valid": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "true_positives": 0, "true_negatives": 0, "false_positives": 0, "false_negatives": 0}

    y_true_bool = np.array([int(t) for t, _ in valid_pairs])
    y_pred_bool = np.array([int(p) for _, p in valid_pairs])

    tp = int(np.sum((y_true_bool == 1) & (y_pred_bool == 1)))
    tn = int(np.sum((y_true_bool == 0) & (y_pred_bool == 0)))
    fp = int(np.sum((y_true_bool == 0) & (y_pred_bool == 1)))
    fn = int(np.sum((y_true_bool == 1) & (y_pred_bool == 0)))

    accuracy = (tp + tn) / len(valid_pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "total_valid": len(valid_pairs),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }


def build_prompt(question, options, thought_process):
    options_text = f"Options: {options} " if options and options.strip() else ""
    return f"""Instruction: You need to conduct hallucination detection on the following thought process. Follow these strict rules:
### Hallucination Type Definitions

Input Misunderstanding Hallucination — Misunderstands the question or context, missing key information or constraints and leading the reasoning in a wrong direction.

Knowledge Error Hallucination — Contains factual medical errors where statements or judgments conflict with authoritative medical knowledge.

Knowledge Fabrication Hallucination — Introduces non-existent medical concepts, entities, or data that deviate from real medical knowledge.

Reasoning Bias Hallucination — Exhibits logical inconsistencies, ignored conditions, or invalid conclusions, causing the reasoning chain to deviate from factual validity.

Question: {question}
{options_text}
Thought Process:
{thought_process}

Output \\boxed[1] if there are any type hallucinations; \\boxed[0] if there are none. Detection Confidence level: High / Medium / Low  

Focus only on thought process hallucination detection. Do NOT answer the medical question itself. Do NOT include any explanation.
"""


def save_results(results, output_file):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_checkpoint(output_file):
    if not Path(output_file).exists():
        return set(), []
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {item["id"] for item in results}, results

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def run_one_seed(seed: int, llm: LLM, tokenizer, data: List[dict], args) -> Dict[str, float]:
    set_all_seeds(seed)
    stem = Path(args.output_file).with_suffix("")
    out_file = Path(str(stem) + f"_seed{seed}.json")

    processed_ids, results = load_checkpoint(out_file)

    prompts, pending_data = [], []
    for item in data:
        if item.get("id") in processed_ids:
            continue
        q, o, tp = item["question"].strip(), item.get("options", ""), item["model_think_process"].strip()
        prompt = build_prompt(q, o, tp)
        messages = [{"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


        token_ids = tokenizer(full_prompt).input_ids
        if len(token_ids) > llm.llm_engine.model_config.max_model_len-512:
            continue

        prompts.append(full_prompt)
        pending_data.append(item)



    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        seed=seed,
    )

    batch_size = args.batch_size * len(args.gpu_ids.split(","))
    y_true, y_pred = [], []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_items = pending_data[i:i + batch_size]

        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            continue

        batch_results = []
        for item, output in zip(batch_items, outputs):
            text = output.outputs[0].text.strip()
            pred = extract_hallucination_result(text)
            gt = extract_ground_truth(item)

            batch_results.append({
                "id": item["id"],
                "question": item["question"],
                "options": item.get("options", ""),
                "model_think_process": item["model_think_process"],
                "model_generated_text": text,
                "predicted_hallucination": pred,
                "ground_truth_hallucination": gt,
                "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            if pred and gt:
                y_true.append(gt)
                y_pred.append(pred)

        results.extend(batch_results)
        if (i // batch_size + 1) % args.save_interval == 0:
            save_results(results, out_file)
            print(f"[seed={seed}] metrics (so far): {calculate_metrics(y_true, y_pred)}")

    save_results(results, out_file)
    final_metrics = calculate_metrics(y_true, y_pred)

    for k, v in final_metrics.items():
        print(f"{k}: {v}")

 
    valid_labels = {"0", "1"}
    invalid_samples = [
        r for r in results
        if r.get("predicted_hallucination") not in valid_labels or r.get("ground_truth_hallucination") not in valid_labels
    ]
    if invalid_samples:
        invalid_file = Path(out_file).with_name(Path(out_file).stem + "_invalid.json")
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump(invalid_samples, f, ensure_ascii=False, indent=2)
 

    return final_metrics

def aggregate_metrics(per_seed_metrics: List[Dict[str, float]]) -> Dict[str, dict]:
    keys = ["accuracy", "precision", "recall", "f1"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics]
        agg[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }
    tv = [m.get("total_valid", 0) for m in per_seed_metrics]
    agg["total_valid"] = {
        "mean": float(np.mean(tv)),
        "min": int(np.min(tv)),
        "max": int(np.max(tv)),
    }
    return agg


def main():
    args = parse_args()
    multiprocessing.freeze_support()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=len(args.gpu_ids.split(",")),
        trust_remote_code=True,
        dtype="auto",
        max_num_seqs=16,
        max_model_len=args.max_tokens,
        gpu_memory_utilization=0.9
    )

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_seed_metrics = []
    for sd in SEEDS:
        m = run_one_seed(sd, llm, tokenizer, data, args)
        per_seed_metrics.append(m)

    summary = {
        "seeds": SEEDS,
        "per_seed_metrics": per_seed_metrics,
        "aggregate": aggregate_metrics(per_seed_metrics)
    }

    summary_file = Path(args.output_file).with_suffix("")
    summary_file = Path(str(summary_file) + "_summary_seeds.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    main()
