

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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

SEEDS = [0]


def parse_args():
    parser = argparse.ArgumentParser(description="Medical hallucination detection (single seed, numeric output)")
    parser.add_argument("--model_path", type=str, )
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--gpu_ids", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--repetition_penalty", type=float)
    return parser.parse_args()


def extract_hallucination_result(text: str):
    
    matches = re.findall(r'\\boxed[\{\[]\s*([0-4])\s*[\}\]]', text)
    if not matches:
        matches = re.findall(r'(?<!\d)([0-4])(?!\d)', text)
    if matches:
        return matches[-1]
    return None


def extract_ground_truth(item: dict) -> Union[str, None]:
 
    htype = item.get("hallucination_type")
    if isinstance(htype, str):
        match = re.search(r'(\d+)', htype)
        if match:
            return match.group(1)
    return None


def calculate_metrics(y_true, y_pred):
 
    hallucination_labels = {"1", "2", "3", "4"}
    all_labels = {"0", "1", "2", "3", "4"}


    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred)
                   if t in hallucination_labels and p in all_labels]

    if not valid_pairs:
        return {
            "total_valid": 0,
            "Balanced_ACC": 0.0,
            "miss_rate_C0_pred": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "per_class_metrics": {}
        }


    y_true_int = np.array([int(t) for t, _ in valid_pairs])
    y_pred_int = np.array([int(p) for _, p in valid_pairs])


    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_int, y_pred_int, labels=[1, 2, 3, 4], zero_division=0
    )

    cm = confusion_matrix(y_true_int, y_pred_int, labels=[1, 2, 3, 4])
    acc_per_class = np.diag(cm) / np.sum(cm, axis=1)
    acc_per_class = np.nan_to_num(acc_per_class)


    macro_p = float(np.mean(precision))
    macro_r = float(np.mean(recall))
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r + 1e-8)

    
    miss_count = sum(1 for t, p in zip(y_true, y_pred)
                     if t in hallucination_labels and p == "0")
    total_h = sum(1 for t in y_true if t in hallucination_labels)
    miss_rate = miss_count / total_h if total_h > 0 else 0.0

    
    label_names = {
        1: "C1_Input_Misunderstanding",
        2: "C2_Knowledge_Error",
        3: "C3_Knowledge_Fabrication",
        4: "C4_Reasoning_Logic"
    }

    per_class = {
        label_names[lbl]: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "accuracy_within_class": round(float(a), 4),
            "support": int(s)
        }
        for lbl, p, r, f, s, a in zip(range(1, 5),
                                      precision, recall, f1, support, acc_per_class)
    }

    return {
        "total_valid": len(valid_pairs),
        "miss_rate_C0_pred": round(float(miss_rate), 4),
        "precision_macro": round(float(macro_p), 4),
        "recall_macro": round(float(macro_r), 4),
        "f1_macro": round(float(macro_f), 4),
        "per_class_metrics": per_class
    }


def build_prompt(question, options, thought_process):
    options_text = f"Options: {options} " if options and options.strip() else ""
    return f"""Instruction: You need to conduct hallucination detection on the following thought process.
### Hallucination Type Definitions
\\boxed[1] — Input Misunderstanding Hallucination: Misinterprets the question or misses key information or constraints, leading reasoning in the wrong direction. 
\\boxed[2] — Knowledge Error Hallucination: Contains factual medical mistakes or confusion inconsistent with authoritative medical knowledge. 
\\boxed[3] — Knowledge Fabrication Hallucination: Mentions non-existent medical concepts, drugs, anatomy, or fabricated data beyond real-world medicine. 
\\boxed[4] — Reasoning Logic Hallucination: Shows faulty reasoning chains, invalid causality, or ignored logical conditions that lead to incorrect conclusions. 
\\boxed[0] — No hallucination: The reasoning is medically accurate and logically consistent.

Question: {question}
{options_text}
Thought Process:
{thought_process}

### strictly Output Format: \\boxed[0], \\boxed[1], \\boxed[2], \\boxed[3], or \\boxed[4]
Focus only on hallucination detection. Do NOT answer the question itself.
Detection Answer:
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
        if len(token_ids) > llm.llm_engine.model_config.max_model_len - 512:
          
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


    summary_file = Path(str(out_file).replace(".json", "_summary.json"))
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    return final_metrics


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

    sd = SEEDS[0]
    run_one_seed(sd, llm, tokenizer, data, args)

if __name__ == "__main__":
    main()
