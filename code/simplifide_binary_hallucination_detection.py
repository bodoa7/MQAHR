import os
import json
import time
import argparse
import numpy as np
import regex as re
import multiprocessing
from typing import Union
from pathlib import Path
from vllm import LLM, SamplingParams
from modelscope import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Medical hallucination detection (vLLM batch)")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--gpu_ids", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--only_hallucination", type=str, default="yes", choices=["yes", "no"])
    parser.add_argument("--save_interval", type=int
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--repetition_penalty", type=float)
    return parser.parse_args()

def extract_hallucination_result(text):
    
    matches = re.findall(r'\\boxed[\{\[]\s*(Hallucinations|No\s+Hallucinations)\s*[\}\]]', text, re.IGNORECASE)
    if matches:
        final = matches[-1].strip().lower().replace(" ", "")
        if final == "hallucinations":
            return "Hallucinations"
        elif final == "nohallucinations":
            return "No Hallucinations"
    return None

def extract_ground_truth(item: dict) -> Union[str, None]:
    return "Hallucinations"

def calculate_metrics(y_true, y_pred):
    valid_labels = {"Hallucinations", "No Hallucinations"}
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t in valid_labels and p in valid_labels]
    if not valid_pairs:
        return {"total_valid": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "true_positives": 0, "true_negatives": 0, "false_positives": 0, "false_negatives": 0}

    y_true_bool = np.array([1 if t == "Hallucinations" else 0 for t, _ in valid_pairs])
    y_pred_bool = np.array([1 if p == "Hallucinations" else 0 for _, p in valid_pairs])

    tp = np.sum((y_true_bool == 1) & (y_pred_bool == 1))
    tn = np.sum((y_true_bool == 0) & (y_pred_bool == 0))
    fp = np.sum((y_true_bool == 0) & (y_pred_bool == 1))
    fn = np.sum((y_true_bool == 1) & (y_pred_bool == 0))

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
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }

def build_prompt(question, options, thought_process):
    options_text = f"Options: {options} " if options and options.strip() else ""
    return f"""Instruction: You need to conduct hallucination detection on the following thought process. Follow these strict rules:
1) carefully check if the thought process contains **factual errors** (e.g., wrong medical concepts, incorrect function of structures/drugs, misstated institutional responsibilities) — these errors are defined as "hallucinations".
2) Output: \\boxed[Hallucinations] if there are hallucinations; \\boxed[No Hallucinations] if there are none. Do not include any other text.

Question: {question}
{options_text}
Thought Process: 
{thought_process}

Focus only on hallucination detection. Do NOT answer the medical question itself. Do NOT include any explanation. 
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

def main():
    args = parse_args()
    multiprocessing.freeze_support()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Load model
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

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        skip_special_tokens=True
    )

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)



    processed_ids, results = load_checkpoint(args.output_file)

    prompts, pending_data = [], []
    for item in data:
        if item.get("id") in processed_ids:
            continue
        q, o, tp = item["question"].strip(), item.get("options", ""), item["model_think_process"].strip()
        prompt = build_prompt(q, o, tp)
        # print(prompt)
        # input()
        messages = [{"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(full_prompt)
        pending_data.append(item)



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
            save_results(results, args.output_file)
        

    save_results(results, args.output_file)
    final_metrics = calculate_metrics(y_true, y_pred)
    
    for k, v in final_metrics.items():
        print(f"{k}: {v}")

    valid_labels = {"Hallucinations", "No Hallucinations"}
    invalid_samples = [
        r for r in results
        if r.get("predicted_hallucination") not in valid_labels or r.get("ground_truth_hallucination") not in valid_labels
    ]
    if invalid_samples:
        invalid_file = Path(args.output_file).with_name(Path(args.output_file).stem + "_invalid.json")
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump(invalid_samples, f, ensure_ascii=False, indent=2)
       

if __name__ == "__main__":
    main()
