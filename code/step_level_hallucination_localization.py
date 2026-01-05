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
    parser = argparse.ArgumentParser(description="Medical hallucination step detection (right/wrong evaluation)")
    parser.add_argument("--model_path", type=str)
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
    
    matches = re.findall(r'\\boxed[\{\[]\s*(\d+)\s*[\}\]]', text)
    if not matches:
        matches = re.findall(r'(?<!\d)(\d+)(?!\d)', text)
    if matches:
        return matches[-1]
    return None


def extract_ground_truth(item: dict) -> Union[str, None]:
    
    step = item.get("first_step")
    if step is None:
        return None
    step_str = str(step).strip().lower()
    match = re.search(r'(\d+)', step_str)
    if match:
        return match.group(1)
    return None


def calculate_metrics(y_true, y_pred):
    
    if not y_true or not y_pred:
        return {
            "total_valid": 0,
            "accuracy": 0.0
        }

    y_true_int = np.array([int(t) for t in y_true])
    y_pred_int = np.array([int(p) for p in y_pred])
    correct = np.sum(y_true_int == y_pred_int)
    acc = correct / len(y_true_int)

    return {
        "total_valid": len(y_true_int),
        "correct": int(correct),
        "accuracy": round(float(acc), 4)
    }


def build_prompt(question, options, thought_process):
    options_text = f"Options: {options} " if options and options.strip() else ""
    return f"""Instruction: You need to detect whether the following medical reasoning process contains any hallucination, and if so, identify the first step where it appears. Follow these strict rules:
### Hallucination Type Definitions

Input Misunderstanding Hallucination — Misunderstands the question or context, missing key information or constraints and leading the reasoning in a wrong direction.

Knowledge Error Hallucination — Contains factual medical errors where statements or judgments conflict with authoritative medical knowledge.

Knowledge Fabrication Hallucination — Introduces non-existent medical concepts, entities, or data that deviate from real medical knowledge.

Reasoning Logic Hallucination — Exhibits logical inconsistencies, ignored conditions, or invalid conclusions, causing the reasoning chain to deviate from factual validity.

Question: {question}
{options_text}
Thought Process:
{thought_process}

### Output Format
- If no hallucination is found, output:
  \\boxed[0]
- If hallucinations exist, output only the first step number, e.g. \\boxed[9].
Focus only on thought process hallucination detection. Do NOT answer the medical question itself. Do NOT include any explanation.
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
            input()
        except Exception as e:
     
            continue

        batch_results = []
        for item, output in zip(batch_items, outputs):
            text = output.outputs[0].text.strip()
            pred = extract_hallucination_result(text)
            gt = extract_ground_truth(item)

            correct = (pred == gt) if (pred and gt) else False

            batch_results.append({
                "id": item["id"],
                "question": item["question"],
                "options": item.get("options", ""),
                "model_think_process": item["model_think_process"],
                "model_generated_text": text,
                "predicted_first_step": pred,
                "ground_truth_first_step": gt,
                "is_correct": correct,
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
  

    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
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
    data=data[:400]
    sd = SEEDS[0]
    run_one_seed(sd, llm, tokenizer, data, args)

if __name__ == "__main__":
    main()
