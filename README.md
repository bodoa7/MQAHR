# MQAHR
# MQAHR: Medical Question-Answering Hallucination Reasoning Dataset


## 📖 Overview

**MQAHR (Medical Question-Answering Hallucination Reasoning)** is a comprehensive benchmark dataset designed to systematically evaluate the hallucination analysis capabilities of Large Language Models (LLMs) in medical reasoning contexts.

### Key Highlights

- **7,153 hallucinated reasoning samples** with detailed step-by-step thought processes
- **Causally grounded taxonomy** of 4 hallucination types based on cognitive failure modes
- **Multi-level difficulty annotations** (Easy, Medium, Hard)
- **Evaluation of 23 open-source LLMs** spanning general-purpose and medical domains
- **Step-level hallucination localization** for interpretability analysis

## 🎯 Motivation

Large language models are increasingly deployed in medical settings for diagnostic reasoning and clinical decision support. However, LLMs often generate **hallucinations**—responses that are factually inaccurate or logically inconsistent—which pose serious risks to patient safety. 

Most existing hallucination detection approaches focus on final outputs, overlooking how errors originate within the reasoning process. MQAHR addresses this gap by enabling systematic evaluation of **reasoning-level hallucination analysis**.

## 📊 Dataset Structure

### Hallucination Taxonomy

We propose a **causally grounded** classification of hallucinations in medical reasoning:

| Type | Description | Example |
|------|-------------|---------|
| **H1: Input Misunderstanding** | Misinterpretation of key conditions in the problem statement | Overlooking patient symptoms or constraints |
| **H2: Knowledge Confusion** | Errors in factual retrieval where related concepts are blended | Confusing similar diseases, drugs, or biomarkers |
| **H3: Knowledge Fabrication** | Invention of non-existent medical entities or mechanisms | Creating pseudo-medical facts without basis |
| **H4: Reasoning Bias** | Faulty propagation of causal relations during multi-step reasoning | Logically inconsistent conclusions despite correct facts |

### Data Format

```json
{
    "id": 4906,
    "question": "The forces acting through a fixed partial denture on the\nabutment tooth should be directed:",
    "options": "A As far as possible at right angles to the long axis of the teeth\nB Parallel to the long axis of the teeth\nC By decreasing the mesio-distal dimension\nD None of the above",
    "gold_answer": "B",
    "model_think_process": "[step1]: Okay, so I've got this question about fixed partial dentures and the direction of forces acting through the abutment tooth\n\n[step2]: Let me try to think this through step by step\n\n[step3]: First, I remember that abutment teeth are the ones that connect the pontic (the artificial tooth) to the existing natural teeth or abutment teeth\n\n[step4]: Forces on these teeth are important because they affect the stability and longevity of the denture\n\n[step5]: The question is asking about the direction these forces should be directed\n\n[step6]: The options are A, B, C, or D, which says none of the above\n\n[step7]: So I need to recall what I know about how forces act on abutment teeth\n\n[step8]: I think about the mechanics of fixed partial dentures\n\n[step9]: They are supported by the abutment teeth, which in turn are held in place by the pontic and the supporting structures\n\n[step10]: The forces that act on the abutment tooth are usually a combination of compressive, tensile, and shear forces\n\n[step11]: Wait, but the question is about direction\n\n[step12]: So I should focus on whether the forces are directed along the long axis, perpendicular to it, or in some other way\n\n[step13]: I remember that in dental biomechanics, the direction of forces can influence the stress distribution\n\n[step14]: If the forces are parallel to the long axis, they might cause more wear on the abutment tooth, especially if there are multiple teeth in the arch\n\n[step15]: On the other hand, if the forces are directed perpendicular to the long axis, that might distribute the stress more evenly or reduce wear\n\n[step16]: Also, I recall that when designing fixed partial dentures, the framework is often angled slightly backward to reduce the forward forces on the abutment teeth\n\n[step17]: This backward angulation helps to take the forces away from the abutment and distribute them more laterally, which is better for the tooth's longevity\n\n[step18]: So, if the forces are parallel to the long axis, they might lead to more stress along the length of the tooth, which isn't ideal\n\n[step19]: Perpendicular forces would spread the stress more laterally\n\n[step20]: Therefore, the optimal direction would be to have the forces directed perpendicular to the long axis\n\n[step21]: Looking at the options, A says as far as possible at right angles to the long axis\n\n[step22]: That seems to align with what I was thinking\n\n[step23]: B is parallel, which I think is less favorable\n\n[step24]: C talks about decreasing the mesio-distal dimension, which I'm not sure how that relates\n\n[step25]: Maybe that refers to reducing the width of the tooth, but the question is about force direction, not shape\n\n[step26]: So, putting it all together, the forces should be directed as far as possible at right angles to the long axis to minimize wear and stress on the abutment tooth\n\n[step27]: Therefore, the correct answer should be A.\n</think>",
    "inference_answer": "A",
    "hallucination_type": "H2",
    "first_step": "step14",
    "difficulty": "easy"
  },
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 7,153 |
| Average Reasoning Steps | 22.7 |
| Knowledge Confusion (H2) | 63.0% |
| Reasoning Bias (H4) | 24.2% |
| Knowledge Fabrication (H3) | 7.0% |
| Input Misunderstanding (H1) | 5.8% |

**Reasoning Length Distribution:**
- Short (1-15 steps): 43.0%
- Medium (16-30 steps): 27.4%
- Long (31-50 steps): 11.0%
- Very Long (51+ steps): 18.6%

## 💻 Code Structure

### Dataset Annotation
- **`GPT4o_annotations.py`** - Automated annotation pipeline using GPT-4o for initial hallucination classification and difficulty labeling

### Evaluation Scripts
- **`simplified_binary_hallucination_detection.py`** - Binary detection (hallucination present/absent) with simple prompts
- **`fine_grained_hallucination_detection.py`** - Detection with causally grounded instructions specifying hallucination types (H1-H4)
- **`hallucination_subtype_classification.py`** - Identifies specific hallucination subtypes and computes per-type performance metrics
- **`confidence_conditioned_detection.py`** - Detection with explicit confidence expression (High/Medium/Low)
- **`step_level_hallucination_localization.py`** - Locates the first reasoning step where hallucination occurs



## 🔬 Key Findings

### 1. **Reverse Scaling Effect**
Increasing parameter scale of instruction-tuned models does **not** improve hallucination-analysis performance—in fact, performance often declines:
- Qwen2.5-7B-it: **69.47%** sensitivity
- Qwen2.5-72B-it: **63.20%** sensitivity ↓

### 2. **Instruction Granularity Matters**
Fine-grained causally grounded instructions improve detection accuracy but increase task complexity:
- MedGemma-27B-text-it: 94.32% → **96.78%** ✓
- Lingshu-7B: 42.58% → **29.53%** ✗ (smaller models struggle)

### 3. **Confidence Expression Helps**
Incorporating explicit confidence scores yields an average improvement of **5.5%**:
- Qwen2.5-72B-it: 0.6953 → **0.8132** (+16.9%)
- Benefits well-calibrated models but introduces instability in multimodal architectures

### 4. **Text vs. Multimodal Models**
Text-based medical models significantly outperform multimodal counterparts:
- MedGemma-27B-text-it: **94.32%**
- MedGemma-27B-it (multimodal): **63.07%**

### 5. **Hallucination Subtype Difficulty**
| Type | Detection Difficulty |
|------|---------------------|
| Knowledge Confusion (H2) | ⭐ Easiest (highest recall) |
| Reasoning Bias (H4) | ⭐⭐ Moderate |
| Knowledge Fabrication (H3) | ⭐⭐⭐ Hard |
| Input Misunderstanding (H1) | ⭐⭐⭐ Hardest (recall < 0.15) |



## 📜 License
This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).


**⚠️ Important Note**: This dataset is intended for research purposes only. It should not be used directly for clinical decision-making without proper validation and oversight by medical professionals.
