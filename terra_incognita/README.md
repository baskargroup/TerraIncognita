# 🐞 VLM-Based Insect Classification Evaluation

This repository contains scripts to evaluate **Vision-Language Models (VLMs)** on insect classification tasks across hierarchical taxonomic levels (Order → Species). The code supports inference using models like GPT-4, Claude 3, Gemini 2.5, LLaMA-4, Grok, and Qwen2 VL.

## 📦 Prerequisites

Ensure you have the following installed and configured:

- Python 3.8+
- Required packages: `pandas`, `openai`, `anthropic`, `google.generativeai`, etc.
- API keys for:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GEMINI_API_KEY`
  - `TOGETHER_API_KEY`
  - `GROK_API_KEY`

## 📁 Directory Setup

```bash
cd /work/mech-ai/iNaturalist/DETReg/detection_tests/up-detr/VLM_Analysis/data_version4/
```

Ensure the following are present in this folder:

- `new_species_100/` → Directory containing input images
- `new_species_100_classification.xlsx` → Ground truth file
- `vlm_eval_outputs/` → Will be created automatically to save results

## 🧠 Run Inference and Evaluation

```python
import os
from vlm_inference.vlm_analysis import get_image_paths, parse_responses, evaluate_classification
from vlm_inference.vlm_calls import (
    run_gpt_inference, run_claude_inference, run_gemini_inference,
    run_together_inference, run_grok_inference
)

# Set working directory and paths
os.chdir('/work/mech-ai/iNaturalist/DETReg/detection_tests/up-detr/VLM_Analysis/data_version4/')
image_dir = "new_species_100"
ground_truth = "new_species_100_classification.xlsx"
output_dir = "vlm_eval_outputs/"
os.makedirs(output_dir, exist_ok=True)
```

### ✅ Define Model Wrappers

```python
models = {
    "gpt4o": lambda paths: run_gpt_inference(paths, api_key=OPENAI_API_KEY),
    "claude": lambda paths: run_claude_inference(paths, model="claude-3-opus-20240229", api_key=ANTHROPIC_API_KEY),
    "gemini": lambda paths: run_gemini_inference(paths, api_key=GEMINI_API_KEY),
    "llama4scout": lambda paths: run_together_inference(paths, model="meta-llama/Llama-4-Scout-17B-16E-Instruct", api_key=TOGETHER_API_KEY),
    "grok": lambda paths: run_grok_inference(paths, model="grok-2-vision-latest", api_key=GROK_API_KEY),
    # ... add more models as needed
}
```

### 🔍 Run Inference + Evaluation

```python
image_paths = get_image_paths(image_dir)

for model_name, inference_fn in models.items():
    print(f"\n🔍 Running inference with {model_name}...")
    results = inference_fn(image_paths)

    parsed_df = parse_responses(results)
    output_xlsx = os.path.join(output_dir, f"{model_name}_results.xlsx")

    print(f"🧪 Evaluating {model_name} predictions...")
    report = evaluate_classification(parsed_df, ground_truth, output_xlsx=output_xlsx)

    print(f"✅ Results saved to: {output_xlsx}")
    print(f"\n📊 F1 Scores for {model_name}:")
    for level in ["Order", "Family", "Genus", "Species"]:
        f1 = report[level]["macro avg"]["f1-score"]
        print(f"{level}: {f1:.3f}")
```

## 📈 Output

For each model, an Excel file with predictions and evaluation metrics is saved in `vlm_eval_outputs/`. F1 scores are printed per taxonomic level.
