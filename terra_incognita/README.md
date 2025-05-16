# 🐞 TerraIncognita: VLM-Based Insect Classification Benchmark

This repository evaluates **Vision-Language Models (VLMs)** on hierarchical insect classification tasks. Models are tested on their ability to recognize **known** and **novel** species across multiple taxonomic levels (Order, Family, Genus, Species), using real-world field-collected data.

## 📥 Dataset

The dataset can be downloaded from Hugging Face:

👉 [TerraIncognita on Hugging Face](https://huggingface.co/datasets/BGLab/TerraIncognita/)

```bash
pip install huggingface_hub

# Download using CLI
huggingface-cli login  # If private
git lfs install
git clone https://huggingface.co/datasets/BGLab/TerraIncognita/
```

Make sure the dataset folder contains:
- `new_species_100/` (input images)
- `new_species_100_classification.xlsx` (ground truth)

## 🧠 Inference & Evaluation Pipeline

### 📦 Requirements
- Python 3.8+
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`, `GROK_API_KEY`
- Required libraries: `pandas`, `openai`, `anthropic`, `google.generativeai`, etc.

## 📁 Directory Setup

```bash
cd /work/mech-ai/iNaturalist/DETReg/detection_tests/up-detr/VLM_Analysis/data_version4/
```

Ensure the following structure:
- `new_species_100/` – Input images
- `new_species_100_classification.xlsx` – Ground truth labels
- `vlm_eval_outputs/` – Output directory (created automatically)

## 🚀 Run the Evaluation Script

```python
import os
from vlm_inference.vlm_analysis import get_image_paths, parse_responses, evaluate_classification
from vlm_inference.vlm_calls import (
    run_gpt_inference, run_claude_inference, run_gemini_inference,
    run_together_inference, run_grok_inference
)

os.chdir('/work/mech-ai/iNaturalist/DETReg/detection_tests/up-detr/VLM_Analysis/data_version4/')
image_dir = "new_species_100"
ground_truth = "new_species_100_classification.xlsx"
output_dir = "vlm_eval_outputs/"
os.makedirs(output_dir, exist_ok=True)

models = {
    "gpt4o": lambda paths: run_gpt_inference(paths, api_key=OPENAI_API_KEY),
    "claude": lambda paths: run_claude_inference(paths, model="claude-3-opus-20240229", api_key=ANTHROPIC_API_KEY),
    "gemini": lambda paths: run_gemini_inference(paths, api_key=GEMINI_API_KEY),
    "llama4scout": lambda paths: run_together_inference(paths, model="meta-llama/Llama-4-Scout-17B-16E-Instruct", api_key=TOGETHER_API_KEY),
    "grok": lambda paths: run_grok_inference(paths, model="grok-2-vision-latest", api_key=GROK_API_KEY),
    # Add more models as needed
}

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

Each model's predictions are saved as `.xlsx` files in `vlm_eval_outputs/`, along with classification reports including F1 scores for every taxonomic level.
