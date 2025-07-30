# 🐞 TerraIncognita: A Dynamic Benchmark for Species Discovery Using Frontier Models

## [**Project Page**](https://baskargroup.github.io/TerraIncognita/) | [**GitHub**](https://github.com/baskargroup/TerraIncognita) | [**Hugging Face**](https://huggingface.co/datasets/BGLab/TerraIncognita/) | [**Paper**](https://arxiv.org/html/2506.03182v1)
This repository evaluates **Vision-Language Models (VLMs)** on hierarchical insect classification tasks and discovery accuracy. Models are tested on their ability to recognize **known** and **novel** species across multiple taxonomic levels (Order, Family, Genus, Species), using real-world field-collected data.

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
- `Known/` (input images)
- `metadata_known.csv` (ground truth)
- `Novel/` (input images)
- `metadata_novel.csv` (ground truth)
## 🧠 Inference & Evaluation Pipeline

### 📦 Requirements
- Python 3.8+
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `TOGETHER_API_KEY`, `GROK_API_KEY`
- Required libraries: `pandas`, `openai`, `anthropic`, `google.generativeai`, etc.


## 🚀 Run the Evaluation Script 
Here we show an example of running the script on Novel split

```python
import os
from vlm_inference.vlm_analysis import get_image_paths, parse_responses, evaluate_classification
from vlm_inference.vlm_calls import (
    run_gpt_inference, run_claude_inference, run_gemini_inference,
    run_together_inference, run_grok_inference
)

image_dir = "Novel"
ground_truth = "metadata_novel.csv"
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

## License

This dataset is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

You are free to:
- **Share** — copy and redistribute the material in any medium or format.
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

Please cite our work when using this dataset in your research.

## Citation
If you use TerraIncognita in your research, please cite:
```bibtex
@inproceedings{chiranjeevi2025terraincognita,
  title={TerraIncognita: A Dynamic Benchmark for Species Discovery Using Frontier Models},
  author={Chiranjeevi, Shivani and Zaremehrjerdi, Hossein and Deng, Zi K. and Jubery, Talukder Z. and Grele, Ari and Singh, Arti and Singh, Asheesh K. and Sarkar, Soumik and Merchant, Nirav and Greeney, Harold F. and Ganapathysubramanian, Baskar and Hegde, Chinmay},
  booktitle={},
  year={2025}
}
```
---
