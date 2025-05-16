import os
import json
import pandas as pd
from sklearn.metrics import classification_report

# ---------- UTILITY TO GATHER IMAGES ----------
def get_image_paths(root_folder):
    image_paths = []
    for folder in os.listdir(root_folder):
        if folder.startswith("."):
            continue
        for file in os.listdir(os.path.join(root_folder, folder)):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root_folder, folder, file))
    return image_paths

# ---------- PARSE MODEL RESPONSES ----------
def parse_responses(results):
    parsed_results = []
    for item in results:
        row = {
            "Image Path": item["image_path"],
            "Order": "Unknown",
            "Family": "Unknown",
            "Genus": "Unknown",
            "Species": "Unknown"
        }
        try:
            lines = item["response"].splitlines()
            for line in lines:
                line = line.strip()
                if line.lower().startswith("order:"):
                    row["Order"] = line.split(":", 1)[1].strip()
                elif line.lower().startswith("family:"):
                    row["Family"] = line.split(":", 1)[1].strip()
                elif line.lower().startswith("genus:"):
                    row["Genus"] = line.split(":", 1)[1].strip()
                elif line.lower().startswith("species:"):
                    row["Species"] = line.split(":", 1)[1].strip()
            parsed_results.append(row)
        except:
            continue
    return pd.DataFrame(parsed_results)

# ---------- EVALUATION METRICS ----------
def evaluate_classification(pred_df, ground_truth_xlsx, output_xlsx=None):
    gt_df = pd.read_excel(ground_truth_xlsx)
    gt_map = gt_df.set_index("Folder")[["Order", "Family", "Genus", "Species"]].to_dict(orient="index")

    # Extract folder number from image path to match
    pred_df["Folder"] = pred_df["Image Path"].apply(lambda x: int(os.path.basename(os.path.dirname(x))))

    y_true, y_pred = [], []
    for _, row in pred_df.iterrows():
        folder = row["Folder"]
        if folder not in gt_map:
            continue
        y_true.append(gt_map[folder])
        y_pred.append({"Order": row["Order"], "Family": row["Family"], "Genus": row["Genus"], "Species": row["Species"]})

    # Compute flat F1 per taxonomic level
    levels = ["Order", "Family", "Genus", "Species"]
    report = {}
    for level in levels:
        true_labels = [yt[level] for yt in y_true]
        pred_labels = [yp[level] for yp in y_pred]
        report[level] = classification_report(true_labels, pred_labels, output_dict=True)

    if output_xlsx:
        pred_df.to_excel(output_xlsx, index=False)
    return report
