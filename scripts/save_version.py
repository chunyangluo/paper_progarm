import os
import json
import time
import torch

PROJECT_ROOT = r"c:\Users\chuny\Desktop\paper_progarm"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

version = f"v2.0.{int(time.time()) % 10000}"
version_info = {
    "version": version,
    "model_type": "bert_textcnn",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "training_config": {
        "seed": 42, "max_length": 128, "batch_size": 16,
        "learning_rate": 2e-05, "weight_decay": 0.01, "epochs": 9,
        "warmup_ratio": 0.1, "patience": 3, "grad_accum_steps": 4,
        "dataset": "dataset_20260411_chifraud.csv"
    },
    "training_info": {
        "best_f1": 0.9934, "best_epoch": 6,
        "total_time_minutes": 16.6,
        "trainable_params": 28794562, "total_params": 102710722
    },
    "test_metrics": {
        "accuracy": 0.9937, "precision": 0.9939, "recall": 0.9936,
        "f1_score": 0.9937, "auc_score": 0.9985,
        "confusion_matrix": [[3086, 19], [20, 3085]],
        "inference_speed": 262.3
    },
    "is_active": True, "is_deployed": True,
}

report_path = os.path.join(OUTPUT_DIR, "training_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(version_info, f, ensure_ascii=False, indent=2)
print(f"Training report saved to {report_path}")

version_path = os.path.join(MODEL_DIR, "model_versions.json")
versions = {}
if os.path.exists(version_path):
    with open(version_path, "r", encoding="utf-8") as f:
        versions = json.load(f)

for key in versions:
    if isinstance(versions[key], dict):
        versions[key]["is_active"] = False
        versions[key]["is_deployed"] = False

version_key = f"{version}_{version_info['model_type']}"
versions[version_key] = version_info

with open(version_path, "w", encoding="utf-8") as f:
    json.dump(versions, f, ensure_ascii=False, indent=2)
print(f"Version {version} saved. Total versions: {len(versions)}")

model_path = os.path.join(MODEL_DIR, "bert_textcnn_best.pth")
sd = torch.load(model_path, map_location="cpu", weights_only=True)
fc_w = sd.get("fc.weight")
fc_b = sd.get("fc.bias")
print(f"\nModel weight verification:")
print(f"  fc.weight: shape={fc_w.shape}, mean={fc_w.mean().item():.4f}, std={fc_w.std().item():.4f}")
print(f"  fc.bias: {[round(b, 4) for b in fc_b.tolist()]}")
print(f"  File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
