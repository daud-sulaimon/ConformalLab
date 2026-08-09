"""
One-time script to fetch ImageNet's 1000 class names from Hugging Face
and save them locally, so no experiment script needs to stream from
the network just to read this static, unchanging list.

Run once:
    python scripts/generate_imagenet_classnames.py
"""

import json
from pathlib import Path

from datasets import load_dataset

stream = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
class_names = stream.features["label"].names

output_path = Path("src/datasets/imagenet_class_names.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

print(f"Saved {len(class_names)} class names to {output_path}")