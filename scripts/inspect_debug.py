#!/usr/bin/env python3
"""Utility to sanity-check debug artifacts dumped by scripts/run_inference.py."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

CLASS_NAMES = {
    0: "pedestrian_crossing",
    1: "lane_divider",
    2: "lane_boundary",
}

EXPECTED_SUFFIXES = {
    "input": "input_cam0.png",
    "segmentation": "segmentation.png",
    "confidence": "confidence.png",
    "embedding": "embedding_norm.png",
    "direction": "direction.png",
    "vectorized": "vectorized.png",
    "tensor": "tensors.pt",
}


def analyze_tensor_file(tensor_path: Path) -> Dict[str, object]:
    payload = torch.load(tensor_path, map_location="cpu")
    segmentation: torch.Tensor = payload["segmentation"]
    embedding: torch.Tensor = payload["embedding"]
    direction: torch.Tensor = payload["direction"]

    probs = torch.softmax(segmentation, dim=0)
    pred_classes = torch.argmax(probs, dim=0)
    class_counts = Counter(pred_classes.cpu().numpy().ravel().tolist())
    total_pixels = float(pred_classes.numel())
    class_distribution = {
        CLASS_NAMES.get(cls, f"class_{cls}"): round(count / total_pixels, 6)
        for cls, count in sorted(class_counts.items())
    }

    confidence_map = torch.max(probs, dim=0)[0]
    confidence_stats = summarize_tensor(confidence_map)

    embedding_norm = torch.norm(embedding, dim=0)
    embedding_stats = summarize_tensor(embedding_norm)

    direction_probs = torch.softmax(direction, dim=0)
    direction_pred = torch.argmax(direction_probs, dim=0)
    direction_distribution = Counter(direction_pred.cpu().numpy().ravel().tolist())
    top_direction = int(direction_distribution.most_common(1)[0][0]) if direction_distribution else -1

    return {
        "seg_shape": tuple(segmentation.shape),
        "embedding_shape": tuple(embedding.shape),
        "direction_shape": tuple(direction.shape),
        "class_distribution": class_distribution,
        "avg_confidence": confidence_stats["mean"],
        "confidence_min": confidence_stats["min"],
        "confidence_max": confidence_stats["max"],
        "embedding_norm_mean": embedding_stats["mean"],
        "embedding_norm_max": embedding_stats["max"],
        "top_direction_bin": top_direction,
    }


def summarize_tensor(tensor: torch.Tensor) -> Dict[str, float]:
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def collect_rows(debug_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    tensor_files = sorted(debug_dir.glob("*_tensors.pt"))
    for tensor_path in tensor_files:
        token = tensor_path.stem.replace("_tensors", "")
        stats = analyze_tensor_file(tensor_path)
        file_status = {}
        for key, suffix in EXPECTED_SUFFIXES.items():
            file_status[f"has_{key}"] = (debug_dir / f"{token}_{suffix}").exists()
        rows.append({
            "token": token,
            **stats,
            **file_status,
        })
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames: List[str] = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_copy = row.copy()
            row_copy["class_distribution"] = json.dumps(row_copy["class_distribution"], sort_keys=True)
            writer.writerow(row_copy)


def write_json(rows: List[Dict[str, object]], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect HDMapNet debug_dir dumps")
    parser.add_argument("debug_dir", help="Directory passed via --debug_dir during inference")
    parser.add_argument("--summary_csv", default="debug_summary.csv",
                        help="Filename for the CSV summary (written inside debug_dir by default)")
    parser.add_argument("--summary_json", default="debug_summary.json",
                        help="Filename for the JSON summary (written inside debug_dir by default)")
    parser.add_argument("--tokens", nargs="*", default=None,
                        help="Optional subset of tokens to analyze (defaults to all *_tensors.pt files)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    if not debug_dir.exists():
        raise FileNotFoundError(f"debug_dir not found: {debug_dir}")

    rows = collect_rows(debug_dir)
    if args.tokens:
        lookup = set(args.tokens)
        rows = [row for row in rows if row["token"] in lookup]

    if not rows:
        print("No tensor dumps found to analyze.")
        return

    csv_path = debug_dir / args.summary_csv
    json_path = debug_dir / args.summary_json
    write_csv(rows, csv_path)
    write_json(rows, json_path)
    print(f"Wrote {len(rows)} summaries -> {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
