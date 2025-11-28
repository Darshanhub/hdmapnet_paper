#!/usr/bin/env python3
"""Convenience wrapper for running HDMapNet inference on Google Colab."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow importing run_inference + HDMapNet modules even when the script is invoked via
# "python hdmapnet_paper/scripts/run_colab.py" from outside the repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
THIRD_PARTY_DIR = REPO_ROOT / "third_party" / "HDMapNet"

for path in (SCRIPT_DIR, REPO_ROOT, THIRD_PARTY_DIR):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_inference  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal CLI for Colab: point to a dataset (nuScenes root or plain folder) and a checkpoint"
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to nuScenes-style root or to a folder containing RGB frames")
    parser.add_argument("--model_path", required=True, help="Path to the checkpoint (.pt) file to load")
    parser.add_argument("--output_dir", default="outputs/colab",
                        help="Directory where JSON/PNG outputs will be written")
    parser.add_argument("--image_dir", default=None,
                        help="Optional explicit path for plain image folders (falls back to --dataset)")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to process (<=0 = all)")
    parser.add_argument("--version", default="v1.0-mini", choices=["v1.0-mini", "v1.0-trainval"],
                        help="nuScenes split when using an official dataset")
    parser.add_argument("--model", default="HDMapNet_cam",
                        choices=["HDMapNet_cam", "HDMapNet_lidar", "HDMapNet_fusion", "lift_splat"],
                        help="Model variant to instantiate")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size (Colab RAM friendly)")
    parser.add_argument("--nworkers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU execution even if GPU is available")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only validate checkpoint + dependencies without touching the dataset")
    parser.add_argument("--skip_model_load", action="store_true",
                        help="Skip loading the torch model (just validate dataset + config)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Additional flags forwarded to scripts/run_inference.py (optional)")
    return parser


def assemble_runner_args(colab_args: argparse.Namespace) -> argparse.Namespace:
    base_parser = run_inference.build_arg_parser()
    args = base_parser.parse_args([])

    args.dataroot = colab_args.dataset
    args.modelf = colab_args.model_path
    args.output_dir = colab_args.output_dir
    args.limit = colab_args.limit
    args.version = colab_args.version
    args.model = colab_args.model
    args.bsz = colab_args.bsz
    args.nworkers = colab_args.nworkers
    args.force_cpu = colab_args.force_cpu
    args.dry_run = colab_args.dry_run
    args.skip_model_load = colab_args.skip_model_load

    if colab_args.image_dir:
        args.image_dir = colab_args.image_dir

    if colab_args.extra:
        extra_args = base_parser.parse_args(colab_args.extra)
        for key, value in vars(extra_args).items():
            setattr(args, key, value)

    return args


def main() -> None:
    parser = build_parser()
    colab_args = parser.parse_args()

    dataset_path = Path(colab_args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    model_path = Path(colab_args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if colab_args.image_dir:
        image_dir_path = Path(colab_args.image_dir)
        if not image_dir_path.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir_path}")

    run_args = assemble_runner_args(colab_args)
    run_inference.main(run_args)


if __name__ == "__main__":
    main()
