#!/usr/bin/env python3
"""High-level inference helper that produces both JSON submissions and layman-friendly visualizations."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib  # type: ignore

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
import matplotlib.patches as patches  # type: ignore  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402
import torch  # type: ignore  # noqa: E402
import tqdm  # type: ignore  # noqa: E402
from PIL import Image  # type: ignore  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # type: ignore  # noqa: E402

# Make third_party/HDMapNet importable without modifying PYTHONPATH globally.
REPO_ROOT = Path(__file__).resolve().parents[1]
HDMAPNET_DIR = REPO_ROOT / "third_party" / "HDMapNet"
if str(HDMAPNET_DIR) not in sys.path:
    sys.path.insert(0, str(HDMAPNET_DIR))

from data.dataset import semantic_dataset  # type: ignore  # noqa: E402
from data.const import CAMS, IMG_ORIGIN_H, IMG_ORIGIN_W, NUM_CLASSES  # type: ignore  # noqa: E402
from data.image import normalize_img, img_transform  # type: ignore  # noqa: E402
from model import get_model  # type: ignore  # noqa: E402
from postprocess.vectorize import vectorize  # type: ignore  # noqa: E402

CLASS_NAMES = {
    0: "pedestrian_crossing",
    1: "lane_divider",
    2: "lane_boundary",
}
CLASS_COLORS = {
    0: "tab:orange",
    1: "tab:blue",
    2: "tab:green",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ImageFolderSemanticDataset(Dataset):
    """Minimal shim that mimics the nuScenes dataloader using plain RGB frames."""

    def __init__(self, image_dir: str, data_conf: Dict[str, Any]):
        super().__init__()
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths = sorted(
            [p for p in self.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images with extensions {sorted(IMAGE_EXTENSIONS)} found under {self.image_dir}")

        self.data_conf = data_conf
        self.canvas_size = self._compute_canvas_size()
        self.resize, self.resize_dims = self._compute_resize()

        self.samples = [
            {
                "token": f"image_{idx:05d}_{path.stem}",
                "path": path,
            }
            for idx, path in enumerate(self.image_paths)
        ]

    def _compute_canvas_size(self) -> Tuple[int, int]:
        patch_h = self.data_conf["ybound"][1] - self.data_conf["ybound"][0]
        patch_w = self.data_conf["xbound"][1] - self.data_conf["xbound"][0]
        canvas_h = int(patch_h / self.data_conf["ybound"][2])
        canvas_w = int(patch_w / self.data_conf["xbound"][2])
        return canvas_h, canvas_w

    def _compute_resize(self) -> Tuple[Tuple[float, float], Tuple[int, int]]:
        fH, fW = self.data_conf["image_size"]
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)
        resize_dims = (fW, fH)
        return resize, resize_dims

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_tensor(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = Image.open(path).convert("RGB")
        img, post_rot, post_tran = img_transform(img, self.resize, self.resize_dims)
        img = normalize_img(img)
        return img, post_rot, post_tran

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img, post_rot, post_tran = self._load_image_tensor(rec["path"])

        num_cams = len(CAMS)
        imgs = img.unsqueeze(0).repeat(num_cams, 1, 1, 1)
        post_rots = post_rot.unsqueeze(0).repeat(num_cams, 1, 1)
        post_trans = post_tran.unsqueeze(0).repeat(num_cams, 1)

        trans = torch.zeros(num_cams, 3, dtype=torch.float32)
        rots = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(num_cams, 1, 1)
        intrins = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(num_cams, 1, 1)
        lidar_data = torch.zeros(81920, 5, dtype=torch.float32)
        lidar_mask = torch.zeros(81920, dtype=torch.float32)
        car_trans = torch.zeros(3, dtype=torch.float32)
        yaw_pitch_roll = torch.zeros(3, dtype=torch.float32)

        return (
            imgs,
            trans,
            rots,
            intrins,
            post_trans,
            post_rots,
            lidar_data,
            lidar_mask,
            car_trans,
            yaw_pitch_roll,
        )


def _has_images(path: Path) -> bool:
    for ext in IMAGE_EXTENSIONS:
        if next(path.rglob(f"*{ext}"), None) is not None:
            return True
    return False


def resolve_image_directory(args: argparse.Namespace) -> Optional[Path]:
    if args.image_dir:
        return Path(args.image_dir)

    dataroot = Path(args.dataroot)
    if not dataroot.exists():
        return None

    # nuScenes roots always contain metadata folders such as v1.0-mini and samples/
    if (dataroot / args.version).exists() or (dataroot / "samples").exists():
        return None

    return dataroot if _has_images(dataroot) else None


def build_validation_loader(args: argparse.Namespace,
                            data_conf: Dict[str, Any]) -> Tuple[DataLoader, str]:
    image_dir = resolve_image_directory(args)
    if image_dir is not None:
        dataset = ImageFolderSemanticDataset(str(image_dir), data_conf)
        loader = DataLoader(dataset, batch_size=args.bsz, shuffle=False, num_workers=args.nworkers)
        desc = f"image folder at {image_dir}"
    else:
        _, loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers)
        desc = f"nuScenes {args.version} at {args.dataroot}"
    return loader, desc


def gen_dx_bx(xbound: Sequence[float], ybound: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    dx = np.array([row[2] for row in [xbound, ybound]], dtype=np.float32)
    bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound]], dtype=np.float32)
    return dx, bx


def grid_to_world(points: np.ndarray, dx: np.ndarray, bx: np.ndarray) -> np.ndarray:
    """Convert pixel coordinates (W, H) into real-world meters using HDMapNet bounds."""
    return points * dx + bx


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "json": root / "json",
        "images": root / "images",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def draw_prediction(sample_id: str,
                    coords: List[np.ndarray],
                    line_types: List[int],
                    canvas_shape: Tuple[int, int],
                    output_path: Path) -> None:
    h, w = canvas_shape
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for ax in axes:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Vectorized map" if ax is axes[0] else "Bounding boxes")

    for coord, ltype in zip(coords, line_types):
        color = CLASS_COLORS.get(ltype, "tab:red")
        axes[0].plot(coord[:, 0], coord[:, 1], color=color, linewidth=2)

        min_x, min_y = coord.min(axis=0)
        max_x, max_y = coord.max(axis=0)
        rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                 linewidth=2, edgecolor=color, facecolor="none")
        axes[1].add_patch(rect)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        axes[1].text(cx, cy, CLASS_NAMES.get(ltype, f"class_{ltype}"),
                     color=color, fontsize=8, ha="center", va="center",
                     bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    fig.suptitle(f"Sample {sample_id}")
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)


def save_json(sample_token: str,
              vectors: List[Dict[str, object]],
              submission: Dict[str, Any],
              per_sample_path: Path) -> None:
    per_sample = {
        "meta": {
            "token": sample_token,
            "vector": True,
        },
        "results": vectors,
    }
    per_sample_path.write_text(json.dumps(per_sample, indent=2))
    submission["results"][sample_token] = vectors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HDMapNet inference runner with visuals")
    parser.add_argument("--dataroot", type=str, default="dataset/nuScenes/",
                        help="Path to nuScenes-style dataset (auto-detects plain folders without metadata)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Optional explicit path to a folder that just contains camera frames")
    parser.add_argument("--version", type=str, default="v1.0-mini",
                        choices=["v1.0-mini", "v1.0-trainval"],
                        help="nuScenes split to evaluate")
    parser.add_argument("--modelf", type=str, default="artifacts/checkpoints/model.pt",
                        help="Checkpoint to load")
    parser.add_argument("--model", type=str, default="HDMapNet_cam",
                        choices=["HDMapNet_cam", "HDMapNet_lidar", "HDMapNet_fusion", "lift_splat"],
                        help="Backbone to instantiate")
    parser.add_argument("--bsz", type=int, default=2, help="Batch size for the dataloader")
    parser.add_argument("--nworkers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to write json + images")
    parser.add_argument("--limit", type=int, default=10, help="Number of validation samples to process (<=0 = all)")
    parser.add_argument("--force-cpu", action="store_true", help="Run everything on CPU even if CUDA is available")
    parser.add_argument("--dry-run", action="store_true", help="Only load the checkpoint to verify dependencies")

    # data + model hyper-parameters (mirror train.py defaults)
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--angle_class", type=int, default=36)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--instance_seg", action="store_true", help="Enable instance embedding head")
    parser.add_argument("--direction_pred", action="store_true", help="Enable direction prediction head")
    return parser


def main(args: argparse.Namespace) -> None:
    checkpoint = Path(args.modelf)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Drop your model.pt under artifacts/checkpoints/")

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")

    data_conf = {
        "num_channels": NUM_CLASSES + 1,
        "image_size": args.image_size,
        "xbound": args.xbound,
        "ybound": args.ybound,
        "zbound": args.zbound,
        "dbound": args.dbound,
        "thickness": args.thickness,
        "angle_class": args.angle_class,
    }

    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    if args.dry_run:
        print(f"Loaded {checkpoint} on {device} (dry run mode, skipping dataset/inference).")
        return

    val_loader, data_source_desc = build_validation_loader(args, data_conf)
    print(f"Using {len(val_loader.dataset)} samples from {data_source_desc}")

    dx, bx = gen_dx_bx(args.xbound, args.ybound)
    output_dirs = ensure_dirs(Path(args.output_dir))
    submission: Dict[str, Any] = {
        "meta": {
            "use_camera": args.model in {"HDMapNet_cam", "HDMapNet_fusion", "lift_splat"},
            "use_lidar": args.model in {"HDMapNet_lidar", "HDMapNet_fusion"},
            "use_radar": False,
            "use_external": False,
            "vector": True,
        },
        "results": {},
    }

    summary_rows: List[Dict[str, object]] = []
    sample_cap = None if args.limit <= 0 else args.limit
    processed = 0

    progress = tqdm.tqdm(val_loader, desc="Running inference", dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            (imgs, trans, rots, intrins, post_trans, post_rots,
             lidar_data, lidar_mask, car_trans, yaw_pitch_roll,
             *_rest) = batch

            imgs = imgs.to(device)
            trans = trans.to(device)
            rots = rots.to(device)
            intrins = intrins.to(device)
            post_trans = post_trans.to(device)
            post_rots = post_rots.to(device)
            lidar_data = lidar_data.to(device)
            lidar_mask = lidar_mask.to(device)
            car_trans = car_trans.to(device)
            yaw_pitch_roll = yaw_pitch_roll.to(device)

            segmentation, embedding, direction = model(imgs, trans, rots, intrins,
                                                        post_trans, post_rots, lidar_data,
                                                        lidar_mask, car_trans, yaw_pitch_roll)

            for sample_in_batch in range(segmentation.shape[0]):
                coords, confidences, line_types = vectorize(segmentation[sample_in_batch],
                                                            embedding[sample_in_batch],
                                                            direction[sample_in_batch],
                                                            args.angle_class)
                canvas_shape = segmentation.shape[-2:]
                sample_idx = batch_idx * val_loader.batch_size + sample_in_batch
                rec = val_loader.dataset.samples[sample_idx]
                token = rec["token"]

                vectors = []
                for coord, confidence, line_type in zip(coords, confidences, line_types):
                    world_pts = grid_to_world(coord.astype(np.float32), dx, bx)
                    vectors.append({
                        "pts": world_pts.tolist(),
                        "pts_num": int(len(world_pts)),
                        "type": int(line_type),
                        "confidence_level": float(confidence),
                    })

                per_sample_json = output_dirs["json"] / f"{token}.json"
                save_json(token, vectors, submission, per_sample_json)

                image_path = output_dirs["images"] / f"{token}.png"
                draw_prediction(token, coords, line_types, canvas_shape, image_path)

                summary_row = {
                    "token": token,
                    "pedestrian_crossings": sum(1 for lt in line_types if lt == 0),
                    "lane_dividers": sum(1 for lt in line_types if lt == 1),
                    "lane_boundaries": sum(1 for lt in line_types if lt == 2),
                    "avg_confidence": float(np.mean(confidences) if confidences else 0.0),
                    "image_path": str(image_path),
                    "json_path": str(per_sample_json),
                }
                summary_rows.append(summary_row)

                processed += 1
                progress.set_postfix(samples=processed)
                if sample_cap is not None and processed >= sample_cap:
                    break
            if sample_cap is not None and processed >= sample_cap:
                break

    output_dirs["root"].mkdir(exist_ok=True, parents=True)
    submission_path = output_dirs["root"] / "submission.json"
    submission_path.write_text(json.dumps(submission, indent=2))

    summary_csv = output_dirs["root"] / "summary.csv"
    if summary_rows:
        header = summary_rows[0].keys()
        lines = [",".join(header)]
        for row in summary_rows:
            lines.append(",".join(str(row[h]) for h in header))
        summary_csv.write_text("\n".join(lines))

    print(f"Saved {processed} samples -> {submission_path} and {summary_csv}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
