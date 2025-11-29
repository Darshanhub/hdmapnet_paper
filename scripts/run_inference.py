#!/usr/bin/env python3
"""High-level inference helper that produces both JSON submissions and layman-friendly visualizations."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib  # type: ignore

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
import matplotlib.patches as patches  # type: ignore  # noqa: E402
import matplotlib.colors as mcolors  # type: ignore  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402

# Third-party HDMapNet utilities still reference deprecated numpy aliases (e.g., np.int).
# Guard here so any downstream import continues to work on modern NumPy releases (>=1.26).
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
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
from data.image import normalize_img, img_transform, denormalize_img  # type: ignore  # noqa: E402
import model as hdmapnet_model  # type: ignore  # noqa: E402
from postprocess.vectorize import vectorize  # type: ignore  # noqa: E402

SEGMENTATION_CLASS_NAMES = {
    0: "background",
    1: "pedestrian_crossing",
    2: "lane_divider",
    3: "lane_boundary",
}
SEGMENTATION_COLORS = {
    0: "tab:gray",
    1: "tab:orange",
    2: "tab:blue",
    3: "tab:green",
}
LINE_CLASS_NAMES = {
    0: "pedestrian_crossing",
    1: "lane_divider",
    2: "lane_boundary",
}
LINE_CLASS_COLORS = {
    0: "tab:orange",
    1: "tab:blue",
    2: "tab:green",
}

CLASS_CHANNEL_TO_COLOR: Dict[int, Tuple[int, int, int]] = {}
for class_idx, color_name in SEGMENTATION_COLORS.items():
    r, g, b = (int(c * 255) for c in mcolors.to_rgb(color_name))
    CLASS_CHANNEL_TO_COLOR[class_idx] = (r, g, b)

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
        self.synthetic_intrins = self._build_synthetic_intrinsics()
        self.synthetic_rots, self.synthetic_trans = self._build_synthetic_extrinsics()

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

    def _build_synthetic_intrinsics(self) -> torch.Tensor:
        width, height = self.resize_dims
        focal = max(width, height) * 1.2
        cx = width / 2.0
        cy = height / 2.0
        intrinsic = torch.tensor(
            [
                [focal, 0.0, cx],
                [0.0, focal, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return intrinsic.unsqueeze(0).repeat(len(CAMS), 1, 1)

    def _build_synthetic_extrinsics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        yaw_degs = [60.0, 0.0, -60.0, 120.0, 180.0, -120.0]
        radius = 1.8
        height = 1.5
        rots: List[torch.Tensor] = []
        trans: List[torch.Tensor] = []
        for idx, yaw in enumerate(yaw_degs):
            if idx >= len(CAMS):
                break
            rot = self._rotation_matrix_from_yaw(yaw)
            rots.append(rot)
            yaw_rad = math.radians(yaw)
            trans.append(
                torch.tensor(
                    [radius * math.cos(yaw_rad), radius * math.sin(yaw_rad), height],
                    dtype=torch.float32,
                )
            )
        while len(rots) < len(CAMS):
            rots.append(torch.eye(3, dtype=torch.float32))
            trans.append(torch.tensor([0.0, 0.0, height], dtype=torch.float32))
        return torch.stack(rots), torch.stack(trans)

    @staticmethod
    def _rotation_matrix_from_yaw(yaw_deg: float) -> torch.Tensor:
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        rot = torch.eye(3, dtype=torch.float32)
        rot[0, 0] = c
        rot[0, 1] = -s
        rot[1, 0] = s
        rot[1, 1] = c
        return rot

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

        trans = self.synthetic_trans.clone()
        rots = self.synthetic_rots.clone()
        intrins = self.synthetic_intrins.clone()
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
        if coord.size == 0:
            continue
        color = LINE_CLASS_COLORS.get(ltype, "tab:red")
        axes[0].plot(coord[:, 0], coord[:, 1], color=color, linewidth=2)

        min_x, min_y = coord.min(axis=0)
        max_x, max_y = coord.max(axis=0)
        rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                 linewidth=2, edgecolor=color, facecolor="none")
        axes[1].add_patch(rect)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        label = LINE_CLASS_NAMES.get(ltype, f"class_{ltype}")
        axes[1].text(cx, cy, label, color=color, fontsize=8, ha="center", va="center",
                     bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    fig.suptitle(f"Sample {sample_id}")
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)


def _tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    return denormalize_img(tensor.detach().cpu())


def _render_segmentation(segmentation: torch.Tensor) -> Image.Image:
    probs = torch.softmax(segmentation.detach().cpu(), dim=0)
    pred = torch.argmax(probs, dim=0).numpy()
    color_img = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for channel, rgb in CLASS_CHANNEL_TO_COLOR.items():
        mask = pred == channel
        if np.any(mask):
            color_img[mask] = rgb
    return Image.fromarray(color_img, mode="RGB")


def _render_confidence(segmentation: torch.Tensor) -> Image.Image:
    probs = torch.softmax(segmentation.detach().cpu(), dim=0)
    conf = torch.max(probs, dim=0)[0].numpy()
    conf_img = (np.clip(conf, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(conf_img, mode="L")


def _render_embedding_norm(embedding: torch.Tensor) -> Image.Image:
    emb = torch.norm(embedding.detach().cpu(), dim=0)
    emb = emb / (emb.max() + 1e-8)
    emb_img = (emb.numpy() * 255).astype(np.uint8)
    return Image.fromarray(emb_img, mode="L")


def _render_direction(direction: torch.Tensor, angle_class: int) -> Image.Image:
    direction = torch.softmax(direction.detach().cpu(), dim=0)
    pred = torch.argmax(direction, dim=0).numpy()
    denom = max(angle_class - 1, 1)
    dir_img = (pred.astype(np.float32) / denom * 255).astype(np.uint8)
    return Image.fromarray(dir_img, mode="L")


def save_debug_artifacts(token: str,
                         cams: torch.Tensor,
                         segmentation: torch.Tensor,
                         embedding: torch.Tensor,
                         direction: torch.Tensor,
                         vector_image_path: Path,
                         debug_dir: Path,
                         angle_class: int) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    cam0 = cams[0]
    input_img_path = debug_dir / f"{token}_input_cam0.png"
    _tensor_to_pil_image(cam0).save(input_img_path)

    seg_map_path = debug_dir / f"{token}_segmentation.png"
    _render_segmentation(segmentation).save(seg_map_path)

    conf_map_path = debug_dir / f"{token}_confidence.png"
    _render_confidence(segmentation).save(conf_map_path)

    embed_map_path = debug_dir / f"{token}_embedding_norm.png"
    _render_embedding_norm(embedding).save(embed_map_path)

    direction_map_path = debug_dir / f"{token}_direction.png"
    _render_direction(direction, angle_class).save(direction_map_path)

    if vector_image_path.exists():
        shutil.copy(vector_image_path, debug_dir / f"{token}_vectorized.png")

    tensor_dump = debug_dir / f"{token}_tensors.pt"
    torch.save({
        "token": token,
        "segmentation": segmentation.detach().cpu(),
        "embedding": embedding.detach().cpu(),
        "direction": direction.detach().cpu(),
    }, tensor_dump)


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
    parser.add_argument("--debug_dir", type=str, default=None,
                        help="Optional directory to dump debug inputs/intermediates (defaults to output_dir/debugging)")
    parser.add_argument("--debug_limit", type=int, default=0,
                        help="How many samples to capture in debug_dir (<=0 disables)")
    parser.add_argument("--force-cpu", action="store_true", help="Run everything on CPU even if CUDA is available")
    parser.add_argument("--dry-run", action="store_true", help="Only load the checkpoint to verify dependencies")
    parser.add_argument("--skip_model_load", action="store_true",
                        help="Skip constructing/loading the torch model; useful for local validation without heavy memory usage")

    # data + model hyper-parameters (mirror train.py defaults)
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--angle_class", type=int, default=36)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--instance_seg", dest="instance_seg", action="store_true", default=True,
                        help="Enable instance embedding head (default: enabled)")
    parser.add_argument("--no-instance-seg", dest="instance_seg", action="store_false",
                        help="Disable instance embedding head")
    parser.add_argument("--direction_pred", dest="direction_pred", action="store_true", default=True,
                        help="Enable direction prediction head (default: enabled)")
    parser.add_argument("--no-direction-pred", dest="direction_pred", action="store_false",
                        help="Disable direction prediction head")
    return parser


def main(args: argparse.Namespace) -> None:
    checkpoint = Path(args.modelf)
    if not checkpoint.exists():
        if args.skip_model_load:
            print(f"Warning: checkpoint not found at {checkpoint} (skip_model_load is enabled, continuing anyway)")
        else:
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

    data_aug_conf = {
        "final_dim": tuple(args.image_size),
    }

    if args.skip_model_load:
        val_loader, data_source_desc = build_validation_loader(args, data_conf)

        dataset_size = len(cast(Any, val_loader.dataset))
        print(f"skip_model_load=True → validated {dataset_size} samples from {data_source_desc} and exiting before model init.")

        return

    get_model_params = inspect.signature(get_model).parameters
    model_kwargs = {
        "instance_seg": args.instance_seg,
        "embedded_dim": args.embedding_dim,
        "direction_pred": args.direction_pred,
        "angle_class": args.angle_class,
    }
    if "data_aug_conf" in get_model_params:
        model_kwargs["data_aug_conf"] = data_aug_conf
    else:
        if args.model == "lift_splat":
            print("Warning: get_model() in third_party/HDMapNet/model/__init__.py is missing data_aug_conf support. "
                  "Please pull the latest changes. Falling back to legacy signature without augmentation metadata.")
    model = get_model(args.model, data_conf, **model_kwargs)

    state = torch.load(checkpoint, map_location=device)

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    if args.dry_run:
        print(f"Loaded {checkpoint} on {device} (dry run mode, skipping dataset/inference).")
        return

    val_loader, data_source_desc = build_validation_loader(args, data_conf)
    dataset_size = len(cast(Any, val_loader.dataset))  # torch Dataset implements __len__ at runtime
    print(f"Using {dataset_size} samples from {data_source_desc}")

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

    debug_dir: Optional[Path]
    if args.debug_limit > 0:
        debug_dir = Path(args.debug_dir) if args.debug_dir else Path(args.output_dir) / "debugging"
        debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        debug_dir = None
    debug_limit = max(0, args.debug_limit)
    debug_saved = 0

    summary_rows: List[Dict[str, object]] = []
    sample_cap = None if args.limit <= 0 else args.limit
    processed = 0

    progress = tqdm.tqdm(val_loader, desc="Running inference", dynamic_ncols=True)
    effective_batch_size = val_loader.batch_size or args.bsz
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
            if embedding is None:
                embedding = torch.zeros(
                    (segmentation.shape[0], 1, *segmentation.shape[-2:]),
                    device=segmentation.device,
                    dtype=segmentation.dtype,
                )
            if direction is None:
                direction = torch.zeros(
                    (segmentation.shape[0], args.angle_class, *segmentation.shape[-2:]),
                    device=segmentation.device,
                    dtype=segmentation.dtype,
                )
            for sample_in_batch in range(segmentation.shape[0]):
                coords, confidences, line_types = vectorize(segmentation[sample_in_batch],
                                                            embedding[sample_in_batch],
                                                            direction[sample_in_batch],
                                                            args.angle_class)
                canvas_shape = segmentation.shape[-2:]
                sample_idx = batch_idx * effective_batch_size + sample_in_batch
                dataset_obj = cast(Any, val_loader.dataset)  # may be custom dataset with samples attr
                rec = dataset_obj.samples[sample_idx]
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

                if debug_dir is not None and debug_saved < debug_limit:
                    save_debug_artifacts(
                        token=token,
                        cams=imgs[sample_in_batch],
                        segmentation=segmentation[sample_in_batch],
                        embedding=embedding[sample_in_batch],
                        direction=direction[sample_in_batch],
                        vector_image_path=image_path,
                        debug_dir=debug_dir,
                        angle_class=args.angle_class,
                    )
                    debug_saved += 1

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
