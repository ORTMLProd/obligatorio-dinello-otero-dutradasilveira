"""Post-training static quantization (int8) of the clip model backbone, with measured impact.

Elective (model optimization, 2nd sub-technique on the image model): we quantize the frozen/
fine-tuned ResNet18 backbone to int8 via FX graph-mode PTQ (qnnpack) and MEASURE the impact on
(a) inference latency, (b) model size, and (c) macro-F1 — the consigna requires evaluating the
impact, not assuming it.

Honest finding (see report/bitacora.md): on Apple Silicon (ARM/qnnpack) int8 is *slower* than
FP32 because qnnpack lacks optimised int8 conv kernels there; the win is model size (~4x). On the
x86 serving host (fbgemm) the latency picture differs — impact is hardware/backend dependent.
The quantized model is therefore reported as an experiment, not shipped by default.

Usage:
    uv run python -m src.models.quantize --config ../configs/train_clips.yaml
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader

from src.data.clips_dataset import ClipsDataset
from src.models.clip_config import DEFAULT_CLIP_CONFIG_PATH, ClipTrainConfig
from src.models.clip_export import load_clip_bundle
from src.models.clip_model import build_transforms
from src.models.evaluate import compute_metrics


def _loader(manifest: pd.DataFrame, cfg: ClipTrainConfig, classes: list[str], processed: Path):
    transform = build_transforms(False, cfg.frame_size, cfg.normalize.mean, cfg.normalize.std)
    ds = ClipsDataset(manifest, processed, classes, transform)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)


def quantize_backbone(model, frame_size: int, calib_clips: list[torch.Tensor]):
    """Static PTQ (int8, qnnpack) of the backbone via FX, calibrated on real frames."""
    torch.backends.quantized.engine = "qnnpack"
    backbone = copy.deepcopy(model.backbone).eval()
    qmap = QConfigMapping().set_global(get_default_qconfig("qnnpack"))
    example = torch.randn(1, 3, frame_size, frame_size)
    prepared = prepare_fx(backbone, qmap, example_inputs=(example,))
    with torch.no_grad():
        for clip in calib_clips:  # clip: (K, 3, H, W) -> frames batch
            prepared(clip)
    return convert_fx(prepared)


def _clip_logits(backbone, head, clip: torch.Tensor) -> torch.Tensor:
    """Forward one clip (1, K, 3, H, W) through backbone+mean-pool+head, no grad."""
    with torch.no_grad():
        b, k = clip.shape[0], clip.shape[1]
        feats = backbone(clip.reshape(b * k, *clip.shape[2:])).reshape(b, k, -1)
        return head(feats.mean(dim=1))


def macro_f1(backbone, head, loader: DataLoader, classes: list[str]) -> dict:
    ys, probs = [], []
    for clips, labels in loader:
        logits = _clip_logits(backbone, head, clips)
        probs.append(torch.softmax(logits, dim=1).numpy())
        ys.append(labels.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(probs).astype(np.float64)
    return compute_metrics(y, p.argmax(axis=1), p, classes)


def bench_latency(backbone, head, k: int, size: int, n: int = 50) -> tuple[float, float]:
    """Single-clip CPU latency (ms): returns (p50, p95). Single-threaded for determinism."""
    torch.set_num_threads(1)
    clip = torch.rand(1, k, 3, size, size)
    for _ in range(5):
        _clip_logits(backbone, head, clip)  # warmup
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        _clip_logits(backbone, head, clip)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return median(ts), ts[min(int(0.95 * len(ts)), len(ts) - 1)]


def _state_dict_bytes(module) -> int:
    import io

    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes


def run(cfg: ClipTrainConfig) -> dict:
    import mlflow

    processed = cfg.paths.resolved("processed_dir")
    manifest = pd.read_parquet(cfg.paths.resolved("manifest"))
    classes = sorted(manifest["label"].unique().tolist())
    model, meta = load_clip_bundle(cfg.paths.resolved("model_dir"), device=torch.device("cpu"))
    model.eval()

    test = manifest[manifest["split"] == "test"]
    train = manifest[manifest["split"] == "train"].head(16)  # small calibration subset
    test_loader = _loader(test, cfg, classes, processed)
    calib_loader = _loader(train, cfg, classes, processed)
    calib_clips = [
        clips.reshape(-1, 3, cfg.frame_size, cfg.frame_size) for clips, _ in calib_loader
    ]

    qbackbone = quantize_backbone(model, cfg.frame_size, calib_clips)

    fp32 = {
        "macro_f1": macro_f1(model.backbone, model.head, test_loader, classes)["macro_f1"],
        "size_mb": _state_dict_bytes(model.backbone) / 1e6,
    }
    fp32["p50_ms"], fp32["p95_ms"] = bench_latency(
        model.backbone, model.head, meta.k, cfg.frame_size
    )
    int8 = {
        "macro_f1": macro_f1(qbackbone, model.head, test_loader, classes)["macro_f1"],
        "size_mb": _state_dict_bytes(qbackbone) / 1e6,
    }
    int8["p50_ms"], int8["p95_ms"] = bench_latency(qbackbone, model.head, meta.k, cfg.frame_size)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name="quantization-int8-static"):
        mlflow.log_params({"backend": "qnnpack", "method": "fx-static-ptq", "target": "backbone"})
        for tag, d in (("fp32", fp32), ("int8", int8)):
            mlflow.log_metrics({f"{tag}_{key}": val for key, val in d.items()})

    print("\n=== Quantization int8 (static PTQ, qnnpack) — impacto medido ===")
    print(f"{'':6s} {'macro-F1':>8s} {'size(MB)':>8s} {'p50(ms)':>8s} {'p95(ms)':>8s}")
    for tag, d in (("FP32", fp32), ("int8", int8)):
        f1, sz, p50, p95 = d["macro_f1"], d["size_mb"], d["p50_ms"], d["p95_ms"]
        print(f"{tag:6s} {f1:8.4f} {sz:8.2f} {p50:8.1f} {p95:8.1f}")
    print(
        f"\nsize: {fp32['size_mb'] / int8['size_mb']:.2f}x menor | "
        f"latencia p50: {fp32['p50_ms'] / int8['p50_ms']:.2f}x (qnnpack/ARM)"
    )
    return {"fp32": fp32, "int8": int8}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CLIP_CONFIG_PATH)
    args = parser.parse_args()
    run(ClipTrainConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
