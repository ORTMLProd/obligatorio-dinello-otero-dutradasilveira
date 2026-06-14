import cv2
import numpy as np
import pandas as pd
import torch

from src.data.clips_dataset import ClipsDataset
from src.models.clip_model import build_transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _write_clip(processed, game, wid, k) -> list[str]:
    rels = []
    for i in range(k):
        rel = f"frames/{game}/{wid}/frame_{i}.jpg"
        path = processed / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((40, 40, 3), (i * 10) % 256, dtype=np.uint8))
        rels.append(rel)
    return rels


def test_dataset_returns_clip_tensor_and_label(tmp_path) -> None:
    processed = tmp_path / "processed"
    rows = [
        {"label": "goal", "frame_paths": _write_clip(processed, "g1", 0, 4)},
        {"label": "background", "frame_paths": _write_clip(processed, "g1", 1, 4)},
    ]
    manifest = pd.DataFrame(rows)
    classes = ["background", "goal"]
    transform = build_transforms(augment=False, frame_size=32, mean=MEAN, std=STD)

    ds = ClipsDataset(manifest, processed, classes, transform)
    assert len(ds) == 2
    clip, label = ds[0]
    assert clip.shape == (4, 3, 32, 32) and clip.dtype == torch.float32
    assert label == 1  # "goal" → index 1
    _, bg_label = ds[1]
    assert bg_label == 0  # "background" → index 0
