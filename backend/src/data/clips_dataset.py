"""PyTorch Dataset over the clip manifest (Fase 3.5).

Each item is one clip: the K frame JPGs (paths relative to ``processed_dir``) read with
OpenCV, transformed, and stacked into a ``(K, 3, H, W)`` tensor, plus the integer class index.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class ClipsDataset(Dataset):
    def __init__(
        self, manifest: pd.DataFrame, processed_dir, classes: list[str], transform
    ) -> None:
        self.rows = manifest.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        frames = []
        for rel in row["frame_paths"]:
            bgr = cv2.imread(str(self.processed_dir / rel))
            if bgr is None:
                raise FileNotFoundError(f"missing frame: {self.processed_dir / rel}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(rgb))
        clip = torch.stack(frames)  # (K, 3, H, W)
        return clip, self.class_to_idx[row["label"]]
