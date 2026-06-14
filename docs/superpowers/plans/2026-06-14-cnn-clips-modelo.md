# CNN de clips — sub-proyecto 2 (Modelo) · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar un clasificador multi-frame visual-only (ResNet18 congelada + mean-pool + cabeza MLP) sobre el dataset de clips, con augmentation medida, métricas por clase y un bundle exportado para serving.

**Architecture:** Por clip de K frames, una ResNet18 ImageNet congelada (forward sin gradientes) produce un embedding 512-dim por frame; se promedian los K y una cabeza MLP chica predice la clase. Solo entrena la cabeza. Se entrena sin y con data augmentation para medir el impacto. El bundle exporta la cabeza + los transforms de eval (anti-skew) y `predict_clip` es la fuente única de inferencia.

**Tech Stack:** PyTorch + torchvision (MPS en Apple Silicon), OpenCV (lectura de frames), pandas, scikit-learn (métricas, class weights), MLflow, pytest.

**Spec:** `docs/superpowers/specs/2026-06-14-cnn-clips-modelo-design.md`

**Branch:** `feat/fase-3.5-modelo` (ya creada; spec commiteado ahí).

---

## Convenciones del repo

- Correr desde `backend/` con `uv run ...`. Tests: `uv run pytest`. Lint: `uv run ruff check . && uv run ruff format --check .`.
- Código/docstrings en inglés; commits en español, conventional commits, sin firma de Claude.
- `data/` y `models/` gitignored: frames, parquet y el bundle `.pt` NUNCA se commitean.
- Antes de cada commit: `uv run pytest -q` (suite completa) + ruff. Si ruff format pide cambios, `uv run ruff format .`.

---

## File Structure

- Modify: `backend/pyproject.toml` — grupo `cnn` (torch + torchvision).
- Create: `configs/train_clips.yaml` — config del entrenamiento.
- Create: `backend/src/models/clip_config.py` — config tipada (`ClipTrainConfig`).
- Create: `backend/src/models/clip_model.py` — `pick_device`, `build_transforms`, `ClipClassifier`/`build_clip_model`.
- Create: `backend/src/data/clips_dataset.py` — `ClipsDataset`.
- Create: `backend/src/models/clip_export.py` — `ClipModelMeta`, `save_clip_bundle`, `load_clip_bundle`, `predict_clip`.
- Create: `backend/src/models/train_clips.py` — `fit`, `evaluate_clips`, `run`.
- Create tests: `test_clip_config.py`, `test_clip_model.py`, `test_clips_dataset.py`, `test_clip_export.py`, `test_train_clips.py`.

---

## Task 1: Dependencias `cnn` + config del entrenamiento

**Files:**
- Modify: `backend/pyproject.toml`
- Create: `configs/train_clips.yaml`
- Create: `backend/src/models/clip_config.py`
- Test: `backend/tests/test_clip_config.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clip_config.py`:

```python
from src.models.clip_config import DEFAULT_CLIP_CONFIG_PATH, ClipTrainConfig


def test_clip_config_loads() -> None:
    cfg = ClipTrainConfig.from_yaml(DEFAULT_CLIP_CONFIG_PATH)
    assert cfg.backbone == "resnet18"
    assert cfg.k > 0 and cfg.frame_size > 0
    assert cfg.train.epochs > 0
    assert len(cfg.normalize.mean) == 3 and len(cfg.normalize.std) == 3
    # tracking_uri con default expandido (sin la env var).
    assert cfg.mlflow.tracking_uri.startswith("sqlite") or cfg.mlflow.tracking_uri.startswith("http")


def test_clip_paths_resolved() -> None:
    cfg = ClipTrainConfig.from_yaml(DEFAULT_CLIP_CONFIG_PATH)
    assert cfg.paths.resolved("model_dir").name == "clips-v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_config.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.models.clip_config'`.

- [ ] **Step 3: Add the `cnn` dependency group**

In `backend/pyproject.toml`, add a new group under `[dependency-groups]`:

```toml
# CNN multi-frame de clips (Fase 3.5). Solo en training; el serving (sub-proyecto 4)
# las moverá/sumará a prod. torch trae soporte MPS en Apple Silicon.
cnn = [
    "torch>=2.4",
    "torchvision>=0.19",
]
```

Then: `cd backend && uv sync --group cnn` (descarga torch/torchvision, puede tardar).

- [ ] **Step 4: Create `configs/train_clips.yaml`**

```yaml
# Entrenamiento del clasificador multi-frame de clips (Fase 3.5, visual-only).
seed: 42
backbone: resnet18
pooling: mean # mean | max
k: 8 # frames por clip (debe coincidir con el dataset de clips)
frame_size: 224

head:
  hidden: 256
  dropout: 0.3

train:
  epochs: 25
  patience: 5 # early stopping sobre macro-F1 de validación
  lr: 0.001
  batch_size: 16

# Se entrena sin y con augmentation para medir el impacto (electivo de optimización).
augment: true

# Normalización ImageNet (el backbone es pre-entrenado). Se serializa con el bundle.
normalize:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}
  experiment_name: clips-cnn-v1

paths:
  model_dir: "models/clips-v1"
  metrics_dir: "report/metrics"
  manifest: "data/processed/clips_manifest.parquet"
  processed_dir: "data/processed"
```

- [ ] **Step 5: Create `backend/src/models/clip_config.py`**

```python
"""Typed loader for ``configs/train_clips.yaml`` (Fase 3.5 — CNN de clips)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# backend/src/models/clip_config.py → parents: [0]=models [1]=src [2]=backend [3]=repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def _expand_env(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group("name"), m.group("default") or "")

    return _ENV_PATTERN.sub(repl, value)


class HeadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hidden: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    lr: float = Field(gt=0)
    batch_size: int = Field(gt=0)


class NormalizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean: list[float] = Field(min_length=3, max_length=3)
    std: list[float] = Field(min_length=3, max_length=3)


class ClipMLflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tracking_uri: str
    experiment_name: str


class ClipPathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_dir: str
    metrics_dir: str
    manifest: str
    processed_dir: str

    def resolved(self, name: str) -> Path:
        return REPO_ROOT / getattr(self, name)


class ClipTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    backbone: str
    pooling: str
    k: int = Field(gt=0)
    frame_size: int = Field(gt=0)
    head: HeadConfig
    train: TrainParams
    augment: bool
    normalize: NormalizeConfig
    mlflow: ClipMLflowConfig
    paths: ClipPathsConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> ClipTrainConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if isinstance(raw.get("mlflow", {}).get("tracking_uri"), str):
            raw["mlflow"]["tracking_uri"] = _expand_env(raw["mlflow"]["tracking_uri"])
        return cls.model_validate(raw)


DEFAULT_CLIP_CONFIG_PATH = REPO_ROOT / "configs" / "train_clips.yaml"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_config.py -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock configs/train_clips.yaml backend/src/models/clip_config.py backend/tests/test_clip_config.py
git commit -m "feat: deps cnn (torch/torchvision) + config del entrenamiento de clips"
```

---

## Task 2: `clip_model.py` — device + transforms

**Files:**
- Create: `backend/src/models/clip_model.py`
- Test: `backend/tests/test_clip_model.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clip_model.py`:

```python
import numpy as np
import torch

from src.models.clip_model import build_transforms, pick_device

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def test_pick_device_returns_torch_device() -> None:
    assert isinstance(pick_device(), torch.device)


def test_eval_transform_is_deterministic() -> None:
    frame = (np.arange(64 * 64 * 3, dtype=np.uint8) % 256).reshape(64, 64, 3)
    t = build_transforms(augment=False, frame_size=32, mean=MEAN, std=STD)
    a, b = t(frame), t(frame)
    assert a.shape == (3, 32, 32) and a.dtype == torch.float32
    assert torch.equal(a, b)


def test_augment_transform_outputs_right_shape() -> None:
    frame = (np.arange(64 * 64 * 3, dtype=np.uint8) % 256).reshape(64, 64, 3)
    t = build_transforms(augment=True, frame_size=32, mean=MEAN, std=STD)
    out = t(frame)
    assert out.shape == (3, 32, 32) and out.dtype == torch.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_model.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.models.clip_model'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/models/clip_model.py`:

```python
"""Multi-frame clip classifier: frozen ResNet18 backbone + mean-pool + MLP head.

Visual-only (Fase 3.5). The backbone is a frozen ImageNet ResNet18 used as a per-frame
feature extractor; only the head trains. ``build_transforms`` produces the train (augmented)
and eval (deterministic) image transforms; the eval transform is serialized with the bundle
so training and serving preprocess identically (invariant 3).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


def pick_device() -> torch.device:
    """Best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(augment: bool, frame_size: int, mean: list[float], std: list[float]):
    """Image transform for a single frame (numpy HWC uint8 RGB → normalized CHW tensor)."""
    normalize = transforms.Normalize(mean=mean, std=std)
    if augment:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(frame_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.CenterCrop(frame_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ClipClassifier(nn.Module):
    """Frozen ResNet18 per frame → temporal pool over K frames → MLP head → logits."""

    def __init__(
        self,
        n_classes: int,
        hidden: int = 256,
        dropout: float = 0.3,
        pooling: str = "mean",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def train(self, mode: bool = True) -> "ClipClassifier":
        # Keep the frozen backbone in eval mode so BatchNorm running stats stay fixed.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # clips: (B, K, 3, H, W)
        b, k = clips.shape[0], clips.shape[1]
        frames = clips.reshape(b * k, *clips.shape[2:])
        with torch.no_grad():
            feats = self.backbone(frames)  # (B*K, 512)
        feats = feats.reshape(b, k, -1)
        pooled = feats.max(dim=1).values if self.pooling == "max" else feats.mean(dim=1)
        return self.head(pooled)


def build_clip_model(
    n_classes: int,
    hidden: int = 256,
    dropout: float = 0.3,
    pooling: str = "mean",
    backbone: str = "resnet18",
    pretrained: bool = True,
) -> ClipClassifier:
    """Build the clip classifier. Only ``resnet18`` is supported for now."""
    if backbone != "resnet18":
        raise ValueError(f"unsupported backbone {backbone!r}; only 'resnet18'")
    return ClipClassifier(n_classes, hidden, dropout, pooling, pretrained)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_model.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/models/clip_model.py backend/tests/test_clip_model.py
git commit -m "feat: clip_model device + transforms (ResNet18 congelada, anti-skew)"
```

---

## Task 3: `build_clip_model` — forward y congelamiento

**Files:**
- Modify: `backend/tests/test_clip_model.py` (add tests; the model already exists from Task 2)

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_clip_model.py` (add `from src.models.clip_model import build_clip_model` to the imports):

```python
def test_forward_shape_and_frozen_backbone() -> None:
    # pretrained=False evita descargar pesos ImageNet en el unit test.
    model = build_clip_model(n_classes=5, hidden=32, pretrained=False)
    clips = torch.randn(2, 8, 3, 224, 224)
    out = model(clips)
    assert out.shape == (2, 5)
    # backbone congelado, cabeza entrenable.
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.head.parameters())


def test_max_pooling_keeps_shape() -> None:
    model = build_clip_model(n_classes=3, hidden=16, pooling="max", pretrained=False)
    out = model(torch.randn(1, 4, 3, 224, 224))
    assert out.shape == (1, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_model.py -k "forward_shape or max_pooling" -q`
Expected: FAIL — `build_clip_model` not imported yet (ImportError) OR the test functions are new; they fail until the import line is added. (If `build_clip_model` already imported from Task 2, the tests fail only if behavior is wrong; they should pass once the import is present, since the model exists. In that case this task just adds coverage — run Step 4 and confirm PASS.)

- [ ] **Step 3: (No new implementation needed)**

`build_clip_model` / `ClipClassifier` were created in Task 2. This task adds behavioral coverage (forward shape + frozen backbone). If the import line `build_clip_model` is missing in the test, add it.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_model.py -q`
Expected: PASS (5 passed total).

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_clip_model.py
git commit -m "test: forward shape y congelamiento del clip model"
```

---

## Task 4: `clips_dataset.py` — Dataset de clips

**Files:**
- Create: `backend/src/data/clips_dataset.py`
- Test: `backend/tests/test_clips_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clips_dataset.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clips_dataset.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.data.clips_dataset'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/data/clips_dataset.py`:

```python
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
    def __init__(self, manifest: pd.DataFrame, processed_dir, classes: list[str], transform) -> None:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clips_dataset.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/data/clips_dataset.py backend/tests/test_clips_dataset.py
git commit -m "feat: ClipsDataset (lee frames del manifest, devuelve tensor (K,3,H,W))"
```

---

## Task 5: `clip_export.py` — bundle + predict_clip

**Files:**
- Create: `backend/src/models/clip_export.py`
- Test: `backend/tests/test_clip_export.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clip_export.py`:

```python
import numpy as np

from src.models.clip_export import ClipModelMeta, load_clip_bundle, predict_clip, save_clip_bundle
from src.models.clip_model import build_clip_model

CLASSES = ["background", "card", "corner", "goal", "substitution"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _meta() -> ClipModelMeta:
    return ClipModelMeta(
        backbone="resnet18", pooling="mean", classes=CLASSES, k=8, frame_size=32,
        hidden=32, dropout=0.3, normalize_mean=MEAN, normalize_std=STD,
        model_version="clips-test", metrics={},
    )


def test_save_load_predict_roundtrip(tmp_path) -> None:
    meta = _meta()
    model = build_clip_model(len(CLASSES), hidden=32, pooling="mean", pretrained=False)
    save_clip_bundle(model, meta, tmp_path)

    reloaded, meta2 = load_clip_bundle(tmp_path)
    assert meta2.classes == CLASSES and meta2.k == 8

    frames = [np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8) for _ in range(8)]
    label, proba = predict_clip(reloaded, meta2, frames)
    assert label in CLASSES
    assert proba.shape == (5,)
    assert abs(float(proba.sum()) - 1.0) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_export.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.models.clip_export'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/models/clip_export.py`:

```python
"""Serialize / load the clip model bundle and run inference from it (Fase 3.5).

The bundle stores only the trained head ``state_dict`` plus metadata (the frozen ResNet18
backbone is rebuilt from torchvision's ImageNet weights on load). The eval transform is
reconstructed from the serialized normalize stats + frame_size, so training and serving
preprocess identically (invariant 3). ``predict_clip`` is the single shared inference path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.models.clip_model import build_clip_model, build_transforms

BUNDLE_FILE = "clip_model.pt"


@dataclass
class ClipModelMeta:
    backbone: str
    pooling: str
    classes: list[str]
    k: int
    frame_size: int
    hidden: int
    dropout: float
    normalize_mean: list[float]
    normalize_std: list[float]
    model_version: str
    metrics: dict


def save_clip_bundle(model, meta: ClipModelMeta, model_dir: Path) -> Path:
    """Save the head state_dict + metadata to ``model_dir/clip_model.pt``."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / BUNDLE_FILE
    torch.save({"head_state_dict": model.head.state_dict(), "meta": asdict(meta)}, path)
    return path


def load_clip_bundle(model_dir: Path, device: torch.device | None = None):
    """Rebuild the model (frozen backbone + trained head) from the bundle. Returns (model, meta)."""
    payload = torch.load(model_dir / BUNDLE_FILE, map_location="cpu", weights_only=False)
    meta = ClipModelMeta(**payload["meta"])
    model = build_clip_model(
        len(meta.classes), meta.hidden, meta.dropout, meta.pooling, meta.backbone
    )
    model.head.load_state_dict(payload["head_state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return model, meta


def predict_clip(model, meta: ClipModelMeta, frames, device: torch.device | None = None):
    """Predict one clip. ``frames``: iterable of K numpy HWC uint8 RGB arrays.

    Returns ``(label, proba)`` where proba is a softmax vector ordered as ``meta.classes``.
    """
    transform = build_transforms(False, meta.frame_size, meta.normalize_mean, meta.normalize_std)
    clip = torch.stack([transform(f) for f in frames]).unsqueeze(0)  # (1, K, 3, H, W)
    if device is not None:
        clip = clip.to(device)
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(clip), dim=1)[0].cpu().numpy().astype(np.float64)
    return meta.classes[int(proba.argmax())], proba
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_export.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/models/clip_export.py backend/tests/test_clip_export.py
git commit -m "feat: bundle del clip model + predict_clip (fuente unica de inferencia)"
```

---

## Task 6: `train_clips.py` — fit, evaluate y orquestación

**Files:**
- Create: `backend/src/models/train_clips.py`
- Test: `backend/tests/test_train_clips.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_train_clips.py` (testea el núcleo `fit`/`evaluate_clips` con un modelo dummy y tensores, sin descargar ResNet ni tocar MLflow):

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.train_clips import evaluate_clips, fit


class _Dummy(nn.Module):
    def __init__(self, k: int, fs: int, n: int) -> None:
        super().__init__()
        self.lin = nn.Linear(k * 3 * fs * fs, n)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:  # (B, K, 3, H, W)
        return self.lin(clips.flatten(1))


def _loader(n=12, k=2, fs=8, nc=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, k, 3, fs, fs, generator=g)
    y = torch.randint(0, nc, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=4), k, fs, nc


def test_fit_returns_model_and_valid_f1() -> None:
    loader, k, fs, nc = _loader()
    model = _Dummy(k, fs, nc)
    weights = torch.ones(nc)
    model, best_f1 = fit(
        model, loader, loader, ["a", "b", "c"],
        epochs=2, patience=2, lr=0.01, class_weights=weights, device=torch.device("cpu"),
    )
    assert 0.0 <= best_f1 <= 1.0


def test_evaluate_clips_reports_all_classes() -> None:
    loader, k, fs, nc = _loader()
    model = _Dummy(k, fs, nc)
    metrics = evaluate_clips(model, loader, ["a", "b", "c"], torch.device("cpu"))
    assert set(metrics["per_class"]) == {"a", "b", "c"}
    assert 0.0 <= metrics["macro_f1"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_train_clips.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.models.train_clips'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/models/train_clips.py`:

```python
"""Train the clip classifier (Fase 3.5): frozen ResNet18 + head, head-only training.

``fit``/``evaluate_clips`` are the testable core (work with any nn.Module). ``run`` is the
orchestration (real model, dataset, MLflow, export) and is verified live, like train.py's run().
Trains once without augmentation and once with it, logs both to MLflow and reports the delta,
then exports the best (val macro-F1) bundle.

Usage:
    uv run python -m src.models.train_clips --config ../configs/train_clips.yaml
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.clips_dataset import ClipsDataset
from src.models.clip_config import DEFAULT_CLIP_CONFIG_PATH, ClipTrainConfig
from src.models.clip_export import ClipModelMeta, save_clip_bundle
from src.models.clip_model import build_clip_model, build_transforms, pick_device
from src.models.evaluate import compute_metrics, save_confusion_matrix_png


def evaluate_clips(model, loader, classes: list[str], device: torch.device) -> dict:
    """Run the model over a loader and compute per-class metrics (invariant 5)."""
    model.eval()
    model.to(device)
    ys: list[np.ndarray] = []
    probas: list[np.ndarray] = []
    with torch.no_grad():
        for clips, labels in loader:
            logits = model(clips.to(device))
            probas.append(torch.softmax(logits, dim=1).cpu().numpy())
            ys.append(labels.numpy())
    y_true = np.concatenate(ys)
    proba = np.concatenate(probas).astype(np.float64)
    return compute_metrics(y_true, proba.argmax(axis=1), proba, classes)


def fit(model, train_loader, val_loader, classes, epochs, patience, lr, class_weights, device):
    """Train the head with early stopping on validation macro-F1. Returns (best_model, best_f1)."""
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1, best_state, since = -1.0, copy.deepcopy(model.state_dict()), 0
    for _epoch in range(epochs):
        model.train()
        for clips, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(clips.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        val_f1 = evaluate_clips(model, val_loader, classes, device)["macro_f1"]
        if val_f1 > best_f1:
            best_f1, best_state, since = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _class_weights(labels: np.ndarray, classes: list[str]) -> torch.Tensor:
    from sklearn.utils.class_weight import compute_class_weight

    present = np.array([classes.index(label_str) for label_str in labels])
    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=present)
    return torch.tensor(weights, dtype=torch.float32)


def _make_loader(manifest, cfg, classes, augment, processed_dir, shuffle):
    transform = build_transforms(
        augment, cfg.frame_size, cfg.normalize.mean, cfg.normalize.std
    )
    ds = ClipsDataset(manifest, processed_dir, classes, transform)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle)


def run(cfg: ClipTrainConfig) -> ClipModelMeta:
    """Train no-aug and aug runs, log to MLflow, export the best bundle. Verified live."""
    import mlflow

    _seed_everything(cfg.seed)
    device = pick_device()
    processed_dir = cfg.paths.resolved("processed_dir")
    manifest = pd.read_parquet(cfg.paths.resolved("manifest"))
    classes = sorted(manifest["label"].unique().tolist())

    splits = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}
    weights = _class_weights(splits["train"]["label"].to_numpy(), classes)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    metrics_dir = cfg.paths.resolved("metrics_dir")

    best_meta: ClipModelMeta | None = None
    best_model = None
    best_val = -1.0
    results: dict[str, float] = {}

    for augment in (False, True):
        name = "clips-aug" if augment else "clips-noaug"
        train_loader = _make_loader(splits["train"], cfg, classes, augment, processed_dir, True)
        val_loader = _make_loader(splits["val"], cfg, classes, False, processed_dir, False)
        test_loader = _make_loader(splits["test"], cfg, classes, False, processed_dir, False)

        model = build_clip_model(
            len(classes), cfg.head.hidden, cfg.head.dropout, cfg.pooling, cfg.backbone
        )
        model, val_f1 = fit(
            model, train_loader, val_loader, classes,
            cfg.train.epochs, cfg.train.patience, cfg.train.lr, weights, device,
        )
        test_metrics = evaluate_clips(model, test_loader, classes, device)
        results[name] = test_metrics["macro_f1"]

        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "backbone": cfg.backbone, "pooling": cfg.pooling, "k": cfg.k,
                    "augment": augment, "lr": cfg.train.lr, "epochs": cfg.train.epochs,
                    "batch_size": cfg.train.batch_size, "seed": cfg.seed,
                }
            )
            flat = {f"test_f1_{c}": m["f1"] for c, m in test_metrics["per_class"].items()}
            flat["test_macro_f1"] = test_metrics["macro_f1"]
            flat["val_macro_f1"] = val_f1
            mlflow.log_metrics(flat)
            test_pred = _predict_labels(model, test_loader, device)
            cm = save_confusion_matrix_png(
                _loader_labels(test_loader), test_pred, classes, metrics_dir / f"confusion_{name}.png"
            )
            if cm is not None:
                mlflow.log_artifact(str(cm))

        if val_f1 > best_val:
            best_val = val_f1
            best_model = model
            best_meta = ClipModelMeta(
                backbone=cfg.backbone, pooling=cfg.pooling, classes=classes, k=cfg.k,
                frame_size=cfg.frame_size, hidden=cfg.head.hidden, dropout=cfg.head.dropout,
                normalize_mean=cfg.normalize.mean, normalize_std=cfg.normalize.std,
                model_version=f"clips-v1-{name}", metrics=test_metrics,
            )

    assert best_meta is not None and best_model is not None
    save_clip_bundle(best_model, best_meta, cfg.paths.resolved("model_dir"))
    delta = results.get("clips-aug", 0.0) - results.get("clips-noaug", 0.0)
    print(f"\nno-aug test macro-F1={results.get('clips-noaug'):.3f}")
    print(f"aug    test macro-F1={results.get('clips-aug'):.3f}  (Δ augmentation = {delta:+.3f})")
    print(f"exported best bundle (val macro-F1={best_val:.3f}) → {cfg.paths.resolved('model_dir')}")
    return best_meta


def _predict_labels(model, loader, device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for clips, _labels in loader:
            preds.append(torch.softmax(model(clips.to(device)), dim=1).cpu().numpy().argmax(axis=1))
    return np.concatenate(preds)


def _loader_labels(loader) -> np.ndarray:
    return np.concatenate([labels.numpy() for _clips, labels in loader])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-frame clip classifier.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CLIP_CONFIG_PATH)
    args = parser.parse_args()
    run(ClipTrainConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_train_clips.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run full suite + lint**

Run: `cd backend && uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: todo verde. Si ruff format marca archivos: `uv run ruff format .` y volver a chequear.

- [ ] **Step 6: Commit**

```bash
git add backend/src/models/train_clips.py backend/tests/test_train_clips.py
git commit -m "feat: entrenamiento del clip model (fit/evaluate + orquestacion no-aug/aug)"
```

---

## Task 7: Entrenamiento real (no-aug + aug) — corrida local

> Corre el entrenamiento real sobre el dataset de clips (1140 clips). NO necesita NDA password (los frames ya están extraídos). La primera vez descarga los pesos ImageNet de ResNet18 (~45MB, cacheados). En MPS puede tardar (varias decenas de minutos las dos corridas; early stopping recorta). Es una corrida, no un test automatizado.

- [ ] **Step 1: Entrenar (background)**

Run: `cd backend && uv run python -m src.models.train_clips --config ../configs/train_clips.yaml`
Expected (al final): imprime `no-aug test macro-F1=...`, `aug test macro-F1=... (Δ augmentation = ...)`, y `exported best bundle ... → models/clips-v1`. Verificar que `models/clips-v1/clip_model.pt` existe y que hay runs `clips-noaug`/`clips-aug` en MLflow (experimento `clips-cnn-v1`).

- [ ] **Step 2: Sanity de inferencia con el bundle exportado**

Run:
```bash
cd backend && uv run python -c "
from pathlib import Path
import numpy as np
from src.models.clip_export import load_clip_bundle, predict_clip
from src.models.clip_model import pick_device
model, meta = load_clip_bundle(Path('../models/clips-v1'), pick_device())
frames = [np.random.randint(0,255,(224,224,3),dtype=np.uint8) for _ in range(meta.k)]
label, proba = predict_clip(model, meta, frames, pick_device())
print('pred:', label, '| proba sum:', round(float(proba.sum()),4), '| version:', meta.model_version)
"
```
Expected: imprime una clase válida, `proba sum` ≈ 1.0, y la versión del modelo.

- [ ] **Step 3: Commit de artefactos versionables (matrices de confusión van a MLflow; el bundle es gitignored)**

```bash
git status   # confirmar que NO aparecen models/ ni data/ ni *.pt ni *.png
```
No hay artefactos nuevos para commitear (el bundle y las matrices son gitignored / viven en MLflow). Si quedó algo versionable (p. ej. un summary), commitearlo aparte. Verificar con `git status` que ningún `.pt`/`.png`/`data/` se cuela.

---

## Self-Review (hecho)

- **Cobertura del spec:** config (Task 1), device+transforms+modelo congelado (Tasks 2-3), Dataset (Task 4), bundle+predict_clip anti-skew (Task 5), fit/evaluate/orquestación no-aug-vs-aug + MLflow + matriz de confusión (Task 6), corrida real con impacto de augmentation (Task 7). Métricas por clase vía `compute_metrics` (invariante 5). Visual-only (no usa tabular). Splits por `split` del manifest (game_id, invariante 1). ✓
- **Placeholders:** ninguno; todo con código/comando real. ✓
- **Consistencia de tipos:** `build_clip_model`/`build_transforms`/`pick_device`/`ClipsDataset`/`ClipModelMeta`/`save_clip_bundle`/`load_clip_bundle`/`predict_clip`/`fit`/`evaluate_clips` con firmas consistentes entre tasks y tests. `pretrained=False` en unit tests evita descargar pesos. ✓
- **NDA/datos:** el bundle `.pt`, frames y parquet son gitignored; nada de eso se commitea. ✓
- **Out of scope:** Grad-CAM (3), serving (4), frontend (5).
