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
        model,
        loader,
        loader,
        ["a", "b", "c"],
        epochs=2,
        patience=2,
        lr=0.01,
        class_weights=weights,
        device=torch.device("cpu"),
    )
    assert 0.0 <= best_f1 <= 1.0


def test_evaluate_clips_reports_all_classes() -> None:
    loader, k, fs, nc = _loader()
    model = _Dummy(k, fs, nc)
    metrics = evaluate_clips(model, loader, ["a", "b", "c"], torch.device("cpu"))
    assert set(metrics["per_class"]) == {"a", "b", "c"}
    assert 0.0 <= metrics["macro_f1"] <= 1.0
