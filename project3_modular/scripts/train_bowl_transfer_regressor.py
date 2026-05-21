from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATASET_JSON = PROJECT_ROOT / "data" / "bowl_transfer_dataset" / "bowl_transfer_examples.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "bowl_transfer_dataset" / "bowl_transfer_regressor.pt"


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


class BowlTransferMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 5,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small MLP bowl_xyz -> over-bowl joint pose regressor."
    )

    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def load_examples(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "examples" in raw:
        return list(raw["examples"])
    raise ValueError(f"Unsupported dataset format: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_split(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = max(1, int(round(n * val_fraction))) if n >= 5 else 1
    n_val = min(n_val, n - 1) if n > 1 else 0

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return train_idx, val_idx


def mae_deg(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"

    examples = load_examples(args.dataset_json)
    if len(examples) < 2:
        raise RuntimeError("Need at least 2 examples to train/validate.")

    xs = []
    ys = []
    for ex in examples:
        xs.append(ex["bowl_xyz"])
        ys.append(ex["release_joint_positions"][:5])  # arm joints only; keep gripper unchanged during transfer

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    if x.shape[1] != 3:
        raise ValueError(f"Expected bowl_xyz shape (*,3), got {x.shape}")
    if y.shape[1] != 5:
        raise ValueError(f"Expected arm-joint output shape (*,5), got {y.shape}")

    train_idx, val_idx = make_split(len(examples), args.val_fraction, args.seed)

    x_mean = x[train_idx].mean(axis=0)
    x_std = np.maximum(x[train_idx].std(axis=0), 1e-6)
    y_mean = y[train_idx].mean(axis=0)
    y_std = np.maximum(y[train_idx].std(axis=0), 1e-6)

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    x_t = torch.tensor(x_norm, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)

    train_ds = TensorDataset(x_t[train_idx], y_t[train_idx])
    val_ds = TensorDataset(x_t[val_idx], y_t[val_idx]) if len(val_idx) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    model = BowlTransferMLP(
        input_dim=3,
        output_dim=5,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    print(f"Loaded {len(examples)} examples.")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            if val_ds is not None:
                xb = x_t[val_idx].to(device)
                yb = y_t[val_idx].to(device)
                val_pred = model(xb)
                val_loss = loss_fn(val_pred, yb).item()

                # MAE in original joint units.
                val_pred_orig = val_pred.cpu() * torch.tensor(y_std) + torch.tensor(y_mean)
                yb_orig = y[val_idx]
                val_mae = float(np.mean(np.abs(val_pred_orig.numpy() - yb_orig)))
            else:
                val_loss = float(np.mean(train_losses))
                val_mae = float("nan")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"epoch {epoch:4d} | "
                f"train_loss={np.mean(train_losses):.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"val_joint_MAE={val_mae:.2f}°"
            )

        if no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No best state was recorded.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_all_norm = model(x_t.to(device)).cpu().numpy()
        pred_all = pred_all_norm * y_std + y_mean

    train_mae = float(np.mean(np.abs(pred_all[train_idx] - y[train_idx])))
    val_mae = float(np.mean(np.abs(pred_all[val_idx] - y[val_idx]))) if len(val_idx) else float("nan")

    checkpoint = {
        "model_state_dict": best_state,
        "model_config": {
            "input_dim": 3,
            "output_dim": 5,
            "hidden_dim": args.hidden_dim,
            "num_hidden_layers": args.num_hidden_layers,
        },
        "normalization": {
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
        },
        "joint_names": JOINT_NAMES,
        "dataset_json": str(args.dataset_json),
        "num_examples": len(examples),
        "best_epoch": best_epoch,
        "train_joint_mae": train_mae,
        "val_joint_mae": val_mae,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_path)

    summary_path = args.output_path.with_suffix(".json")
    summary = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Train joint MAE: {train_mae:.2f}°")
    print(f"Val joint MAE:   {val_mae:.2f}°")
    print(f"Saved checkpoint: {args.output_path}")
    print(f"Saved summary:    {summary_path}")


if __name__ == "__main__":
    main()
