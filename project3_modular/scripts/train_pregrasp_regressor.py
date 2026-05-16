from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.pregrasp_joint_regressor import (
    JOINT_KEYS,
    PregraspJointMLP,
    PregraspNormalization,
    save_pregrasp_checkpoint,
)


DEFAULT_DATASET_JSON = (
    PROJECT_ROOT / "data" / "pregrasp_dataset" / "pregrasp_examples.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "pregrasp_regressor" / "best_pregrasp_mlp.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small MLP cube_xyz -> pregrasp_joint_positions regressor."
    )

    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {path}")

    payload = json.loads(path.read_text())
    examples = payload.get("examples", [])

    if len(examples) < 3:
        raise ValueError(
            f"Only found {len(examples)} examples. "
            "Collect at least a few more before training."
        )

    xs: list[list[float]] = []
    ys: list[list[float]] = []

    for ex in examples:
        cube_xyz = ex["cube_xyz"]
        joints = ex["pregrasp_joints"]

        if len(cube_xyz) != 3:
            raise ValueError(f"Expected cube_xyz length 3, got {cube_xyz}")
        if len(joints) != 6:
            raise ValueError(f"Expected pregrasp_joints length 6, got {joints}")

        xs.append([float(v) for v in cube_xyz])
        ys.append([float(v) for v in joints])

    X = np.asarray(xs, dtype=np.float32)
    Y = np.asarray(ys, dtype=np.float32)
    return X, Y


def make_safe_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    std = values.std(axis=axis)
    return np.where(std < 1e-6, 1.0, std).astype(np.float32)


def split_indices(
    n: int,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    if n < 8 or val_fraction <= 0:
        return indices, np.array([], dtype=np.int64)

    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, n - 1)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return train_idx, val_idx


def compute_metrics_deg(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    y_std: torch.Tensor,
) -> dict[str, float]:
    pred = pred_norm * y_std
    target = target_norm * y_std

    abs_err = torch.abs(pred - target)
    mae_all = float(abs_err.mean().item())
    mae_per_sample = abs_err.mean(dim=1)

    return {
        "joint_mae_deg": mae_all,
        "sample_mae_deg_mean": float(mae_per_sample.mean().item()),
        "sample_mae_deg_max": float(mae_per_sample.max().item()),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    y_std_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    losses = []
    preds = []
    targets = []

    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            losses.append(float(loss.item()))
            preds.append(pred.cpu())
            targets.append(yb.cpu())

    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)

    metrics = compute_metrics_deg(
        pred_norm=pred_all,
        target_norm=target_all,
        y_std=y_std_tensor.cpu(),
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    X_raw, Y_raw = load_dataset(args.dataset_json)
    n = len(X_raw)

    train_idx, val_idx = split_indices(n, args.val_fraction, args.seed)

    X_train_raw = X_raw[train_idx]
    Y_train_raw = Y_raw[train_idx]

    x_mean = X_train_raw.mean(axis=0).astype(np.float32)
    x_std = make_safe_std(X_train_raw)
    y_mean = Y_train_raw.mean(axis=0).astype(np.float32)
    y_std = make_safe_std(Y_train_raw)

    normalization = PregraspNormalization(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    X_norm = ((X_raw - x_mean) / x_std).astype(np.float32)
    Y_norm = ((Y_raw - y_mean) / y_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_norm[train_idx]),
        torch.from_numpy(Y_norm[train_idx]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
    )

    val_loader = None
    if len(val_idx) > 0:
        val_ds = TensorDataset(
            torch.from_numpy(X_norm[val_idx]),
            torch.from_numpy(Y_norm[val_idx]),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=len(val_ds),
            shuffle=False,
        )

    model_config = {
        "input_dim": 3,
        "output_dim": 6,
        "hidden_dim": args.hidden_dim,
        "num_hidden_layers": args.num_hidden_layers,
    }

    model = PregraspJointMLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    y_std_tensor = torch.from_numpy(y_std).to(device)

    best_score = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state: dict[str, Any] | None = None

    print(f"Loaded {n} examples.")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    print(f"Device: {device}")
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                y_std_tensor=y_std_tensor,
                device=device,
            )
            score = val_metrics["loss"]
        else:
            val_metrics = None
            score = train_loss

        if score < best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 100 == 0:
            if val_metrics is None:
                print(
                    f"epoch {epoch:4d} | "
                    f"train_loss={train_loss:.5f}"
                )
            else:
                print(
                    f"epoch {epoch:4d} | "
                    f"train_loss={train_loss:.5f} | "
                    f"val_loss={val_metrics['loss']:.5f} | "
                    f"val_joint_MAE={val_metrics['joint_mae_deg']:.2f}°"
                )

        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)

    train_eval = evaluate(
        model=model,
        loader=train_loader,
        loss_fn=loss_fn,
        y_std_tensor=y_std_tensor,
        device=device,
    )

    if val_loader is not None:
        val_eval = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            y_std_tensor=y_std_tensor,
            device=device,
        )
    else:
        val_eval = None

    training_summary = {
        "num_examples": int(n),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "best_selection_loss": float(best_score),
        "train_metrics": train_eval,
        "val_metrics": val_eval,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }

    save_pregrasp_checkpoint(
        path=args.output_path,
        model=model,
        normalization=normalization,
        model_config=model_config,
        joint_keys=JOINT_KEYS,
        training_summary=training_summary,
    )

    summary_path = args.output_path.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(training_summary, indent=2) + "\n")

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Train joint MAE: {train_eval['joint_mae_deg']:.2f}°")
    if val_eval is not None:
        print(f"Val joint MAE:   {val_eval['joint_mae_deg']:.2f}°")
    print(f"Saved checkpoint: {args.output_path}")
    print(f"Saved summary:    {summary_path}")


if __name__ == "__main__":
    main()
