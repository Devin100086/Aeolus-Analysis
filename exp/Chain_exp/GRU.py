import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

METRIC_KEYS = ("Accuracy", "Recall", "Precision", "F1", "AUC")
DEFAULT_LABEL_NAMES = ("ARR.Delay", "DEP.Delay")

@dataclass
class Config:
    data_dir: Path
    year: str
    feature_columns_path: Path
    batch_size: int
    num_workers: int
    hidden_size: int
    num_layers: int
    output_size: int
    lr: float
    weight_decay: float
    max_epochs: int
    checkpoint_path: Path
    best_model_path: Path
    resume: bool
    plot: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train/evaluate GRU for flight delay.")
    parser.add_argument("--data-dir", default="data/Aeolus/Flight_chain/chain_data_2024")
    parser.add_argument("--year", default="2024")
    parser.add_argument("--feature-columns", default="new_feature_columns.pkl")
    parser.add_argument("--batch-size", type=int, default=80000)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--output-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--checkpoint-path", default="gru_model_checkpoint.pth")
    parser.add_argument("--best-model-path", default="gru_best_model.pth")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    return Config(
        data_dir=Path(args.data_dir),
        year=args.year,
        feature_columns_path=Path(args.feature_columns),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.output_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        checkpoint_path=Path(args.checkpoint_path),
        best_model_path=Path(args.best_model_path),
        resume=args.resume,
        plot=args.plot,
    )


def resolve_output_dir(base_dir: Path, year: str) -> Path:
    year_str = str(year)
    if base_dir.name == year_str:
        return base_dir
    return base_dir / year_str


def load_feature_columns(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_datasets(data_dir: Path, year: str) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    train_path = data_dir / f"flight_chain_train_{year}.pt"
    valid_path = data_dir / f"flight_chain_val_{year}.pt"
    test_path = data_dir / f"flight_chain_test_{year}.pt"
    for path in (train_path, valid_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
    train_dataset = torch.load(train_path)
    valid_dataset = torch.load(valid_path)
    test_dataset = torch.load(test_path)
    return train_dataset, valid_dataset, test_dataset


def build_dataloaders(
    train_dataset: TensorDataset,
    valid_dataset: TensorDataset,
    test_dataset: TensorDataset,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader, test_loader


def unpack_batch(batch: Iterable[torch.Tensor]):
    batch = list(batch)
    if len(batch) < 4:
        raise ValueError(f"Unexpected batch format: {len(batch)} tensors")
    dense_feat, sparse_feat, labels, valid_lens = batch[:4]
    delays = batch[4] if len(batch) > 4 else None
    return dense_feat, sparse_feat, labels, valid_lens, delays


def get_labels(two_labels: torch.Tensor, delays: torch.Tensor | None = None) -> torch.Tensor:
    labels = two_labels
    if labels.dim() == 2:
        labels = labels.unsqueeze(-1)
    if delays is not None and labels.size(-1) == 1 and delays.dim() >= 2 and delays.size(-1) == 2:
        labels = (delays > 15).to(labels.dtype)
    return labels


def get_label_names(num_labels: int) -> list[str]:
    if num_labels == 1:
        return [DEFAULT_LABEL_NAMES[0]]
    if num_labels == len(DEFAULT_LABEL_NAMES):
        return list(DEFAULT_LABEL_NAMES)
    return [f"Label_{idx}" for idx in range(num_labels)]


def compute_binary_metrics(final_labels: np.ndarray, final_probs: np.ndarray) -> dict:
    if final_labels.size == 0:
        return {key: 0.0 for key in METRIC_KEYS}
    try:
        fpr, tpr, thresholds = roc_curve(final_labels, final_probs)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    except ValueError:
        optimal_threshold = 0.5
    try:
        auc = roc_auc_score(final_labels, final_probs)
    except ValueError:
        auc = 0.5
    pred_labels = (final_probs >= optimal_threshold).astype(int)
    return {
        "Accuracy": accuracy_score(final_labels, pred_labels),
        "Recall": recall_score(final_labels, pred_labels, zero_division=0),
        "Precision": precision_score(final_labels, pred_labels, zero_division=0),
        "F1": f1_score(final_labels, pred_labels),
        "AUC": auc,
    }


def infer_label_count(sample: tuple[torch.Tensor, ...]) -> tuple[int, bool]:
    labels = sample[2]
    delays = sample[4] if len(sample) > 4 else None
    if labels.dim() >= 2:
        num_labels = labels.size(-1)
    else:
        num_labels = 1
    uses_delays = False
    if num_labels == 1 and delays is not None and delays.dim() >= 2 and delays.size(-1) == 2:
        num_labels = 2
        uses_delays = True
    return num_labels, uses_delays


class FlightDelayGRU(nn.Module):
    def __init__(self, feature_columns, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        dense_feature_cols, sparse_feature_cols = feature_columns
        self.dense_feature_cols = dense_feature_cols
        self.sparse_feature_cols = sparse_feature_cols
        self.embed_layers = nn.ModuleDict(
            {
                f"embed_{i}": nn.Embedding(feat["feat_num"], feat["embed_dim"])
                for i, feat in enumerate(self.sparse_feature_cols)
            }
        )
        embed_dim = self.sparse_feature_cols[0]["embed_dim"]
        self.input_dim = len(self.dense_feature_cols) + len(self.sparse_feature_cols) * embed_dim
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_inputs = x[:, :, : len(self.dense_feature_cols)]
        sparse_inputs = x[:, :, len(self.dense_feature_cols) :].long()
        sparse_embeds = [
            self.embed_layers[f"embed_{i}"](sparse_inputs[:, :, i])
            for i in range(len(self.sparse_feature_cols))
        ]
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)
        x = torch.cat([sparse_embeds, dense_inputs], dim=-1)
        x, _ = self.gru(x)
        return self.fc(x)


def sequence_mask(x: torch.Tensor, valid_len: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    maxlen = x.size(1)
    mask = torch.arange(maxlen, device=x.device)[None, :] < valid_len[:, None]
    x = x.clone()
    x[~mask] = value
    return x


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float = 4.5):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, label: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
        label = label.float()
        loss = nn.functional.binary_cross_entropy_with_logits(pred, label, reduction="none")
        weights = torch.ones_like(label, dtype=loss.dtype, device=label.device)
        weights[label == 1] *= self.pos_weight
        weighted_loss = loss * weights
        masked_loss = sequence_mask(weighted_loss, valid_len)
        total_loss = masked_loss.sum()
        total_valid = valid_len.sum() * label.size(-1)
        if total_valid > 0:
            return total_loss / total_valid
        return torch.tensor(0.0, device=loss.device)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    config: Config,
) -> Tuple[list, list]:
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.best_model_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = MaskedBCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=1e-6
    )

    start_epoch = 0
    best_val_loss = float("inf")
    loss_history = []
    lr_history = []

    if config.resume and config.checkpoint_path.exists():
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        loss_history = checkpoint.get("train_loss_history", [])
        lr_history = checkpoint.get("lr_history", [])

    use_amp = device.type == "cuda"

    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.max_epochs}")

        for batch in progress_bar:
            dense_feat, sparse_feat, two_labels, valid_lens, delays = unpack_batch(batch)
            dense_feat = dense_feat.to(device)
            sparse_feat = sparse_feat.to(device)
            two_labels = two_labels.to(device)
            valid_lens = valid_lens.to(device)

            optimizer.zero_grad(set_to_none=True)
            delays = delays.to(device) if delays is not None else None
            labels = get_labels(two_labels, delays)
            features = torch.cat([dense_feat, sparse_feat], dim=-1)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(features)
                loss = criterion(outputs, labels, valid_lens)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            epoch_loss += loss.item()
            avg_loss = epoch_loss / max(progress_bar.n + 1, 1)
            progress_bar.set_postfix(
                train_loss=f"{avg_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                dense_feat, sparse_feat, two_labels, valid_lens, delays = unpack_batch(batch)
                dense_feat = dense_feat.to(device)
                sparse_feat = sparse_feat.to(device)
                two_labels = two_labels.to(device)
                valid_lens = valid_lens.to(device)
                delays = delays.to(device) if delays is not None else None
                labels = get_labels(two_labels, delays)
                features = torch.cat([dense_feat, sparse_feat], dim=-1)
                outputs = model(features)
                val_loss += criterion(outputs, labels, valid_lens).item()

        avg_val_loss = val_loss / max(len(valid_loader), 1)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_loss_history": loss_history,
            "lr_history": lr_history,
        }
        torch.save(checkpoint, config.checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.best_model_path)

    return loss_history, lr_history


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels, all_valid_lens = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            dense_feat, sparse_feat, two_labels, valid_lens, delays = unpack_batch(batch)
            dense_feat = dense_feat.to(device)
            sparse_feat = sparse_feat.to(device)
            two_labels = two_labels.to(device)
            valid_lens = valid_lens.to(device)

            delays = delays.to(device) if delays is not None else None
            labels = get_labels(two_labels, delays)
            features = torch.cat([dense_feat, sparse_feat], dim=-1)
            logits = model(features)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_valid_lens.append(valid_lens.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    valid_lens = torch.cat(all_valid_lens, dim=0).numpy()

    position_mask = np.arange(logits.shape[1]) < valid_lens[:, None]
    probs = 1.0 / (1.0 + np.exp(-logits))
    label_names = get_label_names(labels.shape[-1])
    metrics = {}
    for idx, label_name in enumerate(label_names):
        final_probs = probs[:, :, idx][position_mask].flatten()
        final_labels = labels[:, :, idx][position_mask].flatten()
        metrics[label_name] = compute_binary_metrics(final_labels, final_probs)
    return metrics


def plot_lr_schedule(lr_history: list, output_base: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = np.arange(len(lr_history))
    max_lr = max(lr_history) if lr_history else 0.0
    min_lr = min(lr_history) if lr_history else 0.0

    fig, ax = plt.subplots()
    ax.plot(epochs, lr_history, color="#1f77b4", marker="o", markersize=5)
    if lr_history:
        ax.annotate(
            f"Initial LR: {max_lr:.1e}",
            xy=(0, max_lr),
            xytext=(5, max_lr * 1.2 if max_lr > 0 else 0.0),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
        ax.annotate(
            f"Min LR: {min_lr:.1e}",
            xy=(epochs[-1], min_lr),
            xytext=(max(epochs[-1] - 20, 0), min_lr * 1.5 if min_lr > 0 else 0.0),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.savefig(output_base.with_suffix(".png"), format="png", bbox_inches="tight")


def plot_loss_curve(loss_history: list, output_base: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        marker="o",
        markersize=4,
        linewidth=1.5,
        color="#1f77b4",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.6)
    plt.savefig(output_base.with_suffix(".pdf"), dpi=300)
    plt.savefig(output_base.with_suffix(".png"), dpi=300)


def save_metrics_csv(
    metrics_train: dict,
    metrics_valid: dict,
    metrics_test: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.csv"
    fieldnames = ["Label", "Metric", "Train", "Val", "Test"]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for label_name, train_metrics in metrics_train.items():
            valid_metrics = metrics_valid.get(label_name, {})
            test_metrics = metrics_test.get(label_name, {})
            for metric_name in METRIC_KEYS:
                writer.writerow(
                    {
                        "Label": label_name,
                        "Metric": metric_name,
                        "Train": train_metrics.get(metric_name, ""),
                        "Val": valid_metrics.get(metric_name, ""),
                        "Test": test_metrics.get(metric_name, ""),
                    }
                )


def main() -> None:
    config = parse_args()
    output_dir = resolve_output_dir(config.checkpoint_path.parent, config.year)
    config.checkpoint_path = output_dir / config.checkpoint_path.name
    config.best_model_path = output_dir / config.best_model_path.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_columns = load_feature_columns(config.feature_columns_path)
    train_dataset, valid_dataset, test_dataset = load_datasets(config.data_dir, config.year)
    train_loader, valid_loader, test_loader = build_dataloaders(
        train_dataset, valid_dataset, test_dataset, config.batch_size, config.num_workers
    )
    num_labels, uses_delays = infer_label_count(train_dataset[0])
    if config.output_size != num_labels:
        print(
            f"Warning: output_size {config.output_size} does not match dataset labels "
            f"({num_labels}). Using output_size={num_labels}."
        )
        config.output_size = num_labels
    if num_labels == 1 and not uses_delays:
        print("Info: dataset provides a single label; only ARR.Delay will be reported.")

    model = FlightDelayGRU(
        feature_columns=feature_columns,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.output_size,
    ).to(device)

    loss_history, lr_history = train(model, train_loader, valid_loader, device, config)

    metrics_train = evaluate_model(model, train_loader, device)
    metrics_valid = evaluate_model(model, valid_loader, device)
    metrics_test = evaluate_model(model, test_loader, device)

    for label_name, train_metrics in metrics_train.items():
        print(f"\n{label_name}")
        print("{:<15} {:<10} {:<10} {:<10}".format("Metric", "Train", "Val", "Test"))
        valid_metrics = metrics_valid.get(label_name, {})
        test_metrics = metrics_test.get(label_name, {})
        for metric_name in METRIC_KEYS:
            print(
                "{:<15} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    metric_name,
                    train_metrics[metric_name],
                    valid_metrics[metric_name],
                    test_metrics[metric_name],
                )
            )

    save_metrics_csv(metrics_train, metrics_valid, metrics_test, config.checkpoint_path.parent)

    if config.plot:
        output_dir = config.checkpoint_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_lr_schedule(lr_history, output_dir / "gru_lr_schedule")
        plot_loss_curve(loss_history, output_dir / "gru_training_loss")


if __name__ == "__main__":
    main()
