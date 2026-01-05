import argparse
import csv
import itertools
import pickle
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl import load_graphs
from dgl.data import DGLDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


SPARSE_FEATURE_NAMES = [
    "MONTH",
    "DAY_OF_WEEK",
    "CRS_ARR_TIME_HOUR",
    "CRS_DEP_TIME_HOUR",
    "ORIGIN",
    "DEST"
]

DENSE_FEATURE_NAMES = [
    "O_TEMP",
    "D_TEMP",
    "O_PRCP",
    "D_PRCP",
    "O_WSPD",
    "D_WSPD",
]

SPARSE_FEATURE_ALIASES = {
    "ORIGIN_LABEL": "ORIGIN",
    "DEST_LABEL": "DEST",
    "OP_CARRIER_LABEL": "OP_CARRIER",
    "ORIGIN_LEVEL": "ORIGIN",
    "DEST_LEVEL": "DEST",
}


@dataclass
class Config:
    graph_dir: Path
    start_date: date
    end_date: date
    vgae_model_path: Path
    vgae_hidden1: int
    vgae_hidden2: int
    label_threshold: int
    feature_columns_path: Path | None
    include_embedding: bool
    embedding_dim: int
    train_size: int
    val_size: int
    test_size: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    att_vector: int
    dropout: float
    use_dnn: bool
    hidden_units: List[int]
    oversample: bool
    output_dir: Path
    plot: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="AFM training with node embeddings.")
    parser.add_argument("--graph-dir", default="data/Aeolus/Flight_network/network_data_2016")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2016-12-30")
    parser.add_argument("--vgae-model-path", default="vgae/node_embedding.pth")
    parser.add_argument("--vgae-hidden1", type=int, default=16)
    parser.add_argument("--vgae-hidden2", type=int, default=8)
    parser.add_argument("--label-threshold", type=int, default=15)
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--include-embedding", dest="include_embedding", action="store_true", default=True)
    parser.add_argument("--no-embedding", dest="include_embedding", action="store_false")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--val-size", type=int, default=99)
    parser.add_argument("--test-size", type=int, default=66)
    parser.add_argument("--batch-size", type=int, default=16000)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--att-vector", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-dnn", action="store_true")
    parser.add_argument("--hidden-units", default="128,64,32")
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--output-dir", default="checkpoints/AFM")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    hidden_units = [int(x) for x in args.hidden_units.split(",") if x.strip()]
    feature_columns_path = Path(args.feature_columns) if args.feature_columns else None

    return Config(
        graph_dir=Path(args.graph_dir),
        start_date=start_date,
        end_date=end_date,
        vgae_model_path=Path(args.vgae_model_path),
        vgae_hidden1=args.vgae_hidden1,
        vgae_hidden2=args.vgae_hidden2,
        label_threshold=args.label_threshold,
        feature_columns_path=feature_columns_path,
        include_embedding=args.include_embedding,
        embedding_dim=args.embedding_dim,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        att_vector=args.att_vector,
        dropout=args.dropout,
        use_dnn=args.use_dnn,
        hidden_units=hidden_units,
        oversample=args.oversample,
        output_dir=Path(args.output_dir),
        plot=args.plot,
    )


def resolve_output_dir(base_dir: Path, year: int) -> Path:
    year_str = str(year)
    if base_dir.name == year_str:
        return base_dir
    return base_dir / year_str


def resolve_feature_names(
    desired: List[str],
    available: set,
    aliases: dict | None = None,
) -> Tuple[List[str], List[str]]:
    resolved = []
    missing = []
    for name in desired:
        if name in available:
            resolved.append(name)
            continue
        if aliases and name in aliases and aliases[name] in available:
            resolved.append(aliases[name])
            continue
        missing.append(name)
    return resolved, missing


def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(1)
    return tensor


class VGAEModel(nn.Module):
    def __init__(self, in_dim: int, hidden1_dim: int, hidden2_dim: int):
        super().__init__()
        from dgl.nn.pytorch import GraphConv

        self.layers = nn.ModuleList(
            [
                GraphConv(in_dim, hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                GraphConv(hidden1_dim, hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                GraphConv(hidden1_dim, hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
            ]
        )

    def encoder(self, g, features):
        h = self.layers[0](g, features)
        mean = self.layers[1](g, h)
        log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), mean.size(1), device=features.device)
        sampled_z = mean + gaussian_noise * torch.exp(log_std)
        return sampled_z


class FlightGraphDataset(DGLDataset):
    def __init__(
        self,
        graph_dir: Path,
        start_date: date,
        end_date: date,
        vgae_model_path: Path,
        vgae_hidden1: int,
        vgae_hidden2: int,
        label_threshold: int,
        include_embedding: bool,
    ):
        self.graph_dir = graph_dir
        self.start_date = start_date
        self.end_date = end_date
        self.vgae_model_path = vgae_model_path
        self.vgae_hidden1 = vgae_hidden1
        self.vgae_hidden2 = vgae_hidden2
        self.label_threshold = label_threshold
        self.include_embedding = include_embedding
        self.sparse_feat_names = None
        self.dense_feat_names = None
        super().__init__(name="flight_graphs")

    def process(self):
        graphs = []
        current = self.start_date
        min_values = None
        max_values = None

        while current <= self.end_date:
            file_name = self.graph_dir / f"graph{current.strftime('%Y%m%d')}.dgl"
            if not file_name.exists():
                current += timedelta(days=1)
                continue
            (g,), _ = load_graphs(str(file_name))

            if "MONTH" in g.ndata:
                g.ndata["MONTH"] = g.ndata["MONTH"] - 1
            if "DAY_OF_WEEK" in g.ndata:
                g.ndata["DAY_OF_WEEK"] = g.ndata["DAY_OF_WEEK"] - 1

            if "ORIGIN_LEVEL" in g.ndata:
                origin_level = g.ndata["ORIGIN_LEVEL"]
                origin_level[torch.isnan(origin_level)] = 2
                g.ndata["ORIGIN_LEVEL"] = origin_level - 1

            if "DEST_LEVEL" in g.ndata:
                dest_level = g.ndata["DEST_LEVEL"]
                dest_level[torch.isnan(dest_level)] = 2
                g.ndata["DEST_LEVEL"] = dest_level - 1

            if self.sparse_feat_names is None or self.dense_feat_names is None:
                available = set(g.ndata.keys())
                self.sparse_feat_names, missing_sparse = resolve_feature_names(
                    SPARSE_FEATURE_NAMES, available
                )
                self.dense_feat_names, missing_dense = resolve_feature_names(
                    DENSE_FEATURE_NAMES, available
                )
                if missing_sparse:
                    print(f"Missing sparse features (skipped): {missing_sparse}")
                if missing_dense:
                    print(f"Missing dense features (skipped): {missing_dense}")
                if not self.sparse_feat_names or not self.dense_feat_names:
                    raise ValueError("Resolved empty feature list; check graph node attributes.")

            sparse_tensors = [ensure_2d(g.ndata[name]) for name in self.sparse_feat_names]
            g.ndata["sparse_feat"] = torch.cat(sparse_tensors, dim=1)

            dense_tensors = [ensure_2d(g.ndata[name]) for name in self.dense_feat_names]
            g.ndata["dense_feat"] = torch.cat(dense_tensors, dim=1)
            g.ndata["dense_feat"][torch.isnan(g.ndata["dense_feat"])] = 0

            if g.ndata["dense_feat"].numel() == 0:
                print(f"Skipping empty graph: {file_name}")
                current += timedelta(days=1)
                continue

            dense_min = torch.min(g.ndata["dense_feat"], dim=0)[0]
            dense_max = torch.max(g.ndata["dense_feat"], dim=0)[0]
            if min_values is None:
                min_values = dense_min
                max_values = dense_max
            else:
                min_values = torch.min(torch.stack((min_values, dense_min)), dim=0)[0]
                max_values = torch.max(torch.stack((max_values, dense_max)), dim=0)[0]

            bins = [self.label_threshold]
            new_label = np.digitize(g.ndata["label"], bins)
            g.ndata["label"] = torch.from_numpy(new_label)

            graphs.append(g)
            current += timedelta(days=1)

        if not graphs:
            raise ValueError("No graphs loaded. Check --graph-dir and date range.")
        if min_values is None or max_values is None:
            raise ValueError("No non-empty graphs found for dense feature scaling.")

        ranges = max_values - min_values
        ranges[ranges == 0] = 1

        vgae_model = None
        if self.include_embedding:
            vgae_model = VGAEModel(
                in_dim=len(self.dense_feat_names),
                hidden1_dim=self.vgae_hidden1,
                hidden2_dim=self.vgae_hidden2,
            )
            vgae_model.load_state_dict(torch.load(self.vgae_model_path, map_location="cpu"))
            vgae_model.eval()

        for g in graphs:
            g.ndata["dense_feat"] = (g.ndata["dense_feat"] - min_values) / ranges
            if self.include_embedding:
                with torch.no_grad():
                    g.ndata["embedding"] = vgae_model.encoder(g, g.ndata["dense_feat"])
        self.graphs = graphs

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


def split_indices(size: int, train_size: int, val_size: int, test_size: int) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.RandomState(seed=42).permutation(size).tolist()
    total = min(size, train_size + val_size + test_size)
    train_end = min(train_size, total)
    val_end = min(train_end + val_size, total)
    train_idx = rng[:train_end]
    val_idx = rng[train_end:val_end]
    test_idx = rng[val_end:total]
    return train_idx, val_idx, test_idx


def build_table(
    dataset: FlightGraphDataset,
    indices: List[int],
    include_embedding: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sub_data = torch.utils.data.Subset(dataset, indices)
    feat_list = []
    label_list = []
    for i in tqdm(range(len(sub_data)), desc="Processing graphs"):
        dense_feat = sub_data[i].ndata["dense_feat"]
        sparse_feat = sub_data[i].ndata["sparse_feat"]
        if include_embedding:
            embedding_feat = sub_data[i].ndata["embedding"]
            feat = torch.cat([dense_feat, embedding_feat, sparse_feat], dim=-1)
        else:
            feat = torch.cat([dense_feat, sparse_feat], dim=-1)
        label_list.append(sub_data[i].ndata["label"])
        feat_list.append(feat)
    features = torch.cat(feat_list, dim=0)
    labels = torch.cat(label_list, dim=0).squeeze()
    return features, labels


def build_feature_columns(
    dataset: FlightGraphDataset,
    embedding_dim: int,
    feature_columns_path: Path | None,
):
    if feature_columns_path and feature_columns_path.exists():
        with feature_columns_path.open("rb") as f:
            return pickle.load(f)

    max_vals = None
    if not dataset.sparse_feat_names:
        raise ValueError("Dataset is missing sparse feature names.")

    for g in dataset:
        sparse_feat = g.ndata["sparse_feat"]
        current_max = torch.max(sparse_feat, dim=0)[0]
        max_vals = current_max if max_vals is None else torch.max(max_vals, current_max)

    dense_feature_names = dataset.dense_feat_names
    sparse_feature_names = dataset.sparse_feat_names

    sparse_feature_cols = []
    for name, max_val in zip(sparse_feature_names, max_vals):
        sparse_feature_cols.append(
            {
                "feat_name": name,
                "feat_num": int(max_val.item()) + 1,
                "embed_dim": embedding_dim,
            }
        )
    return (dense_feature_names, sparse_feature_cols)


class Dnn(nn.Module):
    def __init__(self, hidden_units: List[int], dropout: float = 0.0):
        super().__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(inp, out) for inp, out in zip(hidden_units[:-1], hidden_units[1:])]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.dnn_network:
            x = F.relu(linear(x))
        return self.dropout(x)


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, att_vector: int):
        super().__init__()
        self.att_w = nn.Linear(embed_dim, att_vector)
        self.att_dense = nn.Linear(att_vector, 1)

    def forward(self, bi_interaction: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.att_w(bi_interaction))
        att_scores = self.att_dense(a)
        att_weight = F.softmax(att_scores, dim=1)
        return torch.sum(att_weight * bi_interaction, dim=1)


class AFM(nn.Module):
    def __init__(
        self,
        feature_columns,
        mode: str,
        hidden_units: List[int],
        dense_input_dim: int,
        att_vector: int = 8,
        dropout: float = 0.0,
        use_dnn: bool = False,
    ):
        super().__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        self.mode = mode
        self.use_dnn = use_dnn
        self.dense_input_dim = dense_input_dim

        self.embed_layers = nn.ModuleDict(
            {
                f"embed_{i}": nn.Embedding(feat["feat_num"], feat["embed_dim"])
                for i, feat in enumerate(self.sparse_feature_cols)
            }
        )

        if self.mode == "att":
            self.attention = AttentionLayer(self.sparse_feature_cols[0]["embed_dim"], att_vector)

        self.fea_num = self.dense_input_dim + self.sparse_feature_cols[0]["embed_dim"]
        if self.use_dnn:
            hidden_units = [self.fea_num] + hidden_units
            self.bn = nn.BatchNorm1d(self.fea_num)
            self.dnn_network = Dnn(hidden_units, dropout)
            self.nn_final_linear = nn.Linear(hidden_units[-1], 1)
        else:
            self.nn_final_linear = nn.Linear(self.fea_num, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_inputs = x[:, : self.dense_input_dim]
        sparse_inputs = x[:, self.dense_input_dim :].long()

        sparse_embeds = [
            self.embed_layers[f"embed_{i}"](sparse_inputs[:, i])
            for i in range(sparse_inputs.shape[1])
        ]
        sparse_embeds = torch.stack(sparse_embeds).permute((1, 0, 2))

        pairs = list(itertools.combinations(range(sparse_embeds.shape[1]), 2))
        first = [p[0] for p in pairs]
        second = [p[1] for p in pairs]
        bi_interaction = sparse_embeds[:, first, :] * sparse_embeds[:, second, :]

        if self.mode == "max":
            att_out = torch.sum(bi_interaction, dim=1)
        elif self.mode == "avg":
            att_out = torch.mean(bi_interaction, dim=1)
        else:
            att_out = self.attention(bi_interaction)

        x = torch.cat([att_out, dense_inputs], dim=-1)

        if not self.use_dnn:
            outputs = torch.sigmoid(self.nn_final_linear(x))
        else:
            x = self.bn(x)
            outputs = torch.sigmoid(self.nn_final_linear(self.dnn_network(x)))
            outputs = outputs.squeeze(-1)
        return outputs


def compute_auc(preds: torch.Tensor, labels: torch.Tensor) -> float:
    try:
        return roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
    except ValueError:
        return 0.5


def compute_binary_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    preds_np = preds.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    if labels_np.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_labels = (preds_np >= threshold).astype(int)
    labels_np = labels_np.astype(int)
    return {
        "accuracy": float(accuracy_score(labels_np, pred_labels)),
        "precision": float(precision_score(labels_np, pred_labels, zero_division=0)),
        "recall": float(recall_score(labels_np, pred_labels, zero_division=0)),
        "f1": float(f1_score(labels_np, pred_labels, zero_division=0)),
    }


def oversample_batch(features: torch.Tensor, labels: torch.Tensor):
    try:
        from imblearn.over_sampling import RandomOverSampler
    except ImportError as exc:
        raise RuntimeError("imblearn is required for --oversample") from exc

    sampler = RandomOverSampler(random_state=0)
    features_np, labels_np = sampler.fit_resample(
        features.cpu().numpy(), labels.cpu().numpy()
    )
    return torch.tensor(features_np), torch.tensor(labels_np)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Config,
):
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    history = []
    best_val_auc = -1.0
    best_model_path = config.output_dir / "afm_best_model.pth"
    last_model_path = config.output_dir / "afm_last_model.pth"

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for step, (features, labels) in enumerate(progress_bar, 1):
            optimizer.zero_grad()
            if config.oversample:
                features, labels = oversample_batch(features, labels)
            features = features.to(device)
            labels = labels.to(device)

            preds = model(features).float().squeeze(-1)
            loss = loss_func(preds, labels.float())
            metric = compute_auc(preds.detach(), labels.detach())

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            metric_sum += metric
            progress_bar.set_postfix(loss=f"{loss_sum/step:.4f}", auc=f"{metric_sum/step:.4f}")

        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                preds = model(features).float().squeeze(-1)
                val_loss_sum += loss_func(preds, labels.float()).item()
                val_metric_sum += compute_auc(preds, labels)

        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(step, 1),
                "auc": metric_sum / max(step, 1),
                "val_loss": val_loss_sum / max(len(val_loader), 1),
                "val_auc": val_metric_sum / max(len(val_loader), 1),
            }
        )
        torch.save(model.state_dict(), last_model_path)
        if history[-1]["val_auc"] > best_val_auc:
            best_val_auc = history[-1]["val_auc"]
            torch.save(model.state_dict(), best_model_path)
        print(
            f"Epoch {epoch}: loss={history[-1]['loss']:.4f}, auc={history[-1]['auc']:.4f}, "
            f"val_loss={history[-1]['val_loss']:.4f}, val_auc={history[-1]['val_auc']:.4f}"
        )
    return history


def plot_history(history: list, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    for metric in ("loss", "auc"):
        train_values = [h[metric] for h in history]
        val_values = [h[f"val_{metric}"] for h in history]
        plt.figure()
        plt.plot(epochs, train_values, "bo--")
        plt.plot(epochs, val_values, "ro-")
        plt.title(f"Training and validation {metric}")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([f"train_{metric}", f"val_{metric}"])
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.pdf", format="pdf", bbox_inches="tight")


def evaluate_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    loss_func = nn.BCELoss()
    losses = []
    preds = []
    labels = []
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features).float().squeeze(-1)
            losses.append(loss_func(outputs, targets.float()).item())
            preds.append(outputs.cpu())
            labels.append(targets.cpu())
    preds = torch.cat(preds, dim=0).squeeze()
    labels = torch.cat(labels, dim=0).squeeze()
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "auc": compute_auc(preds, labels),
    }
    metrics.update(compute_binary_metrics(preds, labels))
    return metrics


def save_metrics_csv(
    metrics_train: dict,
    metrics_val: dict,
    metrics_test: dict,
    output_dir: Path,
) -> None:
    output_path = output_dir / "metrics.csv"
    fieldnames = ["Metric", "Train", "Val", "Test"]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(set(metrics_train) | set(metrics_val) | set(metrics_test)):
            writer.writerow(
                {
                    "Metric": key,
                    "Train": metrics_train.get(key, ""),
                    "Val": metrics_val.get(key, ""),
                    "Test": metrics_test.get(key, ""),
                }
            )


def main() -> None:
    config = parse_args()
    config.output_dir = resolve_output_dir(config.output_dir, config.start_date.year)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FlightGraphDataset(
        graph_dir=config.graph_dir,
        start_date=config.start_date,
        end_date=config.end_date,
        vgae_model_path=config.vgae_model_path,
        vgae_hidden1=config.vgae_hidden1,
        vgae_hidden2=config.vgae_hidden2,
        label_threshold=config.label_threshold,
        include_embedding=config.include_embedding,
    )

    train_idx, val_idx, test_idx = split_indices(
        len(dataset), config.train_size, config.val_size, config.test_size
    )

    train_x, train_y = build_table(dataset, train_idx, config.include_embedding)
    val_x, val_y = build_table(dataset, val_idx, config.include_embedding)
    test_x, test_y = build_table(dataset, test_idx, config.include_embedding)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers
    )

    feature_columns = build_feature_columns(
        dataset, config.embedding_dim, config.feature_columns_path
    )
    dense_input_dim = len(dataset.dense_feat_names)
    if config.include_embedding:
        dense_input_dim += config.vgae_hidden2

    model = AFM(
        feature_columns=feature_columns,
        mode="att",
        hidden_units=config.hidden_units,
        dense_input_dim=dense_input_dim,
        att_vector=config.att_vector,
        dropout=config.dropout,
        use_dnn=config.use_dnn,
    ).to(device)

    history = train_model(model, train_loader, val_loader, device, config)

    best_model_path = config.output_dir / "afm_best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    train_metrics = evaluate_loader(model, train_loader, device)
    val_metrics = evaluate_loader(model, val_loader, device)
    test_metrics = evaluate_loader(model, test_loader, device)
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    save_metrics_csv(train_metrics, val_metrics, test_metrics, config.output_dir)

    if config.plot:
        plot_history(history, config.output_dir)


if __name__ == "__main__":
    main()
