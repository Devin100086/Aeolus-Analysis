import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import load_graphs
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score
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
    output_dir: Path
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    epochs: int
    threshold_minutes: int
    task: str
    seed: int
    train_ratio: float
    val_ratio: float
    feature_mode: str
    sparse_embed_dim: int
    batch_nodes: int
    oversample: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Temporal GNN with cross-day state propagation."
    )
    parser.add_argument("--graph-dir", default="data/Aeolus/Flight_network/network_data_2024")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-12-30")
    parser.add_argument("--output-dir", default="checkpoints/TemporalGNN")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threshold-minutes", type=int, default=15)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--feature-mode", choices=["feat", "split"], default="feat")
    parser.add_argument("--sparse-embed-dim", type=int, default=8)
    parser.add_argument("--batch-nodes", type=int, default=4096)
    parser.add_argument("--oversample", action="store_true")
    args = parser.parse_args()

    return Config(
        graph_dir=Path(args.graph_dir),
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        output_dir=Path(args.output_dir),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        threshold_minutes=args.threshold_minutes,
        task=args.task,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        feature_mode=args.feature_mode,
        sparse_embed_dim=args.sparse_embed_dim,
        batch_nodes=args.batch_nodes,
        oversample=args.oversample,
    )


def resolve_output_dir(base_dir: Path, year: int) -> Path:
    year_str = str(year)
    if base_dir.name == year_str:
        return base_dir
    return base_dir / year_str


def load_graph_paths(graph_dir: Path, start: date, end: date) -> List[Path]:
    paths = []
    current = start
    while current <= end:
        name = f"graph{current.strftime('%Y%m%d')}.dgl"
        path = graph_dir / name
        if path.exists():
            paths.append(path)
        current += timedelta(days=1)
    return paths


def split_graphs(paths: List[Path], train_ratio: float, val_ratio: float):
    n_total = len(paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_paths = paths[:n_train]
    val_paths = paths[n_train : n_train + n_val]
    test_paths = paths[n_train + n_val :]
    return train_paths, val_paths, test_paths


def load_graph(path: Path):
    (g,), _ = load_graphs(str(path))
    return g


def sample_node_indices(
    labels: torch.Tensor,
    batch_nodes: int,
    oversample: bool,
    rng: np.random.RandomState,
) -> torch.Tensor:
    num_nodes = labels.shape[0]
    if batch_nodes <= 0 or batch_nodes >= num_nodes:
        return torch.arange(num_nodes)
    if not oversample:
        idx = rng.choice(num_nodes, size=batch_nodes, replace=False)
        return torch.from_numpy(idx).long()
    labels_np = labels.cpu().numpy()
    pos_idx = np.where(labels_np > 0.5)[0]
    neg_idx = np.where(labels_np <= 0.5)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        idx = rng.choice(num_nodes, size=batch_nodes, replace=True)
        return torch.from_numpy(idx).long()
    pos_count = batch_nodes // 2
    neg_count = batch_nodes - pos_count
    pos_sample = rng.choice(pos_idx, size=pos_count, replace=True)
    neg_sample = rng.choice(neg_idx, size=neg_count, replace=True)
    idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(idx)
    return torch.from_numpy(idx).long()


def resolve_feature_names(
    desired: List[str],
    available: set,
    aliases: dict | None = None,
) -> tuple[List[str], List[str]]:
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


def compute_sparse_cardinalities(
    graph_paths: List[Path],
    sparse_names: List[str],
) -> List[int]:
    max_vals = None
    for path in tqdm(graph_paths, desc="Scanning sparse cardinalities"):
        g = load_graph(path)
        if g.num_nodes() == 0:
            continue
        vals = []
        for name in sparse_names:
            t = g.ndata[name]
            if t.dim() > 1:
                t = t.squeeze()
            t = torch.nan_to_num(t.float(), nan=0.0)
            vals.append(int(t.max().item()) if t.numel() else 0)
        max_vals = vals if max_vals is None else [max(a, b) for a, b in zip(max_vals, vals)]
    if max_vals is None:
        raise ValueError("Failed to compute sparse cardinalities; no valid graphs.")
    return [v + 1 for v in max_vals]


def get_node_features(g) -> torch.Tensor:
    if "feat" in g.ndata:
        x = g.ndata["feat"]
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return x.float()

    preferred = [
        "O_LATITUDE",
        "O_LONGITUDE",
        "D_LATITUDE",
        "D_LONGITUDE",
        "FLIGHTS",
        "O_PRCP",
        "O_WSPD",
        "D_PRCP",
        "D_WSPD",
        "DAY_OF_WEEK",
        "MONTH",
        "CRS_ARR_TIME_HOUR",
        "CRS_DEP_TIME_HOUR",
        "ORIGIN",
        "DEST",
    ]
    tensors = []
    for name in preferred:
        if name in g.ndata:
            t = g.ndata[name]
            if t.dim() == 1:
                t = t.unsqueeze(1)
            tensors.append(t.float())
    if not tensors:
        raise ValueError("No usable node features found in graph.")
    return torch.cat(tensors, dim=1)


def get_labels(g, task: str, threshold_minutes: int) -> torch.Tensor:
    if "label" not in g.ndata:
        raise ValueError("Graph is missing 'label' in ndata.")
    y = g.ndata["label"].float()
    if y.dim() > 1:
        y = y.squeeze()
    if task == "classification":
        return (y * 60 > threshold_minutes).float()
    return y


def edge_weight(g) -> torch.Tensor:
    if "INTERVAL_TIME" not in g.edata or g.num_edges() == 0:
        return torch.ones(g.num_edges(), device=g.device)
    w = g.edata["INTERVAL_TIME"].float().squeeze()
    return 1.0 / (1.0 + w)


class SparseFeatureEncoder(nn.Module):
    def __init__(self, cardinalities: List[int], embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=card, embedding_dim=embed_dim) for card in cardinalities]
        )

    def forward(self, sparse_tensors: List[torch.Tensor]) -> torch.Tensor:
        embeds = []
        for emb, t in zip(self.embeddings, sparse_tensors):
            embeds.append(emb(t))
        return torch.cat(embeds, dim=1)


class TemporalStateGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        task: str,
        feature_mode: str = "feat",
        dense_names: List[str] | None = None,
        sparse_names: List[str] | None = None,
        sparse_cardinalities: List[int] | None = None,
        sparse_embed_dim: int = 8,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for in_dim_i in dims:
            self.layers.append(GraphConv(in_dim_i, hidden_dim, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.state_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim * 2, 1)
        self.task = task
        self.feature_mode = feature_mode
        self.dense_names = dense_names or []
        self.sparse_names = sparse_names or []
        self.sparse_cardinalities = sparse_cardinalities or []
        self.sparse_embed_dim = sparse_embed_dim
        self.sparse_encoder = None
        if self.feature_mode == "split" and self.sparse_names:
            self.sparse_encoder = SparseFeatureEncoder(self.sparse_cardinalities, self.sparse_embed_dim)

    def build_features(self, g) -> torch.Tensor:
        if self.feature_mode == "feat":
            return get_node_features(g)

        dense_tensors = [ensure_2d(g.ndata[name]).float() for name in self.dense_names]
        dense = torch.cat(dense_tensors, dim=1) if dense_tensors else None

        if self.sparse_names:
            sparse_tensors = []
            for name, card in zip(self.sparse_names, self.sparse_cardinalities):
                t = g.ndata[name]
                if t.dim() > 1:
                    t = t.squeeze()
                t = torch.nan_to_num(t.float(), nan=0.0).long()
                if card > 0:
                    t = torch.clamp(t, 0, card - 1)
                sparse_tensors.append(t)
            sparse_embeds = self.sparse_encoder(sparse_tensors) if self.sparse_encoder else None
            if dense is None:
                return sparse_embeds
            return torch.cat([dense, sparse_embeds], dim=1)

        if dense is None:
            raise ValueError("No features available for split mode.")
        return dense

    def forward(self, g, w, state):
        h = self.build_features(g)
        for layer in self.layers:
            h = layer(g, h, edge_weight=w)
            h = F.relu(h)
            h = self.dropout(h)

        graph_emb = h.mean(dim=0)
        if state is None:
            state = torch.zeros_like(graph_emb)
        state = self.state_cell(graph_emb, state)

        state_broadcast = state.unsqueeze(0).expand(h.size(0), -1)
        logits = self.out(torch.cat([h, state_broadcast], dim=1)).squeeze(-1)
        return logits, state


def evaluate(model, graph_paths: List[Path], device: torch.device, task: str, threshold_minutes: int):
    model.eval()
    preds = []
    labels = []
    losses = []
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    state = None

    with torch.no_grad():
        for path in graph_paths:
            g = load_graph(path).to(device)
            if g.num_nodes() == 0:
                continue
            y = get_labels(g, task, threshold_minutes).to(device)
            w = edge_weight(g).to(device)
            logits, state = model(g, w, state)
            losses.append(loss_fn(logits, y).item())
            preds.append(logits.detach().cpu())
            labels.append(y.detach().cpu())

    if not preds:
        return {"loss": 0.0, "auc": 0.5}
    preds = torch.cat(preds, dim=0).squeeze()
    labels = torch.cat(labels, dim=0).squeeze()
    if task == "classification":
        auc = roc_auc_score(labels.numpy(), torch.sigmoid(preds).numpy())
    else:
        auc = 0.0
    return {"loss": float(np.mean(losses)) if losses else 0.0, "auc": auc}


def save_metrics_csv(metrics_train: dict, metrics_val: dict, metrics_test: dict, output_dir: Path):
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
    output_dir = resolve_output_dir(config.output_dir, config.start_date.year)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph_paths = load_graph_paths(config.graph_dir, config.start_date, config.end_date)
    if not graph_paths:
        raise ValueError("No graphs found in the specified date range.")

    train_paths, val_paths, test_paths = split_graphs(
        graph_paths, config.train_ratio, config.val_ratio
    )

    sample_graph = load_graph(train_paths[0])
    if sample_graph.num_nodes() == 0:
        raise ValueError("Sample graph has no nodes.")

    dense_names = []
    sparse_names = []
    sparse_cardinalities = []
    if config.feature_mode == "split":
        available = set(sample_graph.ndata.keys())
        dense_names, missing_dense = resolve_feature_names(DENSE_FEATURE_NAMES, available)
        sparse_names, missing_sparse = resolve_feature_names(
            SPARSE_FEATURE_NAMES, available
        )
        if missing_dense:
            print(f"Missing dense features (skipped): {missing_dense}")
        if missing_sparse:
            print(f"Missing sparse features (skipped): {missing_sparse}")
        if not dense_names and not sparse_names:
            raise ValueError("No dense/sparse features found for split mode.")
        sparse_cardinalities = compute_sparse_cardinalities(graph_paths, sparse_names) if sparse_names else []
        dense_dim = 0
        for name in dense_names:
            dense_dim += ensure_2d(sample_graph.ndata[name]).shape[1]
        in_dim = dense_dim + len(sparse_names) * config.sparse_embed_dim
    else:
        in_dim = get_node_features(sample_graph).shape[1]

    model = TemporalStateGNN(
        in_dim=in_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        task=config.task,
        feature_mode=config.feature_mode,
        dense_names=dense_names,
        sparse_names=sparse_names,
        sparse_cardinalities=sparse_cardinalities,
        sparse_embed_dim=config.sparse_embed_dim,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss() if config.task == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_val_auc = -1.0
    best_model_path = output_dir / "temporal_gnn_best.pth"
    last_model_path = output_dir / "temporal_gnn_last.pth"

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        state = None
        step = 0
        rng = np.random.RandomState(config.seed + epoch)
        progress = tqdm(train_paths, desc=f"Epoch {epoch}/{config.epochs}")
        for path in progress:
            g_full = load_graph(path)
            if g_full.num_nodes() == 0:
                continue
            y_full = get_labels(g_full, config.task, config.threshold_minutes)
            num_batches = 1
            if config.batch_nodes > 0:
                num_batches = max(1, math.ceil(g_full.num_nodes() / config.batch_nodes))

            for _ in range(num_batches):
                node_idx = sample_node_indices(
                    y_full,
                    config.batch_nodes,
                    config.oversample and config.task == "classification",
                    rng,
                )
                g = dgl.node_subgraph(g_full, node_idx)
                g = g.to(device)
                y = get_labels(g, config.task, config.threshold_minutes).to(device)
                w = edge_weight(g).to(device)

                optimizer.zero_grad(set_to_none=True)
                logits, state = model(g, w, state)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                if state is not None:
                    state = state.detach()
                epoch_loss += loss.item()
                step += 1
                progress.set_postfix(loss=f"{epoch_loss/max(1, step):.4f}")

        torch.save(model.state_dict(), last_model_path)
        val_metrics = evaluate(model, val_paths, device, config.task, config.threshold_minutes)
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch}: val_loss={val_metrics['loss']:.4f}, val_auc={val_metrics['auc']:.4f}")

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    metrics_train = evaluate(model, train_paths, device, config.task, config.threshold_minutes)
    metrics_val = evaluate(model, val_paths, device, config.task, config.threshold_minutes)
    metrics_test = evaluate(model, test_paths, device, config.task, config.threshold_minutes)
    save_metrics_csv(metrics_train, metrics_val, metrics_test, output_dir)


if __name__ == "__main__":
    main()
