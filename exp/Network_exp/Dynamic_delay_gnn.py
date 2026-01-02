import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

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
    predict_graph: Path | None
    predict_node: int | None
    feature_mode: str
    sparse_embed_dim: int
    batch_nodes: int
    oversample: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Dynamic GNN delay propagation modeling.")
    parser.add_argument("--graph-dir", default="data/Aeolus/Flight_network/network_data_2024")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-12-30")
    parser.add_argument("--output-dir", default="checkpoints/DynamicGNN")
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
    parser.add_argument("--predict-graph", default=None)
    parser.add_argument("--predict-node", type=int, default=None)
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
        predict_graph=Path(args.predict_graph) if args.predict_graph else None,
        predict_node=args.predict_node,
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
        if aliases:
            candidate = None
            if name in aliases:
                alias_val = aliases[name]
                if isinstance(alias_val, (list, tuple, set)):
                    for alt in alias_val:
                        if alt in available:
                            candidate = alt
                            break
                elif alias_val in available:
                    candidate = alias_val
            if candidate is None:
                for alias_name, canonical in aliases.items():
                    if canonical == name and alias_name in available:
                        candidate = alias_name
                        break
            if candidate:
                resolved.append(candidate)
                continue
        missing.append(name)
    return resolved, missing


def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(1)
    return tensor


def normalize_sparse_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.float()
    if name in ("ORIGIN_LEVEL", "DEST_LEVEL"):
        t = torch.nan_to_num(t, nan=2.0) - 1
    else:
        t = torch.nan_to_num(t, nan=0.0)
        if name in ("MONTH", "DAY_OF_WEEK") and t.numel() > 0:
            if t.min() >= 1:
                t = t - 1
    if t.dim() > 1:
        t = t.squeeze()
    return t.long()


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
            if name not in g.ndata:
                vals.append(0)
                continue
            t = normalize_sparse_tensor(name, g.ndata[name])
            vals.append(int(t.max().item()) if t.numel() else 0)
        if max_vals is None:
            max_vals = vals
        else:
            max_vals = [max(a, b) for a, b in zip(max_vals, vals)]
    if max_vals is None:
        raise ValueError("Failed to compute sparse cardinalities; no valid graphs.")
    return [v + 1 for v in max_vals]


def compute_dense_minmax(
    graph_paths: List[Path],
    dense_names: List[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    min_values = None
    max_values = None
    for path in tqdm(graph_paths, desc="Scanning dense min/max"):
        g = load_graph(path)
        if g.num_nodes() == 0:
            continue
        dense_tensors = []
        for name in dense_names:
            if name not in g.ndata:
                dense_tensors.append(torch.zeros(g.num_nodes(), 1))
                continue
            dense_tensors.append(torch.nan_to_num(ensure_2d(g.ndata[name]).float(), nan=0.0))
        if not dense_tensors:
            continue
        dense_feat = torch.cat(dense_tensors, dim=1)
        if dense_feat.numel() == 0:
            continue
        dense_min = torch.min(dense_feat, dim=0)[0]
        dense_max = torch.max(dense_feat, dim=0)[0]
        if min_values is None:
            min_values = dense_min
            max_values = dense_max
        else:
            min_values = torch.min(torch.stack((min_values, dense_min)), dim=0)[0]
            max_values = torch.max(torch.stack((max_values, dense_max)), dim=0)[0]
    if min_values is None or max_values is None:
        raise ValueError("Failed to compute dense min/max; no valid graphs.")
    return min_values, max_values


def get_node_features(g) -> torch.Tensor:
    if "feat" in g.ndata:
        x = g.ndata["feat"]
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return torch.nan_to_num(x.float(), nan=0.0)

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
            tensors.append(torch.nan_to_num(t.float(), nan=0.0))
    if not tensors:
        raise ValueError("No usable node features found in graph.")
    return torch.cat(tensors, dim=1)


def get_labels(g, task: str, threshold_minutes: int) -> torch.Tensor:
    if "ARR_DELAY" in g.ndata:
        y = g.ndata["ARR_DELAY"].float() / 60.0
    elif "label" in g.ndata:
        y = g.ndata["label"].float()
    else:
        raise ValueError("Graph is missing 'label' and 'ARR_DELAY' in ndata.")
    y = torch.nan_to_num(y, nan=0.0)
    if y.dim() > 1:
        y = y.squeeze()
    if task == "classification":
        return (y * 60 > threshold_minutes).float()
    return y


def edge_weight(g) -> torch.Tensor:
    if "INTERVAL_TIME" not in g.edata:
        return torch.ones(g.num_edges(), device=g.device)
    w = torch.nan_to_num(g.edata["INTERVAL_TIME"].float().squeeze(), nan=0.0)
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


class TemporalGraphModel(nn.Module):
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
        dense_min: torch.Tensor | None = None,
        dense_max: torch.Tensor | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for in_dim_i in dims:
            self.layers.append(GraphConv(in_dim_i, hidden_dim, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)
        self.task = task
        self.feature_mode = feature_mode
        self.dense_names = dense_names or []
        self.sparse_names = sparse_names or []
        self.sparse_cardinalities = sparse_cardinalities or []
        self.sparse_embed_dim = sparse_embed_dim
        self.sparse_encoder = None
        if self.feature_mode == "split" and self.sparse_names:
            self.sparse_encoder = SparseFeatureEncoder(self.sparse_cardinalities, self.sparse_embed_dim)
        self.register_buffer("dense_min", dense_min)
        self.register_buffer("dense_max", dense_max)

    def build_features(self, g) -> torch.Tensor:
        if self.feature_mode == "feat":
            return get_node_features(g)

        dense_tensors = []
        for name in self.dense_names:
            if name not in g.ndata:
                dense_tensors.append(torch.zeros(g.num_nodes(), 1, device=g.device))
                continue
            dense_tensors.append(
                torch.nan_to_num(ensure_2d(g.ndata[name]).float(), nan=0.0)
            )
        dense = torch.cat(dense_tensors, dim=1) if dense_tensors else None
        if dense is not None and self.dense_min is not None and self.dense_max is not None:
            denom = self.dense_max - self.dense_min
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            dense = (dense - self.dense_min) / denom
            dense = torch.nan_to_num(dense, nan=0.0)

        if self.sparse_names:
            sparse_tensors = []
            for name, card in zip(self.sparse_names, self.sparse_cardinalities):
                if name not in g.ndata:
                    sparse_tensors.append(torch.zeros(g.num_nodes(), dtype=torch.long, device=g.device))
                    continue
                t = normalize_sparse_tensor(name, g.ndata[name])
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

    def forward(self, g, w):
        h = self.build_features(g)
        for layer in self.layers:
            h = layer(g, h, edge_weight=w)
            h = F.relu(h)
            h = self.dropout(h)
        out = self.out(h).squeeze(-1)
        return out


def split_graphs(paths: List[Path], seed: int, train_ratio: float, val_ratio: float):
    rng = np.random.RandomState(seed=seed)
    indices = rng.permutation(len(paths))
    n_train = int(len(paths) * train_ratio)
    n_val = int(len(paths) * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return (
        [paths[i] for i in train_idx],
        [paths[i] for i in val_idx],
        [paths[i] for i in test_idx],
    )


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


def evaluate(model, graph_paths: List[Path], device: torch.device, task: str, threshold_minutes: int):
    model.eval()
    preds = []
    labels = []
    losses = []
    loss_fn = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()

    with torch.no_grad():
        for path in graph_paths:
            g = load_graph(path).to(device)
            if g.num_nodes() == 0:
                continue
            y = get_labels(g, task, threshold_minutes).to(device)
            w = edge_weight(g).to(device)
            logits = model(g, w)
            losses.append(loss_fn(logits, y).item())
            preds.append(logits.detach().cpu())
            labels.append(y.detach().cpu())

    if not preds:
        return {"loss": 0.0, "auc": 0.5}
    preds = torch.nan_to_num(torch.cat(preds, dim=0).squeeze(), nan=0.0)
    labels = torch.nan_to_num(torch.cat(labels, dim=0).squeeze(), nan=0.0)
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
        graph_paths, config.seed, config.train_ratio, config.val_ratio
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
            SPARSE_FEATURE_NAMES, available, SPARSE_FEATURE_ALIASES
        )
        if missing_dense:
            print(f"Missing dense features (skipped): {missing_dense}")
        if missing_sparse:
            print(f"Missing sparse features (skipped): {missing_sparse}")
        if not dense_names and not sparse_names:
            raise ValueError("No dense/sparse features found for split mode.")
        sparse_cardinalities = compute_sparse_cardinalities(graph_paths, sparse_names) if sparse_names else []
        dense_min, dense_max = (None, None)
        if dense_names:
            dense_min, dense_max = compute_dense_minmax(graph_paths, dense_names)
        dense_dim = 0
        for name in dense_names:
            if name in sample_graph.ndata:
                dense_dim += ensure_2d(sample_graph.ndata[name]).shape[1]
            else:
                dense_dim += 1
        in_dim = dense_dim + len(sparse_names) * config.sparse_embed_dim
    else:
        in_dim = get_node_features(sample_graph).shape[1]
    model = TemporalGraphModel(
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
        dense_min=dense_min,
        dense_max=dense_max,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss() if config.task == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_val_auc = -1.0
    best_model_path = output_dir / "dynamic_gnn_best.pth"
    last_model_path = output_dir / "dynamic_gnn_last.pth"

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
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
                logits = model(g, w)
                loss = loss_fn(logits, y)
                if torch.isnan(loss):
                    print(f"Warning: NaN loss at {path}, skipping batch")
                    continue
                loss.backward()
                optimizer.step()
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

    if config.predict_graph and config.predict_node is not None:
        g = load_graph(config.predict_graph).to(device)
        w = edge_weight(g).to(device)
        logits = model(g, w)
        if config.task == "classification":
            prob = torch.sigmoid(logits[config.predict_node]).item()
            print(f"Predicted delay probability: {prob:.4f}")
        else:
            pred = logits[config.predict_node].item()
            print(f"Predicted delay (hours): {pred:.4f}")


if __name__ == "__main__":
    main()
