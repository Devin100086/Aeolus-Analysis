import argparse
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
from dgl.nn.pytorch import GraphConv
from tqdm import tqdm


DENSE_FEATURE_NAMES = [
    "O_TEMP",
    "D_TEMP",
    "O_PRCP",
    "D_PRCP",
    "O_WSPD",
    "D_WSPD",
]


@dataclass
class Config:
    graph_dir: Path
    start_date: date
    end_date: date
    output_path: Path
    hidden1_dim: int
    hidden2_dim: int
    epochs: int
    lr: float
    weight_decay: float
    neg_ratio: int
    kl_weight: float
    seed: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train VGAE for node embeddings.")
    parser.add_argument("--graph-dir", default="data/Aeolus/Flight_network/network_data_2016")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2016-12-30")
    parser.add_argument("--output-path", default="vgae/node_embedding.pth")
    parser.add_argument("--hidden1-dim", type=int, default=16)
    parser.add_argument("--hidden2-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return Config(
        graph_dir=Path(args.graph_dir),
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        output_path=Path(args.output_path),
        hidden1_dim=args.hidden1_dim,
        hidden2_dim=args.hidden2_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        neg_ratio=args.neg_ratio,
        kl_weight=args.kl_weight,
        seed=args.seed,
    )


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


def load_graph(path: Path):
    (g,), _ = load_graphs(str(path))
    return g


def ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(1)
    return tensor


def build_dense_features(g, dense_names: List[str]) -> torch.Tensor:
    tensors = []
    for name in dense_names:
        if name not in g.ndata:
            tensors.append(torch.zeros(g.num_nodes(), 1, device=g.device))
            continue
        t = torch.nan_to_num(ensure_2d(g.ndata[name]).float(), nan=0.0)
        tensors.append(t)
    if not tensors:
        raise ValueError("No dense features found in graph.")
    return torch.cat(tensors, dim=1)


def compute_dense_minmax(graph_paths: List[Path], dense_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    min_values = None
    max_values = None
    for path in tqdm(graph_paths, desc="Scanning dense min/max"):
        g = load_graph(path)
        if g.num_nodes() == 0:
            continue
        dense_feat = build_dense_features(g, dense_names)
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


class VGAEModel(nn.Module):
    def __init__(self, in_dim: int, hidden1_dim: int, hidden2_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphConv(in_dim, hidden1_dim, activation=F.relu, allow_zero_in_degree=True),
                GraphConv(hidden1_dim, hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
                GraphConv(hidden1_dim, hidden2_dim, activation=lambda x: x, allow_zero_in_degree=True),
            ]
        )

    def encode(self, g, features):
        h = self.layers[0](g, features)
        mean = self.layers[1](g, h)
        log_std = self.layers[2](g, h)
        return mean, log_std

    def reparameterize(self, mean, log_std):
        gaussian_noise = torch.randn_like(mean)
        return mean + gaussian_noise * torch.exp(log_std)

    def forward(self, g, features):
        mean, log_std = self.encode(g, features)
        z = self.reparameterize(mean, log_std)
        return z, mean, log_std


def dot_product_decode(z: torch.Tensor, edges: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    src, dst = edges
    return (z[src] * z[dst]).sum(dim=1)


def compute_loss(
    z: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    pos_edges: Tuple[torch.Tensor, torch.Tensor],
    neg_edges: Tuple[torch.Tensor, torch.Tensor],
    kl_weight: float,
) -> torch.Tensor:
    pos_score = dot_product_decode(z, pos_edges)
    neg_score = dot_product_decode(z, neg_edges)
    pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
    neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    recon_loss = pos_loss + neg_loss
    kl_loss = -0.5 * torch.mean(1 + 2 * log_std - mean.pow(2) - torch.exp(2 * log_std))
    return recon_loss + kl_weight * kl_loss


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_paths = load_graph_paths(config.graph_dir, config.start_date, config.end_date)
    if not graph_paths:
        raise ValueError("No graphs found in the specified date range.")

    sample_graph = load_graph(graph_paths[0])
    if sample_graph.num_nodes() == 0:
        raise ValueError("Sample graph has no nodes.")
    dense_names = [name for name in DENSE_FEATURE_NAMES if name in sample_graph.ndata]
    if not dense_names:
        raise ValueError("No dense features available for VGAE training.")

    dense_min, dense_max = compute_dense_minmax(graph_paths, dense_names)
    ranges = dense_max - dense_min
    ranges[ranges == 0] = 1

    model = VGAEModel(len(dense_names), config.hidden1_dim, config.hidden2_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step = 0
        progress = tqdm(graph_paths, desc=f"Epoch {epoch}/{config.epochs}")
        for path in progress:
            g = load_graph(path)
            if g.num_nodes() == 0 or g.num_edges() == 0:
                continue
            g = g.to(device)
            dense_feat = build_dense_features(g, dense_names)
            dense_feat = (dense_feat - dense_min.to(device)) / ranges.to(device)
            dense_feat = torch.nan_to_num(dense_feat, nan=0.0)

            pos_edges = g.edges()
            num_pos = g.num_edges()
            num_neg = max(1, num_pos * config.neg_ratio)
            if device.type == "cpu":
                neg_edges = dgl.sampling.global_uniform_negative_sampling(
                    g, num_samples=num_neg
                )
            else:
                neg_edges = dgl.sampling.global_uniform_negative_sampling(
                    g.to("cpu"), num_samples=num_neg
                )
                neg_edges = (neg_edges[0].to(device), neg_edges[1].to(device))

            optimizer.zero_grad(set_to_none=True)
            z, mean, log_std = model(g, dense_feat)
            loss = compute_loss(z, mean, log_std, pos_edges, neg_edges, config.kl_weight)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1
            progress.set_postfix(loss=f"{epoch_loss/max(1, step):.4f}")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.output_path)
    print(f"Saved VGAE weights to {config.output_path}")


if __name__ == "__main__":
    main()
