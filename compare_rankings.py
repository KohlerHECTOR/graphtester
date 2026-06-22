"""Compare rankings of graph-state y values produced by several predictors.

All predictors are compared against the *perfect* predictor, which ranks graph
states directly by the loaded y value.

Predictors
----------
* "0-layer": a 0-layer MPNN with mean readout cannot use any structural
    information; its prediction depends only on the number of nodes. We model it
    as the mean y over all graphs that share the same node count.
* "1-layer": a perfect 1-layer MPNN can only distinguish graphs up to their
    1-WL colour refinement after a single message-passing step. We reuse
    graphtester's 1-WL hashing (``_estimate_hashes_at_k_iterations``) to group
    graphs that are indistinguishable after k=1 iterations, then predict the
    per-group mean y. This is exactly the optimal predictor whose MSE is the
    ``lower_bound_mse`` reported by ``graphtester`` at 1 layer.
* "x-features": groups graphs whose x feature vectors agree up to a tolerance and
    predicts the per-group mean y -- the optimal predictor that only sees the x
    features. Reported for several tolerances (1e-1 down to 1e-5).

Results are reported per AIG family and for the aggregate, using Spearman rho and
Kendall tau.
"""

from pathlib import Path

import igraph as ig
import numpy as np
from scipy.stats import kendalltau, spearmanr

from abc_py import AbcInterface
from graphtester.evaluate.dataset import _estimate_hashes_at_k_iterations

DATASET_ROOT = Path("../../sb3-abc")

WL_ITERS = 1  # number of message-passing layers for the WL-based predictor


AIGS = [
    "i10", "apex1", "C1355", "C6288", "dalu",
    "k2", "bc0", "C5315", "C7552", "mainpla",
]


def load(aig: str):
    """Load a dataset. Returns (x, y_min) or raises FileNotFoundError."""
    p = DATASET_ROOT / f"dataset-{aig}" / "res.npz"
    data = np.load(p)
    x_all = data["x"]
    y_all = data["y"]
    return x_all, np.amin(y_all, axis=1)


def aig_path(aig: str, i: int) -> Path:
    return DATASET_ROOT / f"dataset-{aig}" / f"{i}.aig"


def extract_aig(abc):
    """Extract node types and edges from the current AIG in the ABC manager."""
    n = abc.numNodes()
    node_types = np.zeros(n, dtype=int)
    edges = []
    for i in range(n):
        node = abc.aigNode(i)
        node_types[i] = node.nodeType()
        if node.hasFanin0():
            edges.append((node.fanin0(), i))
        if node.hasFanin1():
            edges.append((node.fanin1(), i))
    edges = np.asarray(edges, dtype=int) if edges else np.zeros((0, 2), dtype=int)
    return node_types, edges


def _build_graph(node_labels, edges) -> ig.Graph:
    """Build an undirected igraph graph with node-type labels (matches R1 extraction)."""
    n = len(node_labels)
    edge_list = [(int(s), int(d)) for s, d in edges] if len(edges) else []
    g = ig.Graph(n=n, edges=edge_list, directed=False)
    g.vs["label"] = [str(lab) for lab in node_labels]
    return g


def load_graphs(aig: str, n_graphs: int):
    """Return the igraph graphs for the first ``n_graphs`` AIGs of a family."""
    man = AbcInterface()
    man.end()
    man.start()
    graphs = []
    for i in range(n_graphs):
        man.read(str(aig_path(aig, i)))
        stats = man.aigStats()
        node_types, edges = extract_aig(man)
        g = _build_graph(node_types, edges)
        g["depth"] = int(stats.lev)
        graphs.append(g)
    return graphs


def zero_layer_prediction(y: np.ndarray, node_counts: np.ndarray) -> np.ndarray:
    """Mean y over all graphs sharing the same node count (0-layer MPNN)."""
    pred = np.empty_like(y, dtype=float)
    for size in np.unique(node_counts):
        mask = node_counts == size
        pred[mask] = y[mask].mean()
    return pred


def size_depth_prediction(
    y: np.ndarray, node_counts: np.ndarray, depths: np.ndarray
) -> np.ndarray:
    """Mean y over all graphs sharing the same (node count, depth) pair.

    A graph-level baseline that uses two cheap global descriptors -- the number
    of nodes and the AIG logic depth -- without any message passing.
    """
    pred = np.empty_like(y, dtype=float)
    keys = np.stack([node_counts, depths], axis=1)
    for key in np.unique(keys, axis=0):
        mask = (node_counts == key[0]) & (depths == key[1])
        pred[mask] = y[mask].mean()
    return pred


def x_feature_prediction(x: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    """Mean y over all graphs whose x feature vectors agree up to ``tol``.

    Like the other baselines, this is a grouping predictor: x is snapped to a grid
    of spacing ``tol`` so that near-identical feature vectors fall in the same
    group, and the optimal prediction for a group is its mean y. It is the best
    any model that only sees the x features (at resolution ``tol``) can do.
    """
    pred = np.empty_like(y, dtype=float)
    snapped = np.round(x / tol)
    _, inverse = np.unique(snapped, axis=0, return_inverse=True)
    for group in np.unique(inverse):
        mask = inverse == group
        pred[mask] = y[mask].mean()
    return pred


def wl_layer_prediction(graphs, y: np.ndarray, k: int) -> np.ndarray:
    """Perfect k-layer MPNN prediction via graphtester's 1-WL hashing.

    Graphs sharing the same 1-WL hash after ``k`` iterations are indistinguishable
    to a k-layer MPNN; the optimal prediction is the per-group mean of y.
    """
    hashes, _ = _estimate_hashes_at_k_iterations(list(graphs), iterations=k)
    hashes_at_k = hashes[k]
    groups: dict = {}
    for hsh, val in zip(hashes_at_k, y):
        groups.setdefault(hsh, []).append(val)
    means = {hsh: float(np.mean(vals)) for hsh, vals in groups.items()}
    return np.array([means[hsh] for hsh in hashes_at_k])


def rank_metrics(perfect: np.ndarray, pred: np.ndarray):
    """Rank association between the true order and a predictor's order."""
    rho, _ = spearmanr(perfect, pred)
    tau, _ = kendalltau(perfect, pred)
    return rho, tau


def nmse(y: np.ndarray, pred: np.ndarray) -> float:
    """MSE normalized by max(1, var(y)), matching the plotted curves."""
    return float(np.mean((y - pred) ** 2) / max(1.0, np.var(y)))


def _print_row(name, perfect, pred):
    rho, tau = rank_metrics(perfect, pred)
    print(f"  {name:<10} {rho:>10.4f} {tau:>10.4f} {nmse(perfect, pred):>10.4g}")


HEADER = f"  {'predictor':<10} {'Spearman':>10} {'Kendall':>10} {'NMSE':>10}"


def main():
    available = []
    for aig in AIGS:
        try:
            load(aig)
            available.append(aig)
        except FileNotFoundError:
            print(f"[skip] {aig}: dataset missing")

    mse_results: dict = {}
    rank_results: dict = {}
    all_y, all_counts, all_depths, all_one_layer, all_x = [], [], [], [], []
    for aig in available:
        x, y = load(aig)
        y = y.astype(float)
        graphs = load_graphs(aig, len(y))
        counts = np.array([g.vcount() for g in graphs])
        depths = np.array([g["depth"] for g in graphs])

        zero = zero_layer_prediction(y, counts)
        size_depth = size_depth_prediction(y, counts, depths)
        one = wl_layer_prediction(graphs, y, WL_ITERS)
        xfeat = x_feature_prediction(x, y, 1e-100)

        print(f"\n{aig}")
        print(HEADER)
        for name, pred in [("0-layer", zero), ("size+depth", size_depth), ("x", xfeat), ("1-layer", one)]:
            _print_row(name, y, pred)
            mse_results[f"{aig}|{name}"] = nmse(y, pred)
            rho, tau = rank_metrics(y, pred)
            rank_results[f"{aig}|{name}|rho"] = rho
            rank_results[f"{aig}|{name}|tau"] = tau

        all_y.append(y)
        all_counts.append(counts)
        all_depths.append(depths)
        all_one_layer.append(one)
        all_x.append(x)

    if all_y:
        y = np.concatenate(all_y)
        counts = np.concatenate(all_counts)
        depths = np.concatenate(all_depths)
        x = np.concatenate(all_x)
        zero = zero_layer_prediction(y, counts)
        size_depth = size_depth_prediction(y, counts, depths)
        one = np.concatenate(all_one_layer)

        print("\naggregate")
        print(HEADER)
        xfeat = x_feature_prediction(x, y, 1e-100)
        for name, pred in [("0-layer", zero), ("size+depth", size_depth), ("x", xfeat), ("1-layer", one)]:
            _print_row(name, y, pred)
            mse_results[f"aggregate|{name}"] = nmse(y, pred)
            rho, tau = rank_metrics(y, pred)
            rank_results[f"aggregate|{name}|rho"] = rho
            rank_results[f"aggregate|{name}|tau"] = tau

        np.savez("res/predictor_mse.npz", **mse_results)
        np.savez("res/predictor_ranks.npz", **rank_results)
        print("\nSaved predictor MSEs to res/predictor_mse.npz")
        print("Saved predictor rank metrics to res/predictor_ranks.npz")


if __name__ == "__main__":
    main()
