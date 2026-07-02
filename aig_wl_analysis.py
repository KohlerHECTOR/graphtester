import argparse
from pathlib import Path
from typing import cast
import numpy as np
import igraph as ig
from joblib import Parallel, delayed

import graphtester as gt
from graphtester.io.dataset import Dataset
from igraph import Graph
from abc_py import AbcInterface

def extract_aig(abc) -> tuple[np.ndarray, np.ndarray]:
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

def build_digraph(node_labels, edges) -> Graph:
    n = len(node_labels)
    edge_list = [(int(s), int(d)) for s, d in edges] if len(edges) else []
    g = ig.Graph(n=n, edges=edge_list, directed=False)
    g.vs["label"] = [str(lab) for lab in node_labels]
    return g

def load(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset. Returns (X, Y_all, y_min) or raises FileNotFoundError."""
    p = dataset_path / "res.npz"
    data = np.load(p)
    X = data["x"]
    Y_all = data["y"]
    y_min = np.amin(Y_all, axis=1)
    return X, Y_all, y_min


def aig_path(dataset_path: Path, i: int) -> Path:
    return dataset_path / f"{i}.aig"


def _load_some_aigs(paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    man = AbcInterface()
    man.end()
    man.start()
    to_ret = []
    for path in paths:
        man.read(str(path))
        to_ret.append(extract_aig(man))
    return to_ret

def load_raw_aigs(dataset_path: Path, n_jobs: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (node_types, edges) per graph, loaded in parallel."""
    chunk_size = (1000 + n_jobs - 1) // n_jobs
    chunks = [
        [aig_path(dataset_path, j) for j in range(i, min(i + chunk_size, 1000))]
        for i in range(0, 1000, chunk_size)
    ]
    nested = cast(list[list[tuple[np.ndarray, np.ndarray]]], Parallel(n_jobs=n_jobs)(delayed(_load_some_aigs)(chunk) for chunk in chunks))
    return [item for sublist in nested for item in sublist]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WL analysis on an AIG dataset.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset directory.")
    parser.add_argument("--wl-iters", type=int, default=2, help="Number of WL iterations (default: 2).")
    parser.add_argument("--n-perm-agg", type=int, default=2, help="Number of permutation aggregations (default: 10).")
    parser.add_argument("--n-sub", type=int, default=10, help="Number of subsamples (default: 100).")
    parser.add_argument("--n-cpu", type=int, default=14, help="Number of parallel workers for loading AIGs (default: 1).")
    args = parser.parse_args()

    WL_ITERS = args.wl_iters
    N_PERM_AGG = args.n_perm_agg
    N_SUB = args.n_sub

    dataset_path = args.dataset_path.resolve()
    aig_name = dataset_path.name
    _, _, y = load(dataset_path)
    raws = load_raw_aigs(dataset_path, n_jobs=args.n_cpu)
    y_var = np.var(y)
    graphs = cast(list[Graph], Parallel(n_jobs=args.n_cpu)(delayed(build_digraph)(t, e) for t, e in raws))
    ds = Dataset(graphs=graphs, labels=y.tolist(), name='Default extraction')
    ev = gt.evaluate(
        ds,
        ignore_node_features=False,
        ignore_edge_features=True,
        metrics=["lower_bound_mse"],
        iterations=WL_ITERS,
    )
    curve_aig = ev.as_dataframe()["Lower Bound MSE"].to_numpy() / max(1, y_var)  # TODO: check for too small variance
    perm_lbs = []
    for _ in range(N_PERM_AGG):
        y_perm = np.random.permutation(y)
        ds_perm = Dataset(graphs=graphs, labels=y_perm.tolist(), name="R1_aggregate_perm")
        ev_perm = gt.evaluate(
            ds_perm,
            ignore_node_features=False,
            ignore_edge_features=True,
            metrics=["lower_bound_mse"],
            iterations=WL_ITERS,
        )
        perm_lbs.append(ev_perm.as_dataframe()["Lower Bound MSE"].to_numpy() / max(1, y_var))
    np.save(dataset_path / f"{aig_name}_curve.npy", curve_aig)
    np.save(dataset_path / f"{aig_name}_perm_curves.npy", np.asarray(perm_lbs))
