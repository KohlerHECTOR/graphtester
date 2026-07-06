import argparse
import pickle
from pathlib import Path
from typing import cast
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

from abc_py import AbcInterface


def _evaluate_mse(hashes: dict[int, list[str]], labels: list[float]) -> dict[int, float]:
    """MSE of the optimal per-hash-group mean predictor, at each iteration k.

    Graphs sharing a 1-WL hash are grouped; each group predicts its mean label, and
    the MSE of that prediction is returned per k. This reproduces the grouping-based
    lower bound previously imported from graphtester.
    """
    y = np.asarray(labels, dtype=float)
    mse = {}
    for k, hs in hashes.items():
        groups: dict = {}
        for idx, h in enumerate(hs):
            groups.setdefault(h, []).append(idx)
        pred = np.empty(len(y))
        for idxs in groups.values():
            pred[idxs] = y[idxs].mean()
        mse[k] = float(np.mean((y - pred) ** 2))
    return mse

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

def build_nx_graph(node_labels, edges, directed: bool) -> nx.Graph:
    """Build a networkx graph with node-type labels.

    ``directed=False`` returns an ``nx.Graph`` (orientation ignored). ``directed=True``
    returns an ``nx.DiGraph`` whose edges point ``node -> fanin``: networkx's
    Weisfeiler-Lehman hash aggregates each node's *successors*, so reversing the
    ``fanin -> node`` edges makes WL refine every node by its fanins, matching the
    AIG's fanin -> node information flow.
    """
    g: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    for i, lab in enumerate(node_labels):
        g.add_node(i, label=str(lab))
    g.add_edges_from((int(s), int(d)) for s, d in edges)
    if directed:
        g = cast(nx.DiGraph, g).reverse(copy=True)
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
        print(f"{len(to_ret)}/{len(paths)}", end="\r")
    man.end()
    return to_ret

def _eval_perm(hashes: dict[int, list[str]], perm_labels: list[float], y_var: float) -> np.ndarray:
    d = _evaluate_mse(hashes, perm_labels)
    return np.array([d[k] for k in sorted(d)]) / max(1, y_var)


def _hash_chunk(graphs: list[nx.Graph], iterations: int) -> dict[int, list[str]]:
    """1-WL hashes for a chunk of graphs at each k = 1..iterations."""
    out: dict[int, list[str]] = {k: [] for k in range(1, iterations + 1)}
    for ng in graphs:
        for k in range(1, iterations + 1):
            out[k].append(
                nx.weisfeiler_lehman_graph_hash(ng, node_attr="label", iterations=k)
            )
    return out


def _nx_hashes_at_k_iterations(
    graphs: list[nx.Graph], iterations: int, n_jobs: int = 1
) -> dict[int, list[str]]:
    """Graph-level 1-WL hashes at each k = 1..iterations using networkx.

    Returns ``{k: [hash_for_graph_0, ...]}`` matching the shape consumed by
    ``_evaluate_mse``. networkx's ``weisfeiler_lehman_graph_hash`` uses a
    deterministic blake2b digest, so hashes are stable across processes; the graphs
    are split into ``n_jobs`` chunks hashed in parallel and re-concatenated in order.
    Works for both ``nx.Graph`` and ``nx.DiGraph`` inputs.
    """
    n = len(graphs)
    if n == 0:
        return {k: [] for k in range(1, iterations + 1)}

    n_jobs = max(1, min(n_jobs, n))
    chunk_size = (n + n_jobs - 1) // n_jobs
    chunks = [graphs[i:i + chunk_size] for i in range(0, n, chunk_size)]

    partials = cast(
        list[dict[int, list[str]]],
        Parallel(n_jobs=n_jobs)(delayed(_hash_chunk)(chunk, iterations) for chunk in chunks),
    )

    hashes: dict[int, list[str]] = {k: [] for k in range(1, iterations + 1)}
    for part in partials:
        for k in range(1, iterations + 1):
            hashes[k].extend(part[k])
    return hashes


def parallel_evaluate(
    graphs: list[nx.Graph],
    y: np.ndarray,
    y_var: np.float32,
    wl_iters: int,
    n_jobs: int,
    hashes_save_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hash the graphs (1-WL, networkx, parallel over chunks) then evaluate MSE.

    The permutation null baseline uses a fixed ``n_perms`` label shuffles that is
    independent of ``n_jobs`` -- workers only control parallelism, not the number of
    permutations -- so the baseline has the same statistical power on every dataset
    regardless of the machine it runs on.

    Returns
    -------
    curve : np.ndarray  (wl_iters,)              # k = 1..wl_iters
    perm_curves : np.ndarray  (n_perms, wl_iters)
    """
    hashes = _nx_hashes_at_k_iterations(list(graphs), wl_iters, n_jobs=n_jobs)

    if hashes_save_path is not None:
        with open(hashes_save_path, "wb") as f:
            pickle.dump(hashes, f)

    labels = y.tolist()
    mse_dict = _evaluate_mse(hashes, labels)
    curve = np.array([mse_dict[k] for k in sorted(mse_dict)]) / max(1, y_var)

    rng = np.random.default_rng(42)
    perms = [rng.permutation(y).tolist() for _ in range(n_jobs)]
    perm_curves = cast(
        list[np.ndarray],
        Parallel(n_jobs=n_jobs)(delayed(_eval_perm)(hashes, perm, y_var) for perm in perms),
    )
    return curve, np.asarray(perm_curves)


def load_raw_aigs(dataset_path: Path, n_graphs: int, n_jobs: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (node_types, edges) for graphs ``0..n_graphs-1``, loaded in parallel."""
    chunk_size = max(1, (n_graphs + n_jobs - 1) // n_jobs)
    chunks = [
        [aig_path(dataset_path, j) for j in range(i, min(i + chunk_size, n_graphs))]
        for i in range(0, n_graphs, chunk_size)
    ]
    nested = cast(list[list[tuple[np.ndarray, np.ndarray]]], Parallel(n_jobs=n_jobs)(delayed(_load_some_aigs)(chunk) for chunk in chunks))
    return [item for sublist in nested for item in sublist]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WL analysis on an AIG dataset.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset directory.")
    parser.add_argument("--wl-iters", type=int, default=5, help="Number of WL iterations (default: 5).")
    parser.add_argument("--n-cpu", type=int, default=25, help="Number of parallel workers (default: 25).")
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    aig_name = dataset_path.name
    if ('mainpla' in aig_name) or ('C6288' in aig_name):
        args.n_cpu = 8
    _, _, y = load(dataset_path)
    raws = load_raw_aigs(dataset_path, n_graphs=len(y), n_jobs=args.n_cpu)
    y_var = np.var(y, dtype=np.float32)
    for directed, suffix in [(False, "undirected"), (True, "directed")]:
        graphs = cast(
            list[nx.Graph],
            Parallel(n_jobs=args.n_cpu)(
                delayed(build_nx_graph)(t, e, directed) for t, e in raws
            ),
        )
        curve_aig, perm_lbs = parallel_evaluate(
            graphs, y, y_var,
            wl_iters=args.wl_iters,
            n_jobs=args.n_cpu,
            hashes_save_path=f"{aig_name}_hashes_{suffix}.pkl",
        )
        print(f"curve ({suffix}):", curve_aig)
        np.save(f"{aig_name}_curve_{suffix}.npy", curve_aig)
        np.save(f"{aig_name}_perm_curves_{suffix}.npy", perm_lbs)
