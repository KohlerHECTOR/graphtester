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

def extract_aig_with_edges(aig_filename, gate_to_index={'PI':0, 'AND':1, 'PO':2}):
    with open(aig_filename, "rb") as fh:
        raw = fh.read()

    nl = raw.index(b"\n")
    header = raw[:nl].decode("ascii").split()
    fmt = header[0]
    n_variables, n_inputs, n_latches, n_outputs, n_and = (int(v) for v in header[1:6])
    assert n_latches == 0, "The AIG has latches."

    if fmt == "aag":
        lines = raw.decode("ascii").split("\n")
    else:
        assert fmt == "aig", "Unknown AIG format: {}".format(fmt)
        body = raw[nl + 1:]
        pos = 0

        outputs = []
        for _ in range(n_outputs):
            end = body.index(b"\n", pos)
            outputs.append(body[pos:end].decode("ascii"))
            pos = end + 1

        def _decode():
            nonlocal pos
            value = 0
            shift = 0
            while True:
                ch = body[pos]
                pos += 1
                value |= (ch & 0x7F) << shift
                if not (ch & 0x80):
                    return value
                shift += 7

        ands = []
        for idx in range(n_and):
            delta0 = _decode()
            delta1 = _decode()
            lhs = 2 * (n_inputs + n_latches + 1 + idx)
            rhs0 = lhs - delta0
            rhs1 = rhs0 - delta1
            ands.append("{} {} {}".format(lhs, rhs0, rhs1))

        lines = ["aag {} {} {} {} {}".format(n_variables, n_inputs, n_latches, n_outputs, n_and)]
        lines += [str(2 * (i + 1)) for i in range(n_inputs)]
        lines += outputs
        lines += ands

    x_data = []
    edge_index = []
    edge_attr = []
    for _ in range(n_inputs):
        x_data.append([len(x_data), gate_to_index["PI"]])
    for _ in range(n_and):
        x_data.append([len(x_data), gate_to_index["AND"]])

    for line in lines[1 + n_inputs + n_outputs:]:
        arr = line.replace("\n", "").split(" ")
        if len(arr) != 3:
            continue
        and_index = int(int(arr[0]) / 2) - 1
        fanin_1_index = int(int(arr[1]) / 2) - 1
        fanin_2_index = int(int(arr[2]) / 2) - 1
        fanin_1_not = int(arr[1]) % 2
        fanin_2_not = int(arr[2]) % 2
        if fanin_1_index >= 0:
            edge_index.append([fanin_1_index, and_index])
            edge_attr.append(fanin_1_not)
        if fanin_2_index >= 0:
            edge_index.append([fanin_2_index, and_index])
            edge_attr.append(fanin_2_not)

    for line in lines[1 + n_inputs: 1 + n_inputs + n_outputs]:
        arr = line.replace("\n", "").split(" ")
        if len(arr) != 1:
            continue
        po_fanin_index = int(int(arr[0]) / 2) - 1
        po_not = int(arr[0]) % 2
        po_index = len(x_data)
        x_data.append([po_index, gate_to_index["PO"]])
        if po_fanin_index >= 0:
            edge_index.append([po_fanin_index, po_index])
            edge_attr.append(po_not)

    node_types_arr = np.array([row[1] for row in x_data], dtype=int)
    edges_arr = np.array(edge_index, dtype=int) if edge_index else np.zeros((0, 2), dtype=int)
    edge_attr_arr = np.array(edge_attr, dtype=int) if edge_attr else np.zeros(0, dtype=int)
    return node_types_arr, edges_arr, edge_attr_arr

def build_nx_graph(node_labels, edges, directed: bool, edge_attr: np.ndarray | None = None) -> nx.Graph:
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
    if edge_attr is not None:
        for (s, d), a in zip(edges, edge_attr):
            g.add_edge(int(s), int(d), attr=str(a))
    else:
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


def _hash_chunk(
    raws_chunk: list[tuple], directed: bool, iterations: int
) -> dict[int, list[str]]:
    """1-WL hashes for a chunk of graphs at each k = 1..iterations.

    Each networkx graph is built from its compact ``(node_types, edges)`` arrays,
    hashed, and immediately discarded, so a worker only ever holds one graph in
    memory at a time. Only the (tiny) hash strings cross the process boundary -- the
    heavy networkx objects are never returned to the parent.
    """
    out: dict[int, list[str]] = {k: [] for k in range(1, iterations + 1)}
    for raw in raws_chunk:
        if len(raw) == 3:
            node_types, edges, ea = raw
        else:
            node_types, edges = raw
            ea = None
        ng = build_nx_graph(node_types, edges, directed, ea)
        edge_attr_key = "attr" if ea is not None else None
        for k in range(1, iterations + 1):
            out[k].append(
                nx.weisfeiler_lehman_graph_hash(ng, node_attr="label", edge_attr=edge_attr_key, iterations=k)
            )
        del ng
    return out


def _nx_hashes_from_raws(
    raws: list[tuple[np.ndarray, np.ndarray]],
    directed: bool,
    iterations: int,
    n_jobs: int = 1,
) -> dict[int, list[str]]:
    """Graph-level 1-WL hashes at each k = 1..iterations using networkx.

    Returns ``{k: [hash_for_graph_0, ...]}`` matching the shape consumed by
    ``_evaluate_mse``. The compact ``(node_types, edges)`` arrays are split into
    ``n_jobs`` chunks; each worker builds, hashes, and discards its graphs one at a
    time (see ``_hash_chunk``), so peak memory stays flat in the dataset size and no
    networkx graph is ever held in the parent or shipped between processes.
    networkx's ``weisfeiler_lehman_graph_hash`` uses a deterministic blake2b digest,
    so hashes are stable across processes and chunks re-concatenate in order.
    """
    n = len(raws)
    if n == 0:
        return {k: [] for k in range(1, iterations + 1)}

    n_jobs = max(1, min(n_jobs, n))
    chunk_size = (n + n_jobs - 1) // n_jobs
    chunks = [raws[i:i + chunk_size] for i in range(0, n, chunk_size)]

    partials = cast(
        list[dict[int, list[str]]],
        Parallel(n_jobs=n_jobs)(
            delayed(_hash_chunk)(chunk, directed, iterations) for chunk in chunks
        ),
    )

    hashes: dict[int, list[str]] = {k: [] for k in range(1, iterations + 1)}
    for part in partials:
        for k in range(1, iterations + 1):
            hashes[k].extend(part[k])
    return hashes


def parallel_evaluate(
    raws: list[tuple[np.ndarray, np.ndarray]],
    directed: bool,
    y: np.ndarray,
    y_var: np.float32,
    wl_iters: int,
    n_jobs: int,
    hashes_save_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Hash the graphs (1-WL, networkx, parallel over chunks) then evaluate MSE.

    Graphs are built from their compact ``(node_types, edges)`` arrays and hashed
    inside the workers, one at a time, so no networkx object is held in the parent or
    shipped between processes -- this keeps peak memory flat in the dataset size.

    Returns
    -------
    curve : np.ndarray  (wl_iters,)              # k = 1..wl_iters
    perm_curves : np.ndarray  (n_perm, wl_iters)
    """
    hashes = _nx_hashes_from_raws(raws, directed, wl_iters, n_jobs=n_jobs)

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


def _load_some_aigs_with_edges(paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [extract_aig_with_edges(p) for p in paths]


def load_raw_aigs_with_edges(dataset_path: Path, n_graphs: int, n_jobs: int = 1) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return list of (node_types, edges, edge_attr) for graphs ``0..n_graphs-1``, loaded in parallel."""
    chunk_size = max(1, (n_graphs + n_jobs - 1) // n_jobs)
    chunks = [
        [aig_path(dataset_path, j) for j in range(i, min(i + chunk_size, n_graphs))]
        for i in range(0, n_graphs, chunk_size)
    ]
    nested = cast(list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]], Parallel(n_jobs=n_jobs)(delayed(_load_some_aigs_with_edges)(chunk) for chunk in chunks))
    return [item for sublist in nested for item in sublist]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WL analysis on an AIG dataset.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset directory.")
    parser.add_argument("--wl-iters", type=int, default=50, help="Number of WL iterations (default: 5).")
    parser.add_argument("--n-cpu", type=int, default=25, help="Number of parallel workers (default: 25).")
    parser.add_argument("--use-edges", action="store_true", help="Include edge attributes (complement bits) in WL hashing.")
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    aig_name = dataset_path.name
    if ('mainpla' in aig_name) or ('C6288' in aig_name):
        args.n_cpu = 8
    _, _, y = load(dataset_path)
    prefix = ''
    if args.use_edges:
        raws = load_raw_aigs_with_edges(dataset_path, n_graphs=len(y), n_jobs=args.n_cpu)
        prefix = 'w_edges_'
    else:
        raws = load_raw_aigs(dataset_path, n_graphs=len(y), n_jobs=args.n_cpu)
    y_var = np.var(y, dtype=np.float32)
    for directed, suffix in [(False, "undirected"), (True, "directed")]:
        curve_aig, perm_lbs = parallel_evaluate(
            raws, directed, y, y_var,
            wl_iters=args.wl_iters,
            n_jobs=args.n_cpu,
            hashes_save_path=prefix + f"{aig_name}_hashes_{suffix}.pkl",
        )
        print(f"curve ({suffix}):", curve_aig)
        np.save(prefix + f"{aig_name}_curve_{suffix}.npy", curve_aig)
        np.save(prefix + f"{aig_name}_perm_curves_{suffix}.npy", perm_lbs)
