from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error
import igraph as ig

import graphtester as gt
from graphtester.io.dataset import Dataset
from abc_py import AbcInterface

def extract_aig(abc):
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

def _build_digraph(node_labels, edges):
    n = len(node_labels)
    edge_list = [(int(s), int(d)) for s, d in edges] if len(edges) else []
    g = ig.Graph(n=n, edges=edge_list, directed=False)
    g.vs["label"] = [str(lab) for lab in node_labels]
    return g


def build_R1(node_types, edges):
    return _build_digraph(node_types, edges)

def load(aig: str):
    # TODO: path should be argument

    """Load a dataset. Returns (X, Y_all, y_min) or raises FileNotFoundError."""
    p = DATASET_ROOT / f"dataset-{aig}-all-actions-False-mc-simu-100" / "res.npz"
    data = np.load(p)
    X = data["x"]
    Y_all = data["y"]
    y_min = np.amin(Y_all, axis=1)
    return X, Y_all, y_min


def aig_path(aig: str, i: int) -> Path:
    # TODO: path should be argument
    return DATASET_ROOT / f"dataset-{aig}-all-actions-False-mc-simu-100" / f"{i}.aig"


def load_raw_aigs(aig: str):
    """Return list of (node_types, edges) per graph, cached on disk."""
    man = AbcInterface(); man.end(); man.start()
    types_list, edges_list = [], []
    for i in range(1000):
        man.read(str(aig_path(aig, i)))
        t, e = extract_aig(man)
        types_list.append(t); edges_list.append(e)
    return list(zip(types_list, edges_list))


REPETS = 10
N_SUB = 100
# Increase iters, and increase n-wl test
WL_ITERS = 2
N_PERM_AGG = 10

AIGS = [
    "i10", "apex1", "dalu",
    "k2", "bc0", "mainpla",
]

DATASET_ROOT = Path("../../sb3-abc")

# Discover which datasets are actually available on disk.
available = []
for a in AIGS:
    try:
        load(a)
        available.append(a)
    except FileNotFoundError:
        print(f"[skip] {a}: dataset missing")
print("Available datasets:", available)



for aig in available:
    _, _, y = load(aig)
    raw = load_raw_aigs(aig)
    y_var = np.var(y)
    graphs = [build_R1(t, e) for (t, e) in raw]
    ds = Dataset(graphs=graphs, labels=y.tolist(), name='Default extraction')
    ev = gt.evaluate(
        ds,
        ignore_node_features=False,
        ignore_edge_features=True,
        metrics=["lower_bound_mse"],
        iterations=WL_ITERS,
    )
    curve_aig = ev.as_dataframe()["Lower Bound MSE"].to_numpy()/max(1, y_var) #TODO: check for too small variance
    print(curve_aig)
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
    np.save(f'res/{aig}_curve.npy', curve_aig)
    np.save(f'res/{aig}_perm_curves.npy', np.asarray(perm_lbs))
