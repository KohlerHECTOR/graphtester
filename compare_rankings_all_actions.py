"""Compare rankings of graph-state y values produced by several predictors.

All predictors are compared against the *perfect* predictor, which ranks graph
states directly by the loaded y value.

Predictors
----------
* "0-layer": a 0-layer MPNN with mean readout cannot use any structural
    information; its prediction depends only on the number of nodes. We model it
    as the mean y over all graphs that share the same node count.
* "1-layer": a perfect 1-layer MPNN can only distinguish graphs up to their
    1-WL colour refinement after a single message-passing step. We use networkx's
    ``weisfeiler_lehman_graph_hash`` to group graphs that are indistinguishable after
    k iterations, then predict the per-group mean y. Two variants are reported: one
    treating the AIG as undirected (``nx.Graph``) and one preserving edge direction
    (``nx.DiGraph``).
* "x-features": groups graphs whose x feature vectors agree up to a tolerance and
    predicts the per-group mean y -- the optimal predictor that only sees the x
    features. Reported for several tolerances (1e-1 down to 1e-5).

Results are reported per AIG family and for the aggregate, using Spearman rho and
Kendall tau.
"""

import pickle
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kendalltau, spearmanr, weightedtau

from abc_py import AbcInterface

WL_ITERS = 50  # number of message-passing layers for the WL-based predictor
N_PERMS = 25  # permutations for the per-predictor ranking null distribution
SEEDS = list(range(10))


AIGS = [
    "i10", "apex1", "dalu", "C6288", "C1355", "C5315", "C7552",
    "k2", "bc0", "mainpla"
]


def load(aig: str, seed: int):
    # TODO: path should be argument

    """Load a dataset for a given seed. Returns (x, y_min) or raises FileNotFoundError."""
    mc = 5 if aig in ('mainpla', 'C6288') else 10
    p =f"RES-AIGS/dataset-{aig}-all-actions-True-mc-simu-{mc}-seed-{seed}-rs-False/res.npz"
    
    data = np.load(p)
    x_all = data["x"]
    y_all = data["y"]
    return x_all, np.amin(y_all, axis=1)


def aig_path(aig: str, seed: int, i: int) -> str:
    # TODO: path should be argument
    mc = 5 if aig in ('mainpla', 'C6288') else 10
    return f"RES-AIGS/dataset-{aig}-all-actions-True-mc-simu-{mc}-seed-{seed}-rs-False/{i}.aig"


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


def load_graphs(aig: str, seed: int, n_graphs: int):
    """Return per-graph ``(node_types, edges, depth)`` for the first ``n_graphs`` AIGs."""
    man = AbcInterface()
    man.end()
    man.start()
    graphs = []
    for i in range(n_graphs):
        man.read(str(aig_path(aig, seed, i)))
        stats = man.aigStats()
        node_types, edges = extract_aig(man)
        graphs.append((node_types, edges, int(stats.lev)))
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
    print(x.shape)
    snapped = np.round(x / tol)
    _, inverse = np.unique(snapped, axis=0, return_inverse=True)
    for group in np.unique(inverse):
        mask = inverse == group
        pred[mask] = y[mask].mean()
    return pred


def load_wl_hashes(aig: str, seed: int, directed: bool, k: int, w_edges: bool = False) -> list[str]:
    """Load precomputed 1-WL graph hashes at iteration ``k`` from the pickle written
    by ``aig_wl_analysis.py`` (``{dataset}_hashes_{undirected|directed}.pkl``).

    The pickle holds ``{iteration: [hash_for_graph_0, ...]}`` in graph index order.
    Set ``w_edges=True`` to load hashes from the weighted-edges variant.
    """
    suffix = "directed" if directed else "undirected"
    dataset_prefix = "w_edges_dataset" if w_edges else "dataset"
    mc = 5 if aig in ('mainpla', 'C6288') else 10
    prefix = f"RES-AIGS/{dataset_prefix}-{aig}-all-actions-True-mc-simu-{mc}-seed-{seed}-rs-False"
    with open(f"{prefix}_hashes_{suffix}.pkl", "rb") as f:
        hashes = pickle.load(f)
    return hashes[k]


def wl_layer_prediction(hashes_at_k, y: np.ndarray) -> np.ndarray:
    """Perfect k-layer MPNN prediction from precomputed 1-WL graph hashes.

    Graphs sharing the same 1-WL hash after ``k`` iterations are indistinguishable
    to a k-layer MPNN; the optimal prediction is the per-group mean of y. Works for
    hashes computed on either ``nx.Graph`` or ``nx.DiGraph`` inputs.
    """
    hashes_at_k = list(hashes_at_k)[: len(y)]
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


HEADER = f"  {'predictor':<10} {'Spearman':>10} {'Kendall':>10} {'NMSE':>10}"


def _process_seed(aig: str, seed: int) -> dict | None:
    """Run all predictors for one (aig, seed) pair. Returns a flat results dict."""
    try:
        x, y = load(aig, seed)
    except FileNotFoundError:
        print(f"[skip] {aig} seed {seed}: missing")
        return None
    y = y.astype(float)
    raw = load_graphs(aig, seed, len(y))
    counts = np.array([len(nt) for nt, _e, _d in raw])
    depths = np.array([d for _nt, _e, d in raw])

    zero = zero_layer_prediction(y, counts)
    size_depth = size_depth_prediction(y, counts, depths)
    xfeat = x_feature_prediction(x, y, 1e-8)
    hashes_undir = load_wl_hashes(aig, seed, False, WL_ITERS)
    hashes_dir = load_wl_hashes(aig, seed, True, WL_ITERS)
    wl_undir = wl_layer_prediction(hashes_undir, y)
    wl_dir = wl_layer_prediction(hashes_dir, y)
    if aig != 'mainpla':
        hashes_we_undir = load_wl_hashes(aig, seed, False, WL_ITERS, w_edges=True)
        hashes_we_dir = load_wl_hashes(aig, seed, True, WL_ITERS, w_edges=True)
        wl_we_undir = wl_layer_prediction(hashes_we_undir, y)
        wl_we_dir = wl_layer_prediction(hashes_we_dir, y)
    
        result = {}
        rng = np.random.default_rng(seed)
        for name, pred, pred_func in [
            ("0-layer",       zero,     lambda yp: zero_layer_prediction(yp, counts)),
            ("size+depth",    size_depth, lambda yp: size_depth_prediction(yp, counts, depths)),
            ("x",             xfeat,    lambda yp: x_feature_prediction(x, yp, 1e-8)),
            ("wl-undirected", wl_undir, lambda yp, h=hashes_undir: wl_layer_prediction(h, yp)),
            ("wl-directed",   wl_dir,   lambda yp, h=hashes_dir:   wl_layer_prediction(h, yp)),
            ("wl-we-undirected", wl_we_undir, lambda yp, h=hashes_we_undir: wl_layer_prediction(h, yp)),
            ("wl-we-directed",   wl_we_dir,   lambda yp, h=hashes_we_dir:   wl_layer_prediction(h, yp)),
        ]:
            result[f"{aig}|{name}|seed-{seed}"] = nmse(y, pred)
            rho, tau = rank_metrics(y, pred)
            result[f"{aig}|{name}|seed-{seed}|rho"] = rho
            result[f"{aig}|{name}|seed-{seed}|tau"] = tau
            result[f"{aig}|{name}|seed-{seed}|wtau"] = float(weightedtau(y, pred)[0])  # type: ignore[arg-type]

    else:
        result = {}
        rng = np.random.default_rng(seed)
        for name, pred, pred_func in [
            ("0-layer",       zero,     lambda yp: zero_layer_prediction(yp, counts)),
            ("size+depth",    size_depth, lambda yp: size_depth_prediction(yp, counts, depths)),
            ("x",             xfeat,    lambda yp: x_feature_prediction(x, yp, 1e-8)),
            ("wl-undirected", wl_undir, lambda yp, h=hashes_undir: wl_layer_prediction(h, yp)),
            ("wl-directed",   wl_dir,   lambda yp, h=hashes_dir:   wl_layer_prediction(h, yp)),
        ]:
            result[f"{aig}|{name}|seed-{seed}"] = nmse(y, pred)
            rho, tau = rank_metrics(y, pred)
            result[f"{aig}|{name}|seed-{seed}|rho"] = rho
            result[f"{aig}|{name}|seed-{seed}|tau"] = tau
            result[f"{aig}|{name}|seed-{seed}|wtau"] = float(weightedtau(y, pred)[0])  # type: ignore[arg-type]

    perm_rhos, perm_taus, perm_wtaus = [], [], []
    for _ in range(N_PERMS):
        y_perm = rng.permutation(y)
        r, t = rank_metrics(y, y_perm)
        perm_rhos.append(r)
        perm_taus.append(t)
        perm_wtaus.append(float(weightedtau(y, y_perm)[0]))  # type: ignore[arg-type]
    result[f"{aig}|{name}|seed-{seed}|perm-rho"] = np.array(perm_rhos)
    result[f"{aig}|{name}|seed-{seed}|perm-tau"] = np.array(perm_taus)
    result[f"{aig}|{name}|seed-{seed}|perm-wtau"] = np.array(perm_wtaus)
    print(f"  done: {aig} seed {seed}")
    return result


def main():
    available = []
    for aig in AIGS:
        try:
            load(aig, SEEDS[0])
            available.append(aig)
        except FileNotFoundError:
            print(f"[skip] {aig}: dataset missing")

    jobs = [(aig, seed) for aig in available for seed in SEEDS]
    results = Parallel(n_jobs=len(SEEDS), prefer="processes")(
        delayed(_process_seed)(aig, seed) for aig, seed in jobs
    )

    mse_results: dict = {}
    rank_results: dict = {}
    regret_results: dict = {}
    perm_rank_results: dict = {}
    for res in results:
        if res is None:
            continue
        for key, val in res.items():
            if "|perm-" in key:
                perm_rank_results[key] = val
            elif key.endswith("|rho") or key.endswith("|tau"):
                rank_results[key] = val
            elif key.endswith("|wtau") or "|regret-k" in key:
                regret_results[key] = val
            else:
                val=max(val, 1e-10)
                mse_results[key] = val

    np.savez("RES-DUMMY-PREDICTORS/predictor_mse_all_actions.npz", **mse_results)
    np.savez("RES-DUMMY-PREDICTORS/predictor_ranks_all_actions.npz", **rank_results)
    np.savez("RES-DUMMY-PREDICTORS/predictor_regret_all_actions.npz", **regret_results)
    np.savez("RES-DUMMY-PREDICTORS/predictor_perm_ranks_all_actions.npz", **perm_rank_results)


if __name__ == "__main__":
    main()
