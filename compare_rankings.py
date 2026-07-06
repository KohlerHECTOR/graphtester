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
from scipy.stats import kendalltau, spearmanr

from abc_py import AbcInterface

WL_ITERS = 5  # number of message-passing layers for the WL-based predictor
SEEDS = list(range(10))


AIGS = [
    "i10", "apex1", "dalu", "C6288", "C1355", "C5315", "C7552",
    "k2", "bc0", "mainpla"
]


def load(aig: str, seed: int):
    # TODO: path should be argument

    """Load a dataset for a given seed. Returns (x, y_min) or raises FileNotFoundError."""
    p =f"dataset-{aig}-all-actions-False-mc-simu-100-seed-{seed}-rs-False/res.npz"
    data = np.load(p)
    x_all = data["x"]
    y_all = data["y"]
    return x_all, np.amin(y_all, axis=1)


def aig_path(aig: str, seed: int, i: int) -> str:
    # TODO: path should be argument
    return f"dataset-{aig}-all-actions-False-mc-simu-100-seed-{seed}-rs-False/{i}.aig"


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
    snapped = np.round(x / tol)
    _, inverse = np.unique(snapped, axis=0, return_inverse=True)
    for group in np.unique(inverse):
        mask = inverse == group
        pred[mask] = y[mask].mean()
    return pred


def load_wl_hashes(aig: str, seed: int, directed: bool, k: int) -> list[str]:
    """Load precomputed 1-WL graph hashes at iteration ``k`` from the pickle written
    by ``aig_wl_analysis.py`` (``{dataset}_hashes_{undirected|directed}.pkl``).

    The pickle holds ``{iteration: [hash_for_graph_0, ...]}`` in graph index order.
    """
    suffix = "directed" if directed else "undirected"
    prefix = f"dataset-{aig}-all-actions-False-mc-simu-100-seed-{seed}-rs-False"
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


def gap_weighted_discordance(y: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of the total V* gap mass that ``pred`` misorders.

    A value-weighted Kendall tau: for every pair of states, if ``pred`` ranks them
    in the opposite order to the true optimal value ``y`` (lower = better), the pair
    contributes its true value gap ``|y_i - y_j|`` to the numerator. Normalized by the
    total gap mass over all pairs, giving a number in [0, 1]. Ties in ``pred`` count as
    half a discordance. This directly captures "how much V* is lost to ranking errors".
    """
    n = len(y)
    num = 0.0
    den = 0.0
    for i in range(n):
        dy = y[i] - y[i + 1 :]
        dp = pred[i] - pred[i + 1 :]
        gap = np.abs(dy)
        den += gap.sum()
        sign = dy * dp
        num += gap[sign < 0].sum() + 0.5 * gap[sign == 0].sum()
    return float(num / den) if den > 0 else 0.0


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
    wl_undir = wl_layer_prediction(load_wl_hashes(aig, seed, False, WL_ITERS), y)
    wl_dir = wl_layer_prediction(load_wl_hashes(aig, seed, True, WL_ITERS), y)

    result = {}
    rng = np.random.default_rng(seed)
    for name, pred in [
        ("0-layer", zero),
        ("size+depth", size_depth),
        ("x", xfeat),
        ("wl-undirected", wl_undir),
        ("wl-directed", wl_dir),
    ]:
        result[f"{aig}|{name}|seed-{seed}"] = nmse(y, pred)
        rho, tau = rank_metrics(y, pred)
        result[f"{aig}|{name}|seed-{seed}|rho"] = rho
        result[f"{aig}|{name}|seed-{seed}|tau"] = tau
        result[f"{aig}|{name}|seed-{seed}|gwd"] = gap_weighted_discordance(y, pred)
        print(result[f"{aig}|{name}|seed-{seed}|gwd"])
        # for k in REGRET_KS:
        #     result[f"{aig}|{name}|seed-{seed}|regret-k{k}"] = expected_regret_at_k(
        #         y, pred, k, REGRET_DRAWS, rng
        #     )
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
    for res in results:
        if res is None:
            continue
        for key, val in res.items():
            if key.endswith("|rho") or key.endswith("|tau"):
                rank_results[key] = val
            elif key.endswith("|gwd") or "|regret-k" in key:
                regret_results[key] = val
            else:
                mse_results[key] = val

    # Print summary grouped by aig/seed
    for aig in available:
        print(f"\n{aig}")
        for seed in SEEDS:
            print(f"  seed {seed}")
            for name in ("0-layer", "size+depth", "x", "wl-undirected", "wl-directed"):
                mse_val = mse_results.get(f"{aig}|{name}|seed-{seed}", float('nan'))
                rho = rank_results.get(f"{aig}|{name}|seed-{seed}|rho", float('nan'))
                tau = rank_results.get(f"{aig}|{name}|seed-{seed}|tau", float('nan'))
                gwd = regret_results.get(f"{aig}|{name}|seed-{seed}|gwd", float('nan'))
                
                print(f"    {name:<10} {rho:>10.4f} {tau:>10.4f} {mse_val:>10.4g} {gwd:>9.4g}")

    np.savez("res/predictor_mse.npz", **mse_results)
    np.savez("res/predictor_ranks.npz", **rank_results)
    np.savez("res/predictor_regret.npz", **regret_results)
    print("\nSaved predictor MSEs to res/predictor_mse.npz")
    print("Saved predictor rank metrics to res/predictor_ranks.npz")
    print("Saved predictor regret metrics to res/predictor_regret.npz")


if __name__ == "__main__":
    main()
