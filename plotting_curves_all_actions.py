import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import bootstrap

try:
    _pred_mse = dict(np.load('res/predictor_mse_all_actions.npz'))
except FileNotFoundError:
    _pred_mse = {}


def _collect_mse_vals(key_prefix, name):
    return np.array([
        _pred_mse[f"{key_prefix}|{name}|seed-{s}"]
        for s in range(SEEDS)
        if f"{key_prefix}|{name}|seed-{s}" in _pred_mse
    ])


def _draw_predictor_lines(ax, key_prefix, legend=False):
    """Draw horizontal lines for the size+depth and x-feature predictor MSEs."""
    for name, color, ls, lbl in [
        ("0-layer", 'black', '--', r'$|\mathcal{V}_{\mathcal{G}}| \rightarrow \hat{V}^*(\mathcal{G})$'),
        ("size+depth", 'green', ':', r'$(|\mathcal{V}_{\mathcal{G}}|, d) \rightarrow \hat{V}^*(\mathcal{G})$'),
        ("x", 'magenta', '-.', r'$x_{\mathcal{G}} \rightarrow \hat{V}^*(\mathcal{G})$'),
    ]:
        vals = _collect_mse_vals(key_prefix, name)
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        ax.axhline(mean_val, color=color, linestyle=ls, linewidth=2, label=lbl if legend else None)
        if len(vals) >= 2:
            res = bootstrap((vals,), np.mean, confidence_level=0.95, method='percentile')
            ax.axhspan(res.confidence_interval.low, res.confidence_interval.high, alpha=0.15, color=color)


AIGS = [
    "i10", "apex1", "dalu", "C1355", "C5315", "C7552",
    "k2", "bc0"]
# AIGS = [
#     "i10", "apex1", "dalu", "C6288", "C1355", "C5315", "C7552",
#     "k2", "bc0", "mainpla"
# ]
SEEDS = 10
plt.clf()
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, aig in zip(axes.flat, AIGS):
    curves = np.stack([
        np.load(f'dataset-{aig}-all-actions-True-mc-simu-10-seed-{s}-rs-False_curve_undirected.npy')
        for s in range(SEEDS)
    ], axis=0)
    m = np.mean(curves, axis=0)
    n_steps_c = curves.shape[1]
    ci_low_c = np.empty(n_steps_c)
    ci_high_c = np.empty(n_steps_c)
    for i in range(n_steps_c):
        res = bootstrap((curves[:, i],), np.mean, confidence_level=0.95, method='percentile')
        ci_low_c[i] = res.confidence_interval.low
        ci_high_c[i] = res.confidence_interval.high
    ax.plot(m, label=r'$\mathcal{G} \rightarrow \hat{V}^*(\mathcal{G})$', linewidth=2.5)
    ax.fill_between(np.arange(n_steps_c), ci_low_c, ci_high_c, alpha=0.2)

    curves = np.stack([
        np.load(f'dataset-{aig}-all-actions-True-mc-simu-10-seed-{s}-rs-False_curve_directed.npy')
        for s in range(SEEDS)
    ], axis=0)
    m = np.mean(curves, axis=0)
    n_steps_c = curves.shape[1]
    ci_low_c = np.empty(n_steps_c)
    ci_high_c = np.empty(n_steps_c)
    for i in range(n_steps_c):
        res = bootstrap((curves[:, i],), np.mean, confidence_level=0.95, method='percentile')
        ci_low_c[i] = res.confidence_interval.low
        ci_high_c[i] = res.confidence_interval.high
    ax.plot(m, label=r'$\mathcal{G}^{\rightarrow} \rightarrow \hat{V}^*(\mathcal{G})$', linewidth=2.5, linestyle='-', color='blue')
    ax.fill_between(np.arange(n_steps_c), ci_low_c, ci_high_c, alpha=0.2, color='blue')

    # Combine all seeds and permutations: shape (10 * n_perms, n_steps)
    a = np.concatenate([
        np.load(f'dataset-{aig}-all-actions-True-mc-simu-10-seed-{s}-rs-False_perm_curves_undirected.npy')
        for s in range(SEEDS)
    ], axis=0)
    m = np.mean(a, axis=0)

    n_steps = a.shape[1]
    ci_low = np.empty(n_steps)
    ci_high = np.empty(n_steps)
    for i in range(n_steps):
        res = bootstrap((a[:, i],), np.mean, confidence_level=0.95, method='percentile')
        ci_low[i] = res.confidence_interval.low
        ci_high[i] = res.confidence_interval.high

    ax.plot(m, color='red', label=r'$\mathcal{G} \rightarrow \mathrm{RNDPermut}(\hat{V}^*(\mathcal{G}))$', linewidth=2.5)
    ax.fill_between(np.arange(n_steps), ci_low, ci_high, alpha=0.2, color='red')

    a = np.concatenate([
        np.load(f'dataset-{aig}-all-actions-True-mc-simu-10-seed-{s}-rs-False_perm_curves_directed.npy')
        for s in range(SEEDS)
    ], axis=0)
    m = np.mean(a, axis=0)

    n_steps = a.shape[1]
    ci_low = np.empty(n_steps)
    ci_high = np.empty(n_steps)
    for i in range(n_steps):
        res = bootstrap((a[:, i],), np.mean, confidence_level=0.95, method='percentile')
        ci_low[i] = res.confidence_interval.low
        ci_high[i] = res.confidence_interval.high

    ax.plot(m, label=r'$\mathcal{G}^{\rightarrow} \rightarrow \mathrm{RNDPermut}(\hat{V}^*(\mathcal{G}))$', linewidth=2.5, color='tab:red')
    ax.fill_between(np.arange(n_steps), ci_low, ci_high, alpha=0.2, color='tab:red')
    _draw_predictor_lines(ax, aig, legend=False)
    ax.set_yscale('log')
    ax.set_xticks([0, 1, 2, 3, 4],['1', '2', '3', '4', '5'], fontsize=14)
    ax.set_xlabel('MPNN layers', fontsize=15)
    ax.set_title(aig, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

# Collect legend handles from the first subplot (all line types are drawn there)
_draw_predictor_lines(axes[0, 0], AIGS[0], legend=True)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, fontsize=18,
    loc='center', bbox_to_anchor=(0.3, -0.05), ncol=5,
)

for ax in axes[:, 0]:
    ax.set_ylabel('Theoretical MSE lower bound', fontsize=15)

fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig('per_aig_all_actions.pdf', bbox_inches='tight', dpi=300)
plt.clf()

# --- Bar plot of ranking metrics per AIG ---
try:
    _ranks = dict(np.load('res/predictor_ranks_all_actions.npz'))
except FileNotFoundError:
    _ranks = {}

try:
    _regret = dict(np.load('res/predictor_regret_all_actions.npz'))
except FileNotFoundError:
    _regret = {}

try:
    _perm_ranks = dict(np.load('res/predictor_perm_ranks_all_actions.npz', allow_pickle=True))
except FileNotFoundError:
    _perm_ranks = {}

if _ranks:
    PREDICTORS = ['0-layer', 'size+depth', 'x', 'wl-undirected', 'wl-directed']
    PRED_LABELS = [
        r'$|\mathcal{V}_\mathcal{G}| \rightarrow \hat{V}^*$',
        r'$(|\mathcal{V}|, d)_\mathcal{G} \rightarrow \hat{V}^*$',
        r'$x_\mathcal{G} \rightarrow \hat{V}^*$',
        r'$\mathcal{G} \rightarrow \hat{V}^*(\mathcal{G})$',
        r'$\mathcal{G}^{\rightarrow} \rightarrow \hat{V}^*(\mathcal{G})$',
    ]
    x_pos = np.arange(len(PREDICTORS))
    bar_width = 0.25

    fig_bar, axes_bar = plt.subplots(2, 5, figsize=(22, 8), sharey=True)
    for ax, aig in zip(axes_bar.flat, AIGS):
        rho_means, rho_errs, tau_means, tau_errs, wtau_means, wtau_errs = [], [], [], [], [], []
        for p in PREDICTORS:
            rho_vals = np.array([float(_ranks[f"{aig}|{p}|seed-{s}|rho"]) for s in range(SEEDS) if f"{aig}|{p}|seed-{s}|rho" in _ranks])
            tau_vals = np.array([float(_ranks[f"{aig}|{p}|seed-{s}|tau"]) for s in range(SEEDS) if f"{aig}|{p}|seed-{s}|tau" in _ranks])
            wtau_vals = np.array([float(_regret[f"{aig}|{p}|seed-{s}|wtau"]) for s in range(SEEDS) if f"{aig}|{p}|seed-{s}|wtau" in _regret])
            rho_means.append(float(np.nanmean(rho_vals)) if len(rho_vals) else np.nan)
            tau_means.append(float(np.nanmean(tau_vals)) if len(tau_vals) else np.nan)
            wtau_means.append(float(np.nanmean(wtau_vals)) if len(wtau_vals) else np.nan)
            if len(rho_vals) >= 2:
                res = bootstrap((rho_vals,), np.mean, confidence_level=0.95, method='percentile')
                rho_errs.append([rho_means[-1] - res.confidence_interval.low, res.confidence_interval.high - rho_means[-1]])
            else:
                rho_errs.append([0, 0])
            if len(tau_vals) >= 2:
                res = bootstrap((tau_vals,), np.mean, confidence_level=0.95, method='percentile')
                tau_errs.append([tau_means[-1] - res.confidence_interval.low, res.confidence_interval.high - tau_means[-1]])
            else:
                tau_errs.append([0, 0])
            if len(wtau_vals) >= 2:
                res = bootstrap((wtau_vals,), np.mean, confidence_level=0.95, method='percentile')
                wtau_errs.append([wtau_means[-1] - res.confidence_interval.low, res.confidence_interval.high - wtau_means[-1]])
            else:
                wtau_errs.append([0, 0])
        ax.bar(x_pos - bar_width, rho_means, bar_width, yerr=np.array(rho_errs).T, color='steelblue', capsize=3)
        ax.bar(x_pos, tau_means, bar_width, yerr=np.array(tau_errs).T, color='darkorange', capsize=3)
        ax.bar(x_pos + bar_width, wtau_means, bar_width, yerr=np.array(wtau_errs).T, color='mediumpurple', capsize=3)

        # Three horizontal null lines: one per metric, pooled over all predictors and seeds
        for metric, color, ls in [
            ('perm-rho',  'steelblue',   '--'),
            ('perm-tau',  'darkorange',  '-.'),
            ('perm-wtau', 'mediumpurple', ':'),
        ]:
            parts = [
                _perm_ranks[f"{aig}|{p}|seed-{s}|{metric}"]
                for p in PREDICTORS for s in range(SEEDS)
                if f"{aig}|{p}|seed-{s}|{metric}" in _perm_ranks
            ]
            if not parts:
                continue
            null_vals = np.concatenate(parts)
            null_mean = float(np.mean(null_vals))
            ax.axhline(null_mean, color=color, linestyle=ls, linewidth=1.5, zorder=3)
            if len(null_vals) >= 2:
                res = bootstrap((null_vals,), np.mean, confidence_level=0.95, method='percentile')
                ax.axhspan(res.confidence_interval.low, res.confidence_interval.high,
                           color=color, alpha=0.12, zorder=2)

        ax.set_ylim(-0.01, 1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(PRED_LABELS, fontsize=11, rotation=20, ha='right')
        ax.set_title(aig, fontsize=14)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    for ax in axes_bar[:, 0]:
        ax.set_ylabel('Rank correlation', fontsize=13)

    legend_handles = [
        Patch(color='steelblue',    label='Spearman \u03c1'),
        Patch(color='darkorange',   label='Kendall \u03c4'),
        Patch(color='mediumpurple', label='Weighted \u03c4'),
        Line2D([0], [0], color='steelblue',    linestyle='--', linewidth=1.5, label='Perm null \u03c1'),
        Line2D([0], [0], color='darkorange',   linestyle='-.', linewidth=1.5, label='Perm null \u03c4'),
        Line2D([0], [0], color='mediumpurple', linestyle=':',  linewidth=1.5, label='Perm null w\u03c4'),
    ]
    fig_bar.legend(handles=legend_handles, fontsize=13,
                   loc='center', bbox_to_anchor=(0.5, -0.04), ncol=6)
    fig_bar.tight_layout(rect=(0, 0.08, 1, 1))
    fig_bar.savefig('ranking_bars_all_actions.pdf', bbox_inches='tight', dpi=300)
    plt.clf()