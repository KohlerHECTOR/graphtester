import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap

try:
    _pred_mse = dict(np.load('res/predictor_mse.npz'))
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
    "k2", "bc0"
]
# AIGS = [
#     "i10", "apex1", "dalu", "C6288", "C1355", "C5315", "C7552",
#     "k2", "bc0", "mainpla"
# ]
SEEDS = 10
plt.clf()
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, aig in zip(axes.flat, AIGS):
    curves = np.stack([
        np.load(f'dataset-{aig}-all-actions-False-mc-simu-100-seed-{s}-rs-False_curve.npy')
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

    # Combine all seeds and permutations: shape (10 * n_perms, n_steps)
    a = np.concatenate([
        np.load(f'dataset-{aig}-all-actions-False-mc-simu-100-seed-{s}-rs-False_perm_curves.npy')
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
    _draw_predictor_lines(ax, aig, legend=False)
    ax.set_yscale('log')
    ax.set_xticks([0, 1, 2, 3, 4, 5],['0', '1', '2', '3', '4', '5'], fontsize=14)
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
fig.savefig('per_aig.pdf', bbox_inches='tight')
plt.clf()

# --- Bar plot of ranking metrics per AIG ---
try:
    _ranks = dict(np.load('res/predictor_ranks.npz'))
except FileNotFoundError:
    _ranks = {}

if _ranks:
    PREDICTORS = ['0-layer', 'size+depth', 'x', '1-layer']
    PRED_LABELS = [
        r'$|\mathcal{V}_\mathcal{G}| \rightarrow \hat{V}^*$',
        r'$(|\mathcal{V}|, d)_\mathcal{G} \rightarrow \hat{V}^*$',
        r'$x_\mathcal{G} \rightarrow \hat{V}^*$',
        r'$\mathcal{G} \rightarrow \hat{V}^*(\mathcal{G})$',
    ]
    x_pos = np.arange(len(PREDICTORS))
    bar_width = 0.35

    fig_bar, axes_bar = plt.subplots(2, 5, figsize=(22, 8), sharey=True)
    for ax, aig in zip(axes_bar.flat, AIGS):
        rho_means, rho_errs, tau_means, tau_errs = [], [], [], []
        for p in PREDICTORS:
            rho_vals = np.array([float(_ranks[f"{aig}|{p}|seed-{s}|rho"]) for s in range(SEEDS) if f"{aig}|{p}|seed-{s}|rho" in _ranks])
            tau_vals = np.array([float(_ranks[f"{aig}|{p}|seed-{s}|tau"]) for s in range(SEEDS) if f"{aig}|{p}|seed-{s}|tau" in _ranks])
            rho_means.append(float(np.nanmean(rho_vals)) if len(rho_vals) else np.nan)
            tau_means.append(float(np.nanmean(tau_vals)) if len(tau_vals) else np.nan)
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
        ax.bar(x_pos - bar_width / 2, rho_means, bar_width, yerr=np.array(rho_errs).T, label='Spearman \u03c1', color='steelblue', capsize=3)
        ax.bar(x_pos + bar_width / 2, tau_means, bar_width, yerr=np.array(tau_errs).T, label='Kendall \u03c4', color='darkorange', capsize=3)
        ax.set_ylim(0.8, 1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(PRED_LABELS, fontsize=11, rotation=20, ha='right')
        ax.set_title(aig, fontsize=14)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    for ax in axes_bar[:, 0]:
        ax.set_ylabel('Rank correlation', fontsize=13)

    handles_bar, labels_bar = axes_bar[0, 0].get_legend_handles_labels()
    fig_bar.legend(handles_bar, labels_bar, fontsize=14,
                   loc='center', bbox_to_anchor=(0.5, -0.04), ncol=2)
    fig_bar.tight_layout(rect=(0, 0.08, 1, 1))
    fig_bar.savefig('ranking_bars.pdf', bbox_inches='tight')
    plt.clf()