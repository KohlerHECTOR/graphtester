import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap

# Predictor MSEs (normalized) produced by compare_rankings.py, if available.
try:
    _pred_mse = dict(np.load('res/predictor_mse.npz'))
except FileNotFoundError:
    _pred_mse = {}


def _draw_predictor_lines(ax, key_prefix, legend=False):
    """Draw horizontal lines for the size+depth and x-feature predictor MSEs."""
    zero_key = f"{key_prefix}|0-layer"
    if zero_key in _pred_mse:
        ax.axhline(
            float(np.mean(_pred_mse[zero_key])), color='black', linestyle='--', linewidth=2,
            label=r'$|\mathcal{V}_{\mathcal{G}}| \rightarrow \hat{V}^*(\mathcal{G})$' if legend else None,
        )
    sd_key = f"{key_prefix}|size+depth"
    if sd_key in _pred_mse:
        ax.axhline(
            float(np.mean(_pred_mse[sd_key])), color='green', linestyle=':', linewidth=2,
            label=r'$(|\mathcal{V}_{\mathcal{G}}|, d) \rightarrow \hat{V}^*(\mathcal{G})$' if legend else None,
        )
    key = f"{key_prefix}|x"
    if key in _pred_mse:
        ax.axhline(
            (float(np.mean(_pred_mse[key]))), color='magenta', linestyle='-.', linewidth=2,
            label=(r'$x_{\mathcal{G}} \rightarrow \hat{V}^*(\mathcal{G})$' if legend else None),
        )


AIGS = [
    "i10", "apex1", "dalu",
    "k2", "bc0", "mainpla",
]
plt.clf()
fig, axes = plt.subplots(2, 3, figsize=(20, 8))
for ax, aig in zip(axes.flat, AIGS):
    m = np.load(f'res/{aig}_curve.npy')
    ax.plot(m, label=r'$\mathcal{G} \rightarrow \hat{V}^*(\mathcal{G})$', linewidth=2.5)

    a = np.load(f'res/{aig}_perm_curves.npy')
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
    ax.set_xticks([0, 1, 2],['0', '1', '2'], fontsize=14)
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

    fig_bar, axes_bar = plt.subplots(2, 3, figsize=(22, 8), sharey=True)
    for ax, aig in zip(axes_bar.flat, AIGS):
        rhos = [float(_ranks.get(f"{aig}|{p}|rho", np.nan)) for p in PREDICTORS]
        taus = [float(_ranks.get(f"{aig}|{p}|tau", np.nan)) for p in PREDICTORS]
        ax.bar(x_pos - bar_width / 2, rhos, bar_width, label='Spearman ρ', color='steelblue')
        ax.bar(x_pos + bar_width / 2, taus, bar_width, label='Kendall τ', color='darkorange')
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