"""Out-of-fold evaluation: the report's 2x2 table, the error decomposition and
the appendix figures.

Every number here comes from out-of-fold predictions over all 25,000 training
rows, each produced by a model that never saw the row, with early stopping
nested inside the fold. Nothing touches X_test.csv.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_assignment.data import load_raw
from ml_assignment.losses import brier_score, economic_loss
from ml_assignment.tuning import oof_predictions

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import HYPERPARAMS, PREPROCESS, SPLIT_SEED

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FIGURES = ARTIFACTS / "figures"
N_SPLITS = 5

# Reference palette, slots 1 and 2 (light mode). Identity is carried by hue AND
# linestyle/marker so the figures survive greyscale printing.
C_BRIER, C_ECON = "#2a78d6", "#eb6834"
SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": INK_MUTED,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def style(ax):
    ax.grid(True, alpha=0.7, linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIGURES / (name + '.pdf')}")


def calibration(y, p, n_bins=15):
    """Equal-count bins: mean predicted probability vs observed default rate."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 50:
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return np.array(rows)


if __name__ == "__main__":
    X_trn, y_trn, _ = load_raw()
    y = y_trn.to_numpy().astype(float)

    config = {"hidden": HYPERPARAMS["hidden"], "lr": HYPERPARAMS["lr"],
              "l2": HYPERPARAMS["l2"], "one_hot": PREPROCESS["one_hot"]}
    print(f"config: {config}\ncomputing out-of-fold predictions "
          f"({N_SPLITS} folds x 2 objectives)\n")

    p, epochs = {}, {}
    for name, w in (("brier", 1.0), ("econ", 3.0)):
        p[name], info = oof_predictions(X_trn, y, config, w_default=w,
                                        n_splits=N_SPLITS, split_seed=SPLIT_SEED)
        epochs[name] = [d["best_epoch"] for d in info]
        print(f"  {name}: best epochs per fold {epochs[name]}, "
              f"median {int(np.median(epochs[name]))}")

    np.savez(ARTIFACTS / "oof_predictions.npz", brier=p["brier"], econ=p["econ"], y=y)

    # ---- the 2x2 table -------------------------------------------------
    q = y.mean()
    c_star = 3 * q / (2 * q + 1)
    table = pd.DataFrame(
        [[brier_score(y, p[n]), economic_loss(y, p[n])] for n in ("brier", "econ")]
        + [[q * (1 - q), q * 3 * (1 - c_star) ** 2 + (1 - q) * c_star ** 2]],
        index=["Brier-trained", "Economic-trained", "Best constant"],
        columns=["Brier score", "Economic loss"],
    )
    print("\n=== out-of-sample performance (out-of-fold, n=%d) ===" % len(y))
    print(table.round(5).to_string())
    table.round(6).to_csv(ARTIFACTS / "table_2x2.csv")

    # ---- error balance across defaults and non-defaults -----------------
    d, nd = y == 1, y == 0
    bal = pd.DataFrame(
        [[((y[d] - p[n][d]) ** 2).mean(), ((y[nd] - p[n][nd]) ** 2).mean(),
          p[n][d].mean(), p[n][nd].mean(), p[n].mean(),
          3 * ((y[d] - p[n][d]) ** 2).sum() /
          (3 * ((y[d] - p[n][d]) ** 2).sum() + ((y[nd] - p[n][nd]) ** 2).sum())]
         for n in ("brier", "econ")],
        index=["Brier-trained", "Economic-trained"],
        columns=["MSE on defaults", "MSE on non-defaults", "mean p | default",
                 "mean p | non-default", "mean p", "share of econ loss from defaults"],
    )
    print(f"\n=== error balance (default rate {q:.4f}, "
          f"n_default={int(d.sum())}, n_non={int(nd.sum())}) ===")
    print(bal.round(4).to_string())
    bal.round(6).to_csv(ARTIFACTS / "table_error_balance.csv")

    # ---- theoretical check: p* = 3q/(2q+1) ------------------------------
    implied = 3 * p["brier"] / (2 * p["brier"] + 1)
    print(f"\ncorr(econ, 3q/(2q+1) of Brier) = {np.corrcoef(p['econ'], implied)[0,1]:.4f} | "
          f"mean abs deviation = {np.abs(p['econ'] - implied).mean():.4f}")

    # ---- figures ---------------------------------------------------------
    print("\nfigures:")
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    bins = np.linspace(0, 1, 61)
    ax.hist(p["brier"], bins=bins, histtype="step", linewidth=2, color=C_BRIER,
            label="Brier-trained")
    ax.hist(p["econ"], bins=bins, histtype="step", linewidth=2, color=C_ECON,
            linestyle="--", label="Economic-trained")
    top = ax.get_ylim()[1]
    for xv, lab, frac in ((q, f"base rate {q:.3f}", 0.97),
                          (c_star, f"3q/(2q+1) = {c_star:.3f}", 0.87)):
        ax.axvline(xv, color=INK_MUTED, linewidth=1, linestyle=":")
        ax.text(xv, top * frac, f" {lab}", color=INK_MUTED,
                fontsize=7.5, va="top", ha="left")
    ax.set_xlabel("predicted default probability")
    ax.set_ylabel("observations")
    ax.set_title("Predicted probability distributions (out-of-fold)")
    ax.legend(frameon=False)
    save(style(ax).figure, "fig1_distributions")

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=":",
            label="perfect calibration")
    for name, col, ls, mk in (("brier", C_BRIER, "-", "o"), ("econ", C_ECON, "--", "s")):
        c = calibration(y, p[name])
        ax.plot(c[:, 0], c[:, 1], color=col, linewidth=2, linestyle=ls, marker=mk,
                markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=f"{'Brier' if name == 'brier' else 'Economic'}-trained")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed default rate")
    ax.set_title("Calibration (15 equal-count bins)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(style(ax).figure, "fig2_calibration")

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=":")
    ax.scatter(implied, p["econ"], s=3, alpha=0.06, color=C_ECON, edgecolors="none")
    ax.set_xlabel(r"$3\hat p_{\mathrm{Brier}}\,/\,(2\hat p_{\mathrm{Brier}}+1)$")
    ax.set_ylabel(r"$\hat p_{\mathrm{Economic}}$")
    ax.set_title("Economic predictions vs the theoretical map")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(style(ax).figure, "fig3_theory_check")

    print(f"\nmedian best epoch for step 7 refit: "
          f"brier {int(np.median(epochs['brier']))}, econ {int(np.median(epochs['econ']))}")
