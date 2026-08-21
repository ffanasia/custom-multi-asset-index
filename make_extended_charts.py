import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTES = {
    "light": dict(INK="#1b2320", INK_SOFT="#4a544e", ACCENT="#a35f1b",
                  SIGNAL="#3f6e52", GRID="#cad0c4", AXIS="#a9b1a1"),
    "dark":  dict(INK="#e9ede6", INK_SOFT="#a9b1a4", ACCENT="#dd9a4d",
                  SIGNAL="#6fa383", GRID="#2b322d", AXIS="#3c453d"),
}

with open("extended_results.json") as f:
    R = json.load(f)

OUT = "/home/claude/extended_analysis/"

for theme, pal in PALETTES.items():
    INK, INK_SOFT, ACCENT, SIGNAL, GRID, AXIS = (
        pal["INK"], pal["INK_SOFT"], pal["ACCENT"], pal["SIGNAL"], pal["GRID"], pal["AXIS"])

    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Georgia", "serif"],
        "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.linewidth": 0.8, "font.size": 12,
    })

    def style_ax(ax):
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(AXIS)
        ax.tick_params(labelsize=10.5, length=3)
        ax.grid(True, axis="y", alpha=0.9)
        ax.grid(False, axis="x")

    # ---------------------------------------------------------------
    # Fig 5 -- ablation ladder
    # ---------------------------------------------------------------
    ab = pd.DataFrame(R["ablation"])
    labels = ["Naive\nmomentum\n(gross)", "+ inverse-vol\nsizing\n(gross)",
              "+ defensive\nfallback\n(gross)", "+ transaction\ncosts\n(net)"]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), dpi=180)
    x = np.arange(len(ab))

    ax = axes[0]
    ax.bar(x, ab["sharpe"], color=[INK_SOFT, INK_SOFT, SIGNAL, INK], width=0.58)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.2)
    ax.set_ylabel("Sharpe ratio")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlim(-0.65, len(ab) - 0.35)
    style_ax(ax)
    for i, v in enumerate(ab["sharpe"]):
        ax.text(i, v + (0.015 if v >= 0 else -0.03), f"{v:.2f}", ha="center", fontsize=9.5, color=INK)

    ax = axes[1]
    ax.bar(x, ab["max_drawdown"] * 100, color=[INK_SOFT, INK_SOFT, SIGNAL, INK], width=0.58)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.2)
    ax.set_ylabel("Max drawdown (%)")
    ax.set_xlim(-0.65, len(ab) - 0.35)
    style_ax(ax)
    for i, v in enumerate(ab["max_drawdown"]):
        ax.text(i, v * 100 - 1.5, f"{v*100:.1f}", ha="center", fontsize=9.5, va="top", color=INK)

    fig.suptitle("What each design choice actually buys you", fontsize=13, y=1.03, color=INK)
    fig.text(0.5, -0.03, "first three bars are gross of cost -- only the last nets out transaction costs",
              ha="center", fontsize=8.5, color=INK_SOFT, style="italic")
    fig.tight_layout()
    fig.savefig(OUT + f"fig5_ablation_{theme}.png", dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 6 -- cost sensitivity
    # ---------------------------------------------------------------
    cs = pd.DataFrame(R["cost_sensitivity"])
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=180)
    ax2 = ax.twinx()
    ax.plot(cs["cost_bps"], cs["sharpe"], color=INK, linewidth=2.2, marker="o", label="Sharpe ratio")
    ax2.plot(cs["cost_bps"], cs["max_drawdown"] * 100, color=ACCENT, linewidth=2.0, marker="o",
             linestyle=(0, (4, 2)), label="Max drawdown")
    ax.set_xlabel("Assumed cost (bps per rebalance)")
    ax.set_ylabel("Sharpe ratio")
    ax2.set_ylabel("Max drawdown (%)")
    ax.axvline(5, color=AXIS, linewidth=0.8, linestyle=(0, (1, 1.4)))
    ax.text(5.3, cs["sharpe"].max(), "assumption\nused (5bps)", fontsize=8.5, color=INK_SOFT, va="top")
    for s in ["top"]:
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS); ax.spines["bottom"].set_color(AXIS)
    ax2.spines["right"].set_color(ACCENT); ax2.tick_params(axis="y", colors=ACCENT)
    ax2.yaxis.label.set_color(ACCENT)
    ax.grid(True, axis="y", alpha=0.9); ax.grid(False, axis="x")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=10, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT + f"fig6_cost_sensitivity_{theme}.png", dpi=180, transparent=True)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 7 -- regime breakdown
    # ---------------------------------------------------------------
    rg = pd.DataFrame(R["regimes"])
    fig, ax = plt.subplots(figsize=(10.2, 4.6), dpi=180)
    x = np.arange(len(rg))
    w = 0.36
    b1 = ax.bar(x - w/2, rg["strategy_return"] * 100, width=w, color=INK, label="Strategy (net)")
    b2 = ax.bar(x + w/2, rg["spy_return"] * 100, width=w, color=ACCENT, label="SPY")
    ax.axhline(0, color=AXIS, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace(" ", "\n", 1) for r in rg["regime"]], fontsize=8.8)
    ax.set_ylabel("Period return (%)")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INK)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + (1.5 if h >= 0 else -3),
                     f"{h:.0f}", ha="center", fontsize=8, va="bottom" if h >= 0 else "top", color=INK)
    fig.tight_layout()
    fig.savefig(OUT + f"fig7_regimes_{theme}.png", dpi=180, transparent=True)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 8 -- walk-forward (with SE error bars + SPY OOS reference)
    # ---------------------------------------------------------------
    wf = pd.DataFrame(R["walk_forward"]["grid"]).sort_values("window_days")
    wf["fit_se"] = R["walk_forward"]["fit_se"]
    wf["full_se"] = R["walk_forward"]["full_se"]
    spy_test_sharpe = R["walk_forward"]["spy_test_sharpe"]

    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    ax.errorbar(wf["window_days"], wf["fit_sharpe"], yerr=wf["fit_se"], color=INK_SOFT, linewidth=1.8,
                marker="o", linestyle=(0, (3, 2)), capsize=3, elinewidth=0.9,
                label="Fit period Sharpe ± SE (2016–2019, blind)")
    ax.errorbar(wf["window_days"], wf["full_sharpe"], yerr=wf["full_se"], color=INK, linewidth=2.2,
                marker="o", capsize=3, elinewidth=0.9, label="Full period Sharpe ± SE (2016–2024)")
    ax.axhline(spy_test_sharpe, color=ACCENT, linewidth=1.4, linestyle=(0, (4, 2)))
    fit_w = R["walk_forward"]["fit_period_winner"]
    ax.axvline(fit_w, color=SIGNAL, linewidth=1.2, linestyle=(0, (1, 1.4)))
    ax.set_xlabel("Momentum ranking window (trading days)")
    ax.set_ylabel("Sharpe ratio")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="lower center", labelcolor=INK)
    ax.text(fit_w + 4, wf["full_sharpe"].max() + wf["full_se"].max() + 0.05,
            f"chosen blind on\nfit period ({fit_w}d)", fontsize=8.5, color=SIGNAL, va="bottom")
    ax.text(wf["window_days"].min(), spy_test_sharpe + 0.02,
            f"SPY, same OOS window ({spy_test_sharpe:.2f})", fontsize=8.2, color=ACCENT, va="bottom")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.20)
    fig.text(0.5, -0.02,
             "error bars are ±1 analytical SE on the Sharpe estimate -- fit-period bars overlap heavily, "
             "so no single window is statistically distinguishable from another",
             ha="center", fontsize=8.3, color=INK_SOFT, style="italic")
    fig.tight_layout()
    fig.savefig(OUT + f"fig8_walkforward_{theme}.png", dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 9 -- bootstrap significance test (Sharpe diff, strategy vs SPY)
    # ---------------------------------------------------------------
    sig = R["significance"]
    ci_lo, ci_hi = sig["sharpe_diff_ci_95"]
    center = sig["diff_mean"]
    counts = np.array(sig["diff_hist_counts"])
    edges = np.array(sig["diff_hist_edges"])
    widths = np.diff(edges)
    centers_bins = edges[:-1] + widths / 2

    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=180)
    ax.bar(centers_bins, counts, width=widths, color=INK_SOFT, alpha=0.85, linewidth=0)
    ax.axvline(0, color=AXIS, linewidth=1.2)
    ax.axvline(ci_lo, color=ACCENT, linewidth=1.4, linestyle=(0, (4, 2)))
    ax.axvline(ci_hi, color=ACCENT, linewidth=1.4, linestyle=(0, (4, 2)))
    ax.axvline(center, color=INK, linewidth=1.8)
    ax.set_xlabel("Sharpe(strategy) − Sharpe(SPY), block-bootstrap resample")
    ax.set_ylabel("Resamples")
    style_ax(ax)
    ax.text(ci_hi + 0.02, ax.get_ylim()[1] * 0.9, "95% CI", fontsize=9, color=ACCENT)
    ax.text(0.02, ax.get_ylim()[1] * 0.78, "zero", fontsize=9, color=INK_SOFT)
    fig.suptitle("The 0.65 vs 0.67 Sharpe gap is not statistically distinguishable from noise",
                 fontsize=12.5, y=1.02, color=INK)
    fig.text(0.5, -0.03,
             f"n={sig['n_bootstrap']:,} paired block-bootstrap resamples (block={sig['block_size']}d) -- "
             f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}] includes zero; strategy has the higher Sharpe in only "
             f"{sig['pct_resamples_strategy_higher_sharpe']*100:.0f}% of resamples",
             ha="center", fontsize=8.3, color=INK_SOFT, style="italic")
    fig.tight_layout()
    fig.savefig(OUT + f"fig9_significance_{theme}.png", dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 10 -- deflated Sharpe ratio: DSR vs. assumed number of implicit trials
    # ---------------------------------------------------------------
    ds = R["deflated_sharpe"]
    sweep = pd.DataFrame(ds["sweep"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=180)

    ax = axes[0]
    ax.plot(sweep["N"], sweep["sr0_annualized"], color=ACCENT, linewidth=2.0, marker="o", markersize=4)
    ax.axhline(ds["sr_hat_annualized"], color=INK, linewidth=1.6, linestyle=(0, (4, 2)))
    ax.text(sweep["N"].iloc[3], ds["sr_hat_annualized"] - 0.045, f"observed Sharpe ({ds['sr_hat_annualized']:.2f})",
            fontsize=8.8, color=INK, va="top")
    ax.set_xscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.06)
    ax.set_xlabel("Assumed number of implicit trials (N)")
    ax.set_ylabel("Sharpe ratio (annualized)")
    ax.set_title("“Luck bar”: best-of-N Sharpe expected by chance", fontsize=10.8, color=INK)
    style_ax(ax)

    ax = axes[1]
    ax.plot(sweep["N"], sweep["dsr"] * 100, color=SIGNAL, linewidth=2.0, marker="o", markersize=4)
    ax.axhline(50, color=AXIS, linewidth=0.8, linestyle=(0, (1, 1.4)))
    ax.axvline(ds["n_actual_trials"], color=INK_SOFT, linewidth=1.1, linestyle=(0, (1, 1.4)))
    ax.text(ds["n_actual_trials"] * 1.15, 62,
            f"N={ds['n_actual_trials']}\n(actually tested)", fontsize=8.2, color=INK_SOFT, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Assumed number of implicit trials (N)")
    ax.set_ylabel("Deflated Sharpe ratio (%)")
    ax.set_title("P(Sharpe reflects skill, not selection luck)", fontsize=10.8, color=INK)
    style_ax(ax)

    fig.suptitle("Correcting the backtest's own Sharpe ratio for how many things were tried",
                 fontsize=12.8, y=1.04, color=INK)
    fig.text(0.5, -0.03,
              f"skew={ds['skew']:.2f}, excess kurtosis={ds['kurtosis']-3:.2f} of daily returns folded into the correction "
              "(Bailey & López de Prado, 2014) -- DSR stays above 92% even assuming 100 implicit trials",
              ha="center", fontsize=8.3, color=INK_SOFT, style="italic")
    fig.tight_layout()
    fig.savefig(OUT + f"fig10_deflated_sharpe_{theme}.png", dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)

print("done")
