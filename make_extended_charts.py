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

# these charts are displayed at ~380-680px wide in the actual page layout, far
# smaller than a typical matplotlib default render -- so type sizes here are set
# for legibility at THAT display size, not for a full-bleed standalone image.
# explanatory prose lives in the HTML figcaption (real, crisp, reflowing text)
# rather than baked into the raster image as small annotations -- annotations
# inside the PNG are limited to short axis/legend/value labels that stay legible
# when the image is scaled down.
TICK_FS   = 20
LABEL_FS  = 22
LEGEND_FS = 18
VALUE_FS  = 19
TITLE_FS  = 21

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
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.linewidth": 1.3, "font.size": LABEL_FS,
    })

    def style_ax(ax):
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(AXIS)
        ax.tick_params(labelsize=TICK_FS, length=5, width=1.2)
        ax.grid(True, axis="y", alpha=0.9)
        ax.grid(False, axis="x")

    # ---------------------------------------------------------------
    # Fig 5 -- ablation ladder (stacked vertically: each panel gets full width)
    # ---------------------------------------------------------------
    ab = pd.DataFrame(R["ablation"])
    labels = ["Naive", "+ vol\nsizing", "+ defensive\nfallback", "+ trans.\ncosts"]
    XTICK_FS = 16

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), dpi=150)
    x = np.arange(len(ab))

    ax = axes[0]
    ax.bar(x, ab["sharpe"], color=[INK_SOFT, INK_SOFT, SIGNAL, INK], width=0.62)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=XTICK_FS)
    ax.set_ylabel("Sharpe ratio", fontsize=LABEL_FS)
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xlim(-0.62, len(ab) - 0.38)
    style_ax(ax)
    for i, v in enumerate(ab["sharpe"]):
        ax.text(i, v + (0.02 if v >= 0 else -0.04), f"{v:.2f}", ha="center", fontsize=VALUE_FS, color=INK)

    ax = axes[1]
    ax.bar(x, ab["max_drawdown"] * 100, color=[INK_SOFT, INK_SOFT, SIGNAL, INK], width=0.62)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=XTICK_FS)
    ax.set_ylabel("Max drawdown (%)", fontsize=LABEL_FS)
    ax.set_xlim(-0.62, len(ab) - 0.38)
    ax.set_ylim(top=2, bottom=ab["max_drawdown"].min() * 100 * 1.32)
    style_ax(ax)
    for i, v in enumerate(ab["max_drawdown"]):
        ax.text(i, v * 100 - 2.6, f"{v*100:.1f}", ha="center", fontsize=VALUE_FS, va="top", color=INK)

    fig.tight_layout(h_pad=3.2)
    fig.savefig(OUT + f"fig5_ablation_{theme}.png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 6 -- cost sensitivity
    # ---------------------------------------------------------------
    cs = pd.DataFrame(R["cost_sensitivity"])
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=150)
    ax2 = ax.twinx()
    l1, = ax.plot(cs["cost_bps"], cs["sharpe"], color=INK, linewidth=3.0, marker="o", markersize=7, label="Sharpe")
    l2, = ax2.plot(cs["cost_bps"], cs["max_drawdown"] * 100, color=ACCENT, linewidth=2.6, marker="o", markersize=7,
                    linestyle=(0, (4, 2)), label="Max DD")
    l3 = ax.axvline(5, color=AXIS, linewidth=1.4, linestyle=(0, (1, 1.4)), label="Assumed (5bps)")
    ax.set_xlabel("Cost (bps/rebalance)", fontsize=LABEL_FS)
    ax.set_ylabel("Sharpe ratio", fontsize=LABEL_FS)
    ax2.set_ylabel("Max drawdown (%)", fontsize=LABEL_FS, color=ACCENT)
    for s in ["top"]:
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS); ax.spines["bottom"].set_color(AXIS)
    ax2.spines["right"].set_color(ACCENT)
    ax.tick_params(labelsize=TICK_FS, length=5, width=1.2)
    ax2.tick_params(axis="y", colors=ACCENT, labelsize=TICK_FS, length=5, width=1.2)
    ax.grid(True, axis="y", alpha=0.9); ax.grid(False, axis="x")
    ax.legend(handles=[l1, l2, l3], frameon=False, fontsize=LEGEND_FS, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT + f"fig6_cost_sensitivity_{theme}.png", dpi=150, transparent=True)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 7 -- regime breakdown
    # ---------------------------------------------------------------
    rg = pd.DataFrame(R["regimes"])
    short_labels = ["Pre-2019", "2019–20", "COVID\ncrash", "COVID\nrecovery", "2022\nbear", "2023–24"]
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=150)
    x = np.arange(len(rg))
    w = 0.36
    b1 = ax.bar(x - w/2, rg["strategy_return"] * 100, width=w, color=INK, label="Strategy")
    b2 = ax.bar(x + w/2, rg["spy_return"] * 100, width=w, color=ACCENT, label="SPY")
    ax.axhline(0, color=AXIS, linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels[:len(rg)], fontsize=VALUE_FS - 2)
    ax.set_ylabel("Period return (%)", fontsize=LABEL_FS)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="upper left", labelcolor=INK)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + (2.5 if h >= 0 else -4.5),
                     f"{h:.0f}", ha="center", fontsize=VALUE_FS - 3, va="bottom" if h >= 0 else "top", color=INK)
    fig.tight_layout()
    fig.savefig(OUT + f"fig7_regimes_{theme}.png", dpi=150, transparent=True)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 8 -- walk-forward (with SE error bars + SPY OOS reference)
    # reference lines carry their own legend entries instead of floating
    # annotation text, which stays legible when the image is scaled down.
    # ---------------------------------------------------------------
    wf = pd.DataFrame(R["walk_forward"]["grid"]).sort_values("window_days")
    wf["fit_se"] = R["walk_forward"]["fit_se"]
    wf["full_se"] = R["walk_forward"]["full_se"]
    spy_test_sharpe = R["walk_forward"]["spy_test_sharpe"]
    fit_w = R["walk_forward"]["fit_period_winner"]

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=150)
    ax.errorbar(wf["window_days"], wf["fit_sharpe"], yerr=wf["fit_se"], color=INK_SOFT, linewidth=2.2,
                marker="o", markersize=6, linestyle=(0, (3, 2)), capsize=4, elinewidth=1.2,
                label="Fit period (2016–19)")
    ax.errorbar(wf["window_days"], wf["full_sharpe"], yerr=wf["full_se"], color=INK, linewidth=2.8,
                marker="o", markersize=6, capsize=4, elinewidth=1.2, label="Full period (2016–24)")
    ax.axhline(spy_test_sharpe, color=ACCENT, linewidth=2.0, linestyle=(0, (4, 2)),
               label=f"SPY, OOS ({spy_test_sharpe:.2f})")
    ax.axvline(fit_w, color=SIGNAL, linewidth=1.6, linestyle=(0, (1, 1.4)), label=f"Chosen window ({fit_w}d)")
    ax.set_xlabel("Ranking window (days)", fontsize=LABEL_FS)
    ax.set_ylabel("Sharpe ratio", fontsize=LABEL_FS)
    style_ax(ax)
    ax.set_ylim(top=ax.get_ylim()[1] + 0.15)
    ax.legend(frameon=False, fontsize=LEGEND_FS - 2, loc="upper center", labelcolor=INK, ncol=1)
    fig.tight_layout()
    fig.savefig(OUT + f"fig8_walkforward_{theme}.png", dpi=150, bbox_inches="tight", transparent=True)
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

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    ax.bar(centers_bins, counts, width=widths, color=INK_SOFT, alpha=0.85, linewidth=0)
    ax.axvline(0, color=AXIS, linewidth=1.6, label="Zero")
    ax.axvline(ci_lo, color=ACCENT, linewidth=2.0, linestyle=(0, (4, 2)), label="95% CI")
    ax.axvline(ci_hi, color=ACCENT, linewidth=2.0, linestyle=(0, (4, 2)))
    ax.axvline(center, color=INK, linewidth=2.4, label="Observed")
    ax.set_xlabel("Sharpe diff (strategy − SPY)", fontsize=LABEL_FS - 2)
    ax.set_ylabel("Resamples", fontsize=LABEL_FS)
    style_ax(ax)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=LEGEND_FS, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT + f"fig9_significance_{theme}.png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 10 -- deflated Sharpe ratio: DSR vs. assumed number of implicit trials
    # (stacked vertically: each panel gets full width)
    # ---------------------------------------------------------------
    ds = R["deflated_sharpe"]
    sweep = pd.DataFrame(ds["sweep"])

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.6), dpi=150)

    ax = axes[0]
    ax.plot(sweep["N"], sweep["sr0_annualized"], color=ACCENT, linewidth=2.6, marker="o", markersize=6,
            label="Luck bar")
    ax.axhline(ds["sr_hat_annualized"], color=INK, linewidth=2.2, linestyle=(0, (4, 2)),
               label=f"Observed ({ds['sr_hat_annualized']:.2f})")
    ax.set_xscale("log")
    ax.set_ylim(top=ds["sr_hat_annualized"] + 0.38)
    ax.set_xlabel("Assumed trials (N)", fontsize=LABEL_FS - 2)
    ax.set_ylabel("Sharpe (ann.)", fontsize=LABEL_FS - 2)
    ax.set_title("Best-of-N Sharpe by chance", fontsize=TITLE_FS, color=INK)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=LEGEND_FS - 2, loc="upper left", labelcolor=INK)

    ax = axes[1]
    ax.plot(sweep["N"], sweep["dsr"] * 100, color=SIGNAL, linewidth=2.6, marker="o", markersize=6)
    ax.axhline(50, color=AXIS, linewidth=1.2, linestyle=(0, (1, 1.4)))
    ax.axvline(ds["n_actual_trials"], color=INK_SOFT, linewidth=1.6, linestyle=(0, (1, 1.4)),
               label=f"N={ds['n_actual_trials']} tested")
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Assumed trials (N)", fontsize=LABEL_FS - 2)
    ax.set_ylabel("P(edge is real), %", fontsize=LABEL_FS - 2)
    ax.set_title("Deflated Sharpe ratio", fontsize=TITLE_FS, color=INK)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=LEGEND_FS - 2, loc="lower left", labelcolor=INK)

    fig.tight_layout(h_pad=3.4)
    fig.savefig(OUT + f"fig10_deflated_sharpe_{theme}.png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)

print("done")
