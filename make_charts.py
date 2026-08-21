import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---- palettes, pulled directly from the page's own --ink/--accent/--signal/--line
# tokens for each theme, so the charts blend into the page instead of sitting in a
# fixed-color box regardless of theme. ----
PALETTES = {
    "light": dict(INK="#1b2320", INK_SOFT="#4a544e", ACCENT="#a35f1b",
                  SIGNAL="#3f6e52", GRID="#cad0c4", AXIS="#a9b1a1"),
    "dark":  dict(INK="#e9ede6", INK_SOFT="#a9b1a4", ACCENT="#dd9a4d",
                  SIGNAL="#6fa383", GRID="#2b322d", AXIS="#3c453d"),
}

df = pd.read_csv("/mnt/user-data/uploads/Desktop/S&P project/archive/_git_cleanup/performance_export.csv",
                  parse_dates=["date"])
df = df.set_index("date").sort_index()

OUT = "/home/claude/sp_site/"

RF = 0.02
def sharpe_ratio(r, rf=RF):
    r = pd.Series(r).dropna()
    if r.std() == 0 or len(r) < 2:
        return np.nan
    excess = r - rf / 252
    return np.sqrt(252) * excess.mean() / excess.std()

roll_net_cache = df["ret_net"].rolling(252).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False)
roll_spy_cache = df["ret_spy"].rolling(252).apply(lambda x: sharpe_ratio(pd.Series(x)), raw=False)

np.random.seed(42)
r_arr = df["ret_net"].dropna().values
n = len(r_arr)
block_size = 20
n_sims = 20000
fan_sample = 2000
n_blocks = int(np.ceil(n / block_size))
offsets = np.arange(block_size)
starts = np.random.randint(0, n - block_size + 1, size=(n_sims, n_blocks))
idx = starts[:, :, None] + offsets[None, None, :]
batch = r_arr[idx].reshape(n_sims, n_blocks * block_size)[:, :n]
cum = np.cumprod(1 + batch, axis=1)
fan_paths = cum[:fan_sample]
pct_bands = np.percentile(fan_paths, [5, 25, 50, 75, 95], axis=0)
realized = (1 + r_arr).cumprod()
x_days = np.arange(n)

dd_net = df["cum_net"] / df["cum_net"].cummax() - 1
dd_spy = df["cum_spy"] / df["cum_spy"].cummax() - 1

for theme, pal in PALETTES.items():
    INK, INK_SOFT, ACCENT, SIGNAL, GRID, AXIS = (
        pal["INK"], pal["INK_SOFT"], pal["ACCENT"], pal["SIGNAL"], pal["GRID"], pal["AXIS"])

    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Georgia", "serif"],
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.linewidth": 0.8,
        "font.size": 12,
    })

    def style_ax(ax):
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(AXIS)
        ax.tick_params(labelsize=10.5, length=3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, axis="y", alpha=0.9)
        ax.grid(False, axis="x")

    # Fig 1 -- growth of $1, gross vs net vs SPY
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=180)
    ax.plot(df.index, df["cum_gross"], color=INK_SOFT, linewidth=1.6, linestyle=(0, (1, 1.4)), label="Strategy (gross)")
    ax.plot(df.index, df["cum_net"], color=INK, linewidth=2.2, label="Strategy (net of cost)")
    ax.plot(df.index, df["cum_spy"], color=ACCENT, linewidth=2.0, label="SPY")
    ax.set_ylabel("Growth of $1")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left", handlelength=1.6, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT + f"fig1_growth_{theme}.png", dpi=180, transparent=True)
    plt.close(fig)

    # Fig 2 -- drawdown, net vs SPY
    fig, ax = plt.subplots(figsize=(9.6, 4.0), dpi=180)
    ax.fill_between(df.index, dd_net * 100, 0, color=SIGNAL, alpha=0.2, linewidth=0)
    ax.plot(df.index, dd_net * 100, color=SIGNAL, linewidth=2.0, label="Strategy (net)")
    ax.plot(df.index, dd_spy * 100, color=ACCENT, linewidth=1.6, alpha=0.9, label="SPY")
    ax.set_ylabel("Drawdown (%)")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=10.5, loc="lower left", handlelength=1.6, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT + f"fig2_drawdown_{theme}.png", dpi=180, transparent=True)
    plt.close(fig)

    # Fig 3 -- rolling 1y sharpe, net vs SPY
    fig, ax = plt.subplots(figsize=(9.6, 4.0), dpi=180)
    ax.axhline(0, color=AXIS, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.plot(df.index, roll_net_cache, color=INK, linewidth=2.0, label="Strategy (net)")
    ax.plot(df.index, roll_spy_cache, color=ACCENT, linewidth=1.8, label="SPY")
    ax.set_ylabel("Rolling 1Y Sharpe")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left", handlelength=1.6, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT + f"fig3_sharpe_{theme}.png", dpi=180, transparent=True)
    plt.close(fig)

    # Fig 4 -- monte carlo block-bootstrap fan chart
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=180)
    ax.fill_between(x_days, pct_bands[0], pct_bands[4], color=ACCENT, alpha=0.18, linewidth=0, label="5th–95th pct")
    ax.fill_between(x_days, pct_bands[1], pct_bands[3], color=ACCENT, alpha=0.32, linewidth=0, label="25th–75th pct")
    ax.plot(x_days, pct_bands[2], color=ACCENT, linewidth=1.4, linestyle=(0, (1, 1.4)), label="Median simulated path")
    ax.plot(x_days, realized, color=INK, linewidth=2.2, label="Realized (net of cost)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Trading day")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(labelsize=10.5, length=3)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=10, loc="upper left", handlelength=1.6, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT + f"fig4_montecarlo_{theme}.png", dpi=180, transparent=True)
    plt.close(fig)

print("done")
print("realized terminal:", realized[-1], "sim median terminal:", pct_bands[2][-1])
