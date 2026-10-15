"""local tracker for the v4 momentum book.

python track.py
then open http://127.0.0.1:8765
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import yfinance as yf

import multi_asset_v4 as v4

PORT = 8765


def load_prices():
    raw = yf.download(v4.tickers, start=v4.start_date, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()
    prices = prices.dropna(how="all").ffill().dropna()
    cols = [c for c in v4.tickers if c in prices.columns]
    if not cols or prices.empty:
        raise RuntimeError("no prices came back")
    return prices[cols]


def run_books(prices):
    returns, rel, vol, blend = v4.compute_signals(prices)
    valid = v4.get_valid_dates(prices, blend, rel)
    monthly = v4.build_monthly_weights(prices, blend, rel, vol, valid)
    rets = v4.build_extended_returns(returns)
    daily = v4.apply_weights(monthly, rets)
    gross = (daily * rets).sum(axis=1)
    net = v4.apply_costs_to_returns(gross, v4.transaction_costs(monthly), rets.index)
    bench = v4.equal_weight_returns(prices, valid, returns)
    return monthly, daily, net, bench


def weights_on(row):
    held = row[row > 1e-8].sort_values(ascending=False)
    return [{"ticker": t, "weight": float(w)} for t, w in held.items()]


def since(row):
    changed = row.diff().abs().sum(axis=1).fillna(0) > 1e-8
    hits = changed[changed]
    if hits.empty:
        return None
    return str(hits.index[-1].date())


def window_stats(model, bench):
    both = pd.concat([model.rename("m"), bench.rename("b")], axis=1).dropna()
    if len(both) < 2:
        return None
    years = (both.index[-1] - both.index[0]).days / 365.25

    def pack(r):
        cum = (1 + r).cumprod()
        return {
            "return": float(cum.iloc[-1] - 1),
            "cagr": float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else None,
            "sharpe": float(v4.sharpe_ratio(r)),
            "max_drawdown": float(v4.max_drawdown(cum)),
        }

    return {"model": pack(both["m"]), "equal_weight": pack(both["b"])}


def snapshot():
    prices = load_prices()
    monthly, daily, net, bench = run_books(prices)
    last = prices.index[-1]
    live = daily.loc[last]
    invested = daily.sum(axis=1)
    first = invested[invested > 1e-8].index[0]

    weekly = pd.concat(
        [(1 + net).cumprod().rename("model"), (1 + bench).cumprod().rename("equal_weight")],
        axis=1,
    ).dropna()
    weekly = weekly.loc[first:].resample("W-FRI").last().dropna()

    ytd = str(last.year) + "-01-01"
    return {
        "as_of": str(last.date()),
        "in_force_since": since(daily),
        "holdings": weights_on(live),
        "latest_signal": {
            "date": str(monthly.index[-1].date()),
            "holdings": weights_on(monthly.iloc[-1]),
        },
        "stats": {
            "since_start": window_stats(net.loc[first:], bench.loc[first:]),
            "ytd": window_stats(net.loc[ytd:], bench.loc[ytd:]),
        },
        "series": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "model": float(row.model),
                "equal_weight": float(row.equal_weight),
            }
            for d, row in weekly.iterrows()
        ],
    }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Momentum book</title>
<style>
  :root {
    --ink: #1b2320;
    --soft: #4a544e;
    --accent: #a35f1b;
    --signal: #3f6e52;
    --line: #d7ddd4;
    --paper: #f4f6f1;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font: 16px/1.45 Georgia, "Iowan Old Style", serif;
  }
  main { max-width: 880px; margin: 0 auto; padding: 32px 20px 64px; }
  header { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; }
  h1 { font-size: 28px; font-weight: normal; margin: 0; }
  .asof { color: var(--soft); font-size: 14px; }
  button {
    font: inherit;
    background: transparent;
    color: var(--ink);
    border: 1px solid var(--line);
    padding: 4px 12px;
    cursor: pointer;
  }
  .holdings { display: grid; gap: 10px; margin: 28px 0 8px; }
  .row { display: grid; grid-template-columns: 64px 1fr 64px; gap: 10px; align-items: center; }
  .bar { height: 8px; background: #e3e8df; }
  .bar > span { display: block; height: 100%; background: var(--signal); }
  .wt { text-align: right; font-variant-numeric: tabular-nums; }
  .note { color: var(--soft); font-size: 14px; margin: 8px 0 28px; }
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { border-top: 1px solid var(--line); padding-top: 12px; }
  .card h2 { font-size: 15px; font-weight: normal; color: var(--soft); margin: 0 0 10px; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th { text-align: right; font-weight: normal; color: var(--soft); font-size: 13px; }
  th:first-child, td:first-child { text-align: left; }
  td { padding: 3px 0; }
  canvas { width: 100%; height: 320px; margin-top: 28px; display: block; }
  .legend { display: flex; gap: 18px; color: var(--soft); font-size: 14px; margin-top: 8px; }
  .swatch { display: inline-block; width: 18px; height: 3px; margin-right: 6px; vertical-align: middle; }
  .err { color: #8a2e2e; }
  @media (max-width: 640px) { .cards { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <header>
    <h1>Momentum book</h1>
    <div class="asof" id="asof">loading prices…</div>
  </header>
  <div class="holdings" id="holdings"></div>
  <p class="note" id="note"></p>
  <div class="cards" id="cards"></div>
  <canvas id="chart" width="880" height="320"></canvas>
  <div class="legend">
    <span><i class="swatch" style="background:#1b2320"></i>Model</span>
    <span><i class="swatch" style="background:#a35f1b"></i>Equal weight</span>
  </div>
</main>
<script>
const pct = (x) => (x * 100).toFixed(1) + "%";
const num = (x) => (x == null || Number.isNaN(x)) ? "—" : x.toFixed(2);

function card(title, block) {
  const rows = [
    ["Return", pct(block.model.return), pct(block.equal_weight.return)],
    ["CAGR", pct(block.model.cagr), pct(block.equal_weight.cagr)],
    ["Sharpe", num(block.model.sharpe), num(block.equal_weight.sharpe)],
    ["Worst loss", pct(block.model.max_drawdown), pct(block.equal_weight.max_drawdown)],
  ];
  return `<section class="card"><h2>${title}</h2><table>
    <tr><th></th><th>Model</th><th>Equal weight</th></tr>
    ${rows.map(r => `<tr><td>${r[0]}</td><td style="text-align:right">${r[1]}</td><td style="text-align:right">${r[2]}</td></tr>`).join("")}
  </table></section>`;
}

function draw(series) {
  const canvas = document.getElementById("chart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 320;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const pad = {l: 44, r: 12, t: 12, b: 28};
  const ys = series.flatMap(d => [d.model, d.equal_weight]);
  const min = Math.min(...ys) * 0.98;
  const max = Math.max(...ys) * 1.02;
  const X = i => pad.l + (w - pad.l - pad.r) * (i / (series.length - 1));
  const Y = v => pad.t + (h - pad.t - pad.b) * (1 - (v - min) / (max - min));
  g.strokeStyle = "#d7ddd4";
  g.fillStyle = "#4a544e";
  g.font = "12px Georgia";
  g.textAlign = "right";
  for (let t = 0; t < 4; t++) {
    const v = min + (max - min) * (t / 3);
    const y = Y(v);
    g.beginPath();
    g.moveTo(pad.l, y);
    g.lineTo(w - pad.r, y);
    g.stroke();
    g.fillText(v.toFixed(2), pad.l - 6, y + 4);
  }
  g.textAlign = "center";
  const ticks = [0, Math.floor(series.length / 2), series.length - 1];
  ticks.forEach(i => g.fillText(series[i].date.slice(0, 4), X(i), h - 8));
  function line(key, color) {
    g.beginPath();
    series.forEach((d, i) => i ? g.lineTo(X(i), Y(d[key])) : g.moveTo(X(i), Y(d[key])));
    g.strokeStyle = color;
    g.lineWidth = 1.8;
    g.stroke();
  }
  line("equal_weight", "#a35f1b");
  line("model", "#1b2320");
}

function show(data) {
  window.__data = data;
  document.getElementById("asof").innerHTML =
    `as of ${data.as_of} <button id="reload">refresh</button>`;
  document.getElementById("reload").onclick = () => location.reload();
  document.getElementById("holdings").innerHTML = data.holdings.map(h => `
    <div class="row">
      <div>${h.ticker}</div>
      <div class="bar"><span style="width:${(h.weight * 100).toFixed(1)}%"></span></div>
      <div class="wt">${pct(h.weight)}</div>
    </div>`).join("");
  const signal = data.latest_signal.holdings.map(h => `${h.ticker} ${pct(h.weight)}`).join(", ");
  document.getElementById("note").textContent =
    `In force since ${data.in_force_since}. Latest close would switch to ${signal}, starting the next session. Compared with an equal weight of the same five ETFs. Costs included.`;
  document.getElementById("cards").innerHTML =
    card("Since the rule turned on", data.stats.since_start) +
    card("This year", data.stats.ytd);
  draw(data.series);
}

fetch("/api/snapshot")
  .then(r => r.json())
  .then(show)
  .catch(err => {
    document.getElementById("asof").textContent = "could not load";
    document.getElementById("note").innerHTML = `<span class="err">${err}</span>`;
  });
window.addEventListener("resize", () => {
  if (window.__data) draw(window.__data.series);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/snapshot"):
            try:
                body = json.dumps(snapshot()).encode()
                status = 200
                ctype = "application/json"
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                status = 500
                ctype = "application/json"
        else:
            body = PAGE.encode()
            status = 200
            ctype = "text/html; charset=utf-8"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(fmt % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"tracking on http://127.0.0.1:{PORT}")
    server.serve_forever()
