"""Week 1: build monthly stock/bond/inflation dataset, define inflation regimes,
and compute rolling 36-month stock-bond correlations.

Sources (raw data in ./data/raw):
  - S&P 500 price and dividends: Robert Shiller's monthly data
    (github.com/datasets/s-and-p-500). Prices are monthly averages of daily closes.
  - CPI: U.S. CPI-U, all items, not seasonally adjusted (BLS, via github.com/datasets/cpi-us)
  - 10-year Treasury yield: FRED series GS10, monthly average
    (via github.com/datasets/bond-yields-us-10y)

"""
from math import erfc, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW, OUT, CHARTS = ROOT / "data/raw", ROOT / "data/processed", ROOT / "charts"
for folder in (RAW, OUT, CHARTS):
    folder.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "shiller_sp500.csv": "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
    "cpi_us.csv": "https://raw.githubusercontent.com/datasets/cpi-us/main/data/cpiai.csv",
    "treasury_10y_yield.csv": "https://raw.githubusercontent.com/datasets/bond-yields-us-10y/main/data/monthly.csv",
}
for name, url in SOURCES.items():
    if not (RAW / name).exists():
        print(f"Downloading {name} ...")
        pd.read_csv(url).to_csv(RAW / name, index=False)

WINDOW = 36            # rolling correlation window, months
HIGH_INFL = 3.0        # year-over-year CPI inflation (%) above which a month is "high inflation"
MIN_SPELL = 6         # only shade high-inflation spells at least this many months long

BUCKETS = [-np.inf, 2, 3, 4, 6, np.inf]
BUCKET_LABELS = ["Under 2%", "2–3%", "3–4%", "4–6%", "Over 6%"]

# ---------- load ----------
spx = pd.read_csv(RAW / "shiller_sp500.csv", parse_dates=["Date"], index_col="Date")
cpi = pd.read_csv(RAW / "cpi_us.csv", parse_dates=["Date"], index_col="Date")["Index"]
y10 = pd.read_csv(RAW / "treasury_10y_yield.csv", parse_dates=["Date"], index_col="Date")["Rate"]

# ---------- stock total return ----------
# Shiller dividends end mid-2023 (zeros after); carry the last annual dividend forward.
div = spx["Dividend"].replace(0, np.nan).ffill()
price = spx["SP500"]
stock_ret = (price + div / 12) / price.shift(1) - 1

# ---------- bond total return from yield ----------
# Buy a 10-year par bond at last month's yield, sell it a month later
# (9 11/12 years left) at this month's yield. Semi-annual coupons.
def par_bond_return(y_prev, y_now, maturity=10.0):
    c, y = y_prev / 100, y_now / 100
    n = 2 * (maturity - 1 / 12)
    price = c / y * (1 - (1 + y / 2) ** -n) + (1 + y / 2) ** -n
    return price - 1 + c / 12

bond_ret = par_bond_return(y10.shift(1), y10)

# ---------- inflation ----------
infl_yoy = cpi.pct_change(12) * 100

df = pd.DataFrame({
    "sp500_price": price,
    "sp500_total_return": stock_ret,
    "treasury_10y_yield": y10,
    "treasury_10y_return": bond_ret,
    "cpi": cpi,
    "inflation_yoy_pct": infl_yoy,
}).dropna()

df["regime"] = np.where(df["inflation_yoy_pct"] > HIGH_INFL, "high", "low")
df["rolling_corr_36m"] = df["sp500_total_return"].rolling(WINDOW).corr(df["treasury_10y_return"])

df.to_csv(OUT / "monthly_dataset.csv", float_format="%.5f", index_label="date")
STOCK, BOND = "sp500_total_return", "treasury_10y_return"

# ---------- summary ----------
lines = []
start, end = df.index.min(), df.index.max()
lines.append(f"Sample: {start:%Y-%m} to {end:%Y-%m} ({len(df)} monthly returns, "
             f"{(end - start).days / 365.25:.1f} years)")
lines.append(f"High-inflation threshold: YoY CPI > {HIGH_INFL}%")
lines.append("")

# Calculate correlations by regime
full = df["sp500_total_return"].corr(df["treasury_10y_return"])
lines.append(f"Full-sample stock-bond correlation: {full:+.2f}")
for reg in ["high", "low"]:
    sub = df[df["regime"] == reg]
    c = sub["sp500_total_return"].corr(sub["treasury_10y_return"])
    lines.append(f"  {reg:>4} inflation: {len(sub):4d} months ({len(sub)/len(df):.0%}), "
                 f"correlation {c:+.2f}, avg inflation {sub['inflation_yoy_pct'].mean():.1f}%")
lines.append("")

# contiguous regime spells
spell_id = (df["regime"] != df["regime"].shift()).cumsum()
spells = df.groupby(spell_id).agg(regime=("regime", "first"),
                                  start=("regime", lambda s: s.index.min()),
                                  end=("regime", lambda s: s.index.max()),
                                  months=("regime", "size"))
lines.append("High-inflation spells of 12+ months:")
for _, s in spells[(spells.regime == "high") & (spells.months >= 12)].iterrows():
    sub = df.loc[s.start:s.end]
    c = sub["sp500_total_return"].corr(sub["treasury_10y_return"])
    lines.append(f"  {s.start:%Y-%m} to {s.end:%Y-%m} ({s.months} mo): correlation {c:+.2f}")
lines.append("")

# by decade
lines.append("Correlation by decade:")
for dec, sub in df.groupby((df.index.year // 10) * 10):
    c = sub["sp500_total_return"].corr(sub["treasury_10y_return"])
    lines.append(f"  {dec}s: {c:+.2f}  (avg inflation {sub['inflation_yoy_pct'].mean():.1f}%)")


summary = "\n".join(lines)
(OUT / "week1_summary.txt").write_text(summary + "\n")
print(summary)

# ---------- charts ----------
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})

def shade_high(ax):
    for _, s in spells[spells.regime == "high"].iterrows():
        ax.axvspan(s.start, s.end + pd.offsets.MonthEnd(0), color=ORANGE, alpha=0.12, lw=0)

def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0)


def corr_ci(sub):
    """Correlation, 95% confidence interval (Fisher z), and number of months."""
    r, n = sub[STOCK].corr(sub[BOND]), len(sub)
    z, se = np.arctanh(r), 1 / sqrt(n - 3)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), n
 
df["bucket"] = pd.cut(df["inflation_yoy_pct"], BUCKETS, labels=BUCKET_LABELS)
buckets = pd.DataFrame([corr_ci(g) for _, g in df.groupby("bucket", observed=True)],
                       index=BUCKET_LABELS, columns=["r", "ci_low", "ci_high", "months"])
 
high, low = df[df["regime"] == "high"], df[df["regime"] == "low"]
r_high, _, _, n_high = corr_ci(high)
r_low, _, _, n_low = corr_ci(low)
z_stat = (np.arctanh(r_high) - np.arctanh(r_low)) / sqrt(1 / (n_high - 3) + 1 / (n_low - 3))
p_value = erfc(abs(z_stat) / sqrt(2))   # two-sided
 
lines = ["Correlation of monthly S&P 500 and 10-year Treasury returns by inflation level",
         f"{'Inflation':<10} {'Months':>6} {'Corr':>6}   95% CI"]
for label, b in buckets.iterrows():
    lines.append(f"{label:<10} {b.months:>6.0f} {b.r:>+6.2f}   [{b.ci_low:+.2f}, {b.ci_high:+.2f}]")
lines += ["",
          f"High inflation (> {HIGH_INFL:g}%): {r_high:+.2f} over {n_high} months",
          f"Low inflation (<= {HIGH_INFL:g}%): {r_low:+.2f} over {n_low} months",
          f"Fisher z-test for the difference: z = {z_stat:.2f}, p = {p_value:.1g}"]
summary = "\n".join(lines)
(OUT / "key_charts_stats.txt").write_text(summary + "\n")
print(summary)
 
 
# ---------- chart style ----------
RED, BLUE, INK, MUTED, GRID, SHADE = "#d6452a", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})
 
 
# ---------- Chart 1: rolling correlation, red above zero / blue below ----------
rc = df["rolling_corr_36m"].dropna()
x, y = mdates.date2num(rc.index), rc.to_numpy()
 
# Insert a point exactly where the line crosses zero, so the colour switches there.
xs, ys = [x[0]], [y[0]]
for i in range(1, len(x)):
    if y[i - 1] * y[i] < 0:
        xs.append(x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i]))
        ys.append(0.0)
    xs.append(x[i])
    ys.append(y[i])
xs, ys = np.array(xs), np.array(ys)
segments = np.stack([np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1)
colors = np.where((ys[:-1] + ys[1:]) / 2 >= 0, RED, BLUE)
 
spell_id = (df["regime"] != df["regime"].shift()).cumsum()
spells = df.groupby(spell_id).agg(regime=("regime", "first"),
                                  start=("regime", lambda s: s.index.min()),
                                  end=("regime", lambda s: s.index.max()),
                                  months=("regime", "size"))
long_high = spells[(spells.regime == "high") & (spells.months >= MIN_SPELL)]
 
fig, ax = plt.subplots(figsize=(10, 4.8))
for _, s in long_high.iterrows():
    ax.axvspan(s.start, s.end + pd.offsets.MonthEnd(0), color=SHADE, alpha=0.18, lw=0)
ax.add_collection(LineCollection(segments, colors=colors, linewidths=2.4, capstyle="round"))
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(-1, 1)
ax.set_ylabel("36-month correlation")
ax.set_title("Bonds stop hedging stocks when inflation is high",
             loc="left", fontsize=13, color=INK, fontweight="bold", pad=18)
ax.text(0, 1.01, "Rolling 36-month correlation of S&P 500 and 10-year Treasury monthly returns, 1956–2026",
        transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
 
notes = [("1970s–80s high inflation:\nstocks and bonds move together", "1969-01-01", 0.78, RED),
         ("2000–2020 low inflation:\nbonds hedge stocks", "2004-01-01", -0.9, BLUE),
         ("2022: both\nfell together", "2016-06-01", 0.78, RED)]
for text, date, ypos, color in notes:
    ax.text(pd.Timestamp(date), ypos, text, color=color, fontsize=9, va="center")
 
ax.text(0.01, 0.04, f"Gray: high-inflation spells of {MIN_SPELL}+ months (CPI > {HIGH_INFL:g}% a year)",
        transform=ax.transAxes, color=MUTED, fontsize=8.5)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.xaxis.set_major_locator(mdates.YearLocator(10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(CHARTS / "rolling_correlation_signed.png", dpi=160)
plt.close(fig)
 
 
# ---------- Chart 2: correlation by inflation bucket ----------
fig, ax = plt.subplots(figsize=(7, 4.8))
pos = np.arange(len(buckets))
bar_colors = [RED if c >= 0 else BLUE for c in buckets["r"]]
err = np.vstack([buckets["r"] - buckets["ci_low"], buckets["ci_high"] - buckets["r"]])
ax.bar(pos, buckets["r"], color=bar_colors, width=0.62)
ax.errorbar(pos, buckets["r"], yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=5)
ax.axhline(0, color=MUTED, lw=1)
 
for p, (_, b) in zip(pos, buckets.iterrows()):
    above = b.r >= 0
    ax.text(p, (b.ci_high + 0.03) if above else (b.ci_low - 0.03), f"{b.r:+.2f}",
            ha="center", va="bottom" if above else "top", fontsize=10, color=INK, fontweight="bold")
    ax.text(p, -0.6, f"{b.months:.0f} months", ha="center", fontsize=8.5, color=MUTED)
 
ax.set_xticks(pos, buckets.index)
ax.set_xlabel("Year-over-year CPI inflation that month")
ax.set_ylabel("Stock-bond correlation")
ax.set_ylim(-0.65, 0.6)
ax.set_title("Stock-bond correlation rises with inflation", loc="left", fontsize=13,
             color=INK, fontweight="bold", pad=18)
ax.text(0, 1.01, "Monthly returns, May 1953 – Aug 2026. Bars show 95% confidence intervals.",
        transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
ax.text(0.99, 0.16,
        f"High vs low inflation (> {HIGH_INFL:g}%):\n{r_high:+.2f} vs {r_low:+.2f}, z = {z_stat:.1f}, {p_text}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(CHARTS / "correlation_by_inflation.png", dpi=160)
plt.close(fig)
 