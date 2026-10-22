"""Week 1: build monthly stock/bond/inflation dataset, define inflation regimes,
and compute rolling 36-month stock-bond correlations.

Sources (raw data in ./data/raw):
  - S&P 500 price and dividends: Robert Shiller's monthly data
    (github.com/datasets/s-and-p-500). Prices are monthly averages of daily closes.
  - CPI: U.S. CPI-U, all items, not seasonally adjusted (BLS, via github.com/datasets/cpi-us)
  - 10-year Treasury yield: FRED series GS10, monthly average
    (via github.com/datasets/bond-yields-us-10y)

"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

rc = df["rolling_corr_36m"].dropna()
lines.append("")
lines.append(f"Rolling 36m correlation: min {rc.min():+.2f} ({rc.idxmin():%Y-%m}), "
             f"max {rc.max():+.2f} ({rc.idxmax():%Y-%m}), latest {rc.iloc[-1]:+.2f} ({rc.index[-1]:%Y-%m})")
lines.append(f"Share of months with positive rolling correlation: {(rc > 0).mean():.0%}")

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

# Chart 1: rolling correlation with high-inflation shading
fig, ax = plt.subplots(figsize=(10, 4.5))
shade_high(ax)
ax.plot(rc.index, rc, color=BLUE, lw=2)
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylim(-1, 1)
ax.set_ylabel("Correlation")
ax.set_title("Rolling 36-month correlation: S&P 500 vs 10-year Treasury returns",
             loc="left", fontsize=12, color=INK)
ax.text(0.01, 0.03, f"Shaded: high inflation (YoY CPI > {HIGH_INFL:g}%)",
        transform=ax.transAxes, color=MUTED, fontsize=9)
style(ax)
fig.tight_layout()
fig.savefig(CHARTS / "rolling_correlation.png", dpi=160)
plt.close(fig)

# Chart 2: inflation with regime threshold
fig, ax = plt.subplots(figsize=(10, 3.5))
shade_high(ax)
ax.plot(df.index, df["inflation_yoy_pct"], color=INK, lw=1.5)
ax.axhline(HIGH_INFL, color=ORANGE, lw=1, ls="--")
ax.text(df.index[5], HIGH_INFL + 0.4, f"{HIGH_INFL:g}% threshold", color=MUTED, fontsize=9)
ax.set_ylabel("YoY CPI inflation (%)")
ax.set_title("U.S. inflation and regime split", loc="left", fontsize=12, color=INK)
style(ax)
fig.tight_layout()
fig.savefig(CHARTS / "inflation_regimes.png", dpi=160)
plt.close(fig)

# Chart 3: rolling correlation vs inflation (scatter)
both = df.dropna(subset=["rolling_corr_36m"])
infl_36 = df["inflation_yoy_pct"].rolling(WINDOW).mean().loc[both.index]
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(infl_36, both["rolling_corr_36m"], s=10, color=BLUE, alpha=0.5, lw=0)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel("Average inflation over the same 36 months (%)")
ax.set_ylabel("36-month stock-bond correlation")
ax.set_title("Higher inflation, more positive correlation?", loc="left", fontsize=12, color=INK)
ax.grid(color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(CHARTS / "correlation_vs_inflation.png", dpi=160)
plt.close(fig)
