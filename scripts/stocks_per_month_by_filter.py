"""
Number of stocks per month under the three universe definitions used across the
backtest notebooks: full universe (no trim), mild trim (bottom 10% by market cap
dropped each month, model_backtest_mild_filter.ipynb), and HXZ microcap exclusion
(below NYSE 20th-percentile market cap dropped each month, model_backtest_hxz_filter.ipynb).

Filter logic here is copied as-is from those two notebooks so the counts line up
with what each backtest actually ran on.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "."
PANEL_FILE = os.path.join(DATA_DIR, "panel_features.parquet")
OUT_FILE = os.path.join(DATA_DIR, "output", "stocks_per_month_by_filter.png")

MICROCAP_TRIM_PCT = 0.10  # mild filter: drop bottom 10% of market cap, each month
HXZ_PCTILE = 0.20         # HXZ filter: drop below NYSE 20th-percentile market cap

panel = pd.read_parquet(PANEL_FILE, columns=["permno", "mthcaldt", "primaryexch", "size"])
mktcap = np.exp(panel["size"])

full_count = panel.groupby("mthcaldt")["permno"].nunique()

mild_cutoff = mktcap.groupby(panel["mthcaldt"]).transform(lambda s: s.quantile(MICROCAP_TRIM_PCT))
mild_count = panel.loc[mktcap >= mild_cutoff].groupby("mthcaldt")["permno"].nunique()

nyse_mktcap = mktcap.where(panel["primaryexch"] == "N")
hxz_cutoff = nyse_mktcap.groupby(panel["mthcaldt"]).transform(lambda s: s.quantile(HXZ_PCTILE))
hxz_count = panel.loc[mktcap >= hxz_cutoff].groupby("mthcaldt")["permno"].nunique()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(full_count.index, full_count.values, label="Full universe", linewidth=1.2)
ax.plot(mild_count.index, mild_count.values, label="Mild trim (bottom 10% mktcap)", linewidth=1.2)
ax.plot(hxz_count.index, hxz_count.values, label="HXZ (NYSE 20th-pct mktcap)", linewidth=1.2)
ax.set_title("Number of Stocks per Month by Universe Filter")
ax.set_xlabel("Date")
ax.set_ylabel("# Stocks")
ax.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=150)
print(f"Saved: {OUT_FILE}")

for name, s in [("Full", full_count), ("Mild", mild_count), ("HXZ", hxz_count)]:
    print(f"{name:5s} — median {s.median():,.0f}, min {s.min():,.0f}, max {s.max():,.0f}")
