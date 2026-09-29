"""Generate a one-year business dataset with a built-in story for the demo.

Story hidden in the data (the hearing should uncover it):
  * From 2025-10-15 Furniture sales in the West region collapse (x0.45)
  * East Electronics ramps up from September (up to x1.5)
  * The Online channel share grows from 30% to 60% over the year, at lower margin
  * One-day promo spike in North on 2025-12-05 (x2.6)

    python scripts/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
dates = pd.date_range("2025-01-01", "2025-12-31")
regions = {"North": 1.0, "South": 0.85, "East": 1.1, "West": 0.95}
cats = {"Electronics": (1.4, 0.72), "Furniture": (1.0, 0.65), "Clothing": (0.8, 0.55), "Groceries": (1.2, 0.85)}
channels = {"Store": 0.0, "Online": 0.06, "Partner": 0.03}   # extra cost ratio

rows = []
for i, d in enumerate(dates):
    season = 1 + 0.15 * np.sin(2 * np.pi * (d.dayofyear - 60) / 365) + (0.25 if d.month in (11, 12) else 0)
    weekend = 1.12 if d.dayofweek >= 5 else 1.0
    p_online = 0.3 + 0.3 * i / len(dates)
    for region, rf in regions.items():
        for cat, (cf, cost_ratio) in cats.items():
            mult = 1.0
            if region == "West" and cat == "Furniture" and d >= pd.Timestamp("2025-10-15"):
                mult *= 0.45
            if region == "East" and cat == "Electronics" and d >= pd.Timestamp("2025-09-01"):
                mult *= 1 + 0.5 * min((d - pd.Timestamp("2025-09-01")).days / 120, 1)
            if region == "North" and d == pd.Timestamp("2025-12-05"):
                mult *= 2.6
            ch = rng.choice(["Online", "Store", "Partner"], p=[p_online, (1 - p_online) * 0.7, (1 - p_online) * 0.3])
            revenue = 1000 * rf * cf * season * weekend * mult * rng.normal(1, 0.08)
            units = max(1, int(revenue / rng.uniform(20, 60)))
            cost = revenue * (cost_ratio + channels[ch])
            rows.append((d.strftime("%Y-%m-%d"), region, cat, ch, units, round(revenue, 2), round(cost, 2), round(revenue - cost, 2)))

df = pd.DataFrame(rows, columns=["date", "region", "category", "channel", "units", "revenue", "cost", "profit"])
root = Path(__file__).resolve().parent.parent
df.to_csv(root / "data" / "sample_business.csv", index=False)
df.to_csv(root / "public" / "sample_business.csv", index=False)
print(f"Wrote {len(df):,} rows")
