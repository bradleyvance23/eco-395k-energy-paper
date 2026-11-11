import numpy as np
from data import load, prep
from config import OUT

lv, d = prep(load())
pre = d.diesel.loc[:"2026-02"]

t = d.diesel.loc["2025-09":].to_frame("chg")
t["level"] = np.exp(lv.diesel.loc["2025-09":])
t["cum"] = 100 * (lv.diesel.loc["2025-09":] - lv.diesel.loc["2026-02"])
print(t.round(2))
t.round(3).to_csv(OUT / "diesel2026.csv")

r7 = pre.rolling(7).sum().dropna()
print("monthly sd", round(pre.std(), 2), "min", round(pre.min(), 2), "max", round(pre.max(), 2))
print("7-month sum, through 2026-02: min", round(r7.min(), 1), "max", round(r7.max(), 1))