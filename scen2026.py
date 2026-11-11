import numpy as np, pandas as pd
from data import load, prep
from lp import run_lp
from config import OUT, COVID

lv, d = prep(load())
x = d.diesel.loc["2026-03":].values
up, dn = x.clip(min=0), x.clip(max=0)
N = 25
months = pd.period_range("2026-03", periods=N, freq="M").astype(str)
cum = np.cumsum(np.r_[x, np.zeros(N - len(x))])

rows = []
for y in ["fah", "fafh", "ppi_food"]:
    for case, ex in {"lower (drop 2020-21)": COVID, "upper (baseline)": ()}.items():
        f = run_lp(d, y, ["diesel"], ["core", "wage", "dspi"], range(N),
                   contemp=["dspi"], exclude=ex)
        rows.append(pd.DataFrame({
            "outcome": y, "case": case, "month": months,
            "diesel_pct": 100 * (np.exp(cum / 100) - 1),
            "pred": np.convolve(x, f.coef)[:N],
            "lo": np.convolve(up, f.lo)[:N] + np.convolve(dn, f.hi)[:N],
            "hi": np.convolve(up, f.hi)[:N] + np.convolve(dn, f.lo)[:N]}))
        print(y, case, "coef h=12:", round(f.coef[12], 3))

out = pd.concat(rows)
out.round(3).to_csv(OUT / "scenario2026.csv", index=False)
print(out[out.month.isin(["2026-09", "2027-03", "2027-09", "2028-03"])].round(2).to_string(index=False))