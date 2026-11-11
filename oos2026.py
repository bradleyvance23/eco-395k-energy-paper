import numpy as np, pandas as pd
from data import load, prep
from lp import run_lp
from config import OUT

lv, d = prep(load())
fit = run_lp(d.loc[:"2026-02"], "fafh", ["diesel"], ["core", "wage", "dspi"],
             range(7), contemp=["dspi"])

x = d.diesel.loc["2026-03":].values
up, dn = x.clip(min=0), x.clip(max=0)

out = pd.DataFrame({"month": d.loc["2026-03":].index.astype(str)})
out["pred"] = np.convolve(x, fit.coef)[:7]
out["lo"] = np.convolve(up, fit.lo)[:7] + np.convolve(dn, fit.hi)[:7]
out["hi"] = np.convolve(up, fit.hi)[:7] + np.convolve(dn, fit.lo)[:7]
out["actual"] = 100 * (lv.fafh.loc["2026-03":] - lv.fafh.loc["2026-02"]).values
for w in (6, 12, 24):
    out[f"gap{w}"] = out.actual - d.fafh.loc[:"2026-02"].tail(w).mean() * np.arange(1, 8)
out["gap0"] = out.actual

print(out.round(2).to_string(index=False))
out.round(3).to_csv(OUT / "oos2026.csv", index=False)