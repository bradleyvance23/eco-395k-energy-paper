import pandas as pd
from config import OUT, HORIZONS, COVID, OCT25
from data import load, prep, adf_table
from lp import run_lp

OUT.mkdir(exist_ok=True)
lv, d = prep(load())
print(d.index[0], d.index[-1], len(d))
print("up months:", (d.diesel > 0).sum(), "down months:", (d.diesel < 0).sum())
adf_table(lv, d).to_csv(OUT / "adf.csv")

CTRL = ["core", "wage", "dspi"]


def lp(y="fafh", controls=CTRL, shocks=("diesel",), **kw):
    return run_lp(d, y, list(shocks), controls, HORIZONS, contemp=["dspi"], **kw)


def asym(**kw):
    return lp(shocks=("up", "dn"), test="up = dn", **kw)


lp().to_csv(OUT / "baseline.csv", index=False)
asym().to_csv(OUT / "asym.csv", index=False)
lp("ppi_food").to_csv(OUT / "upstream.csv", index=False)

variants = {
    "baseline": {},
    "placebo core": dict(y="core", controls=["wage", "dspi"]),
    "drop 2020-21": dict(exclude=COVID),
    "mask Oct 2025": dict(exclude=OCT25),
    "lags 3": dict(lags=3),
    "lags 12": dict(lags=12),
    "no wage": dict(controls=["core", "dspi"]),
    "total wage": dict(controls=["core", "wage_all", "dspi"]),
}

rows = []
for name, kw in variants.items():
    sym, asy = lp(**kw), asym(**kw)
    r = {"variant": name}
    for h in (12, 24):
        r[f"coef_h{h}"] = sym.query("h == @h").coef.iloc[0]
        r[f"wald_p_h{h}"] = asy.query("h == @h").wald_p.iloc[0]
        r[f"n_h{h}"] = sym.query("h == @h").n.iloc[0]
    rows.append(r)
pd.DataFrame(rows).round(3).to_csv(OUT / "robustness.csv", index=False)