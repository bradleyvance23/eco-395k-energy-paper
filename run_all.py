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


def lp(y="fafh", controls=CTRL, shocks=("diesel",), horizons=HORIZONS, **kw):
    return run_lp(d, y, list(shocks), controls, horizons, contemp=["dspi"], **kw)


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
    s = lp(horizons=(12, 24), **kw)
    a = asym(horizons=(12, 24), **kw).query("shock == 'up'")
    rows.append(s.assign(variant=name, wald_p=a.wald_p.values)[["variant", "h", "coef", "se", "wald_p", "n"]])
pd.concat(rows).round(3).to_csv(OUT / "robustness.csv", index=False)

rows = []
for y in ["fafh", "ppi_food", "fah"]:
    for name, kw in {"baseline": {}, "drop 2020-21": dict(exclude=COVID)}.items():
        rows.append(lp(y, horizons=(12, 24), **kw).assign(outcome=y, variant=name))
cols = ["outcome", "variant", "h", "coef", "se", "lo", "hi", "n"]
pd.concat(rows)[cols].round(3).to_csv(OUT / "upstream_robust.csv", index=False)

lp("fah").to_csv(OUT / "fah.csv", index=False)
run_lp(d, "diesel", ["brent"], [], HORIZONS).to_csv(OUT / "crude_diesel.csv", index=False)