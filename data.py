import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from config import RAW, SERIES, START, END, GAPPED


def load():
    cols = {}
    for name, sid in SERIES.items():
        s = pd.read_csv(RAW / f"{sid}.csv", index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
        s.index = s.index.to_period("M")
        cols[name] = s
        print(f"{sid}: {s.index[0]} to {s.index[-1]}, n={len(s)}")
    lv = pd.concat(cols, axis=1).asfreq("M")
    assert lv.index[0] == pd.Period(START, "M") and lv.index[-1] == pd.Period(END, "M")
    return lv


def prep(lv):
    gaps = {c: list(lv.index[lv[c].isna()]) for c in lv}
    lv = np.log(lv)
    for c in GAPPED:
        lv[c] = lv[c].interpolate(limit=1, limit_area="inside")
    assert {c: g for c, g in gaps.items() if g} == {c: [pd.Period("2025-10", "M")] for c in GAPPED}
    assert not lv.isna().any().any()
    d = 100 * lv.diff().dropna()
    d["up"], d["dn"] = d.diesel.clip(lower=0), d.diesel.clip(upper=0)
    return lv, d


def adf_table(lv, d):
    cols = list(lv)
    return pd.DataFrame({
        "p_levels": {c: adfuller(lv[c])[1] for c in cols},
        "p_diffs": {c: adfuller(d[c])[1] for c in cols},
    })