import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller
from config import RAW, SERIES, START, END, GAPPED


def load():
    files = {n: pd.read_csv(RAW / f"{sid}.csv", index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
             for n, sid in SERIES.items()}
    lv = pd.concat(files, axis=1).to_period("M")
    assert lv.index[0] == pd.Period(START, "M") and lv.index[-1] == pd.Period(END, "M")
    return lv


def prep(lv):
    assert list(lv.index[lv.isna().any(axis=1)]) == [pd.Period("2025-10", "M")]
    assert not lv.drop(columns=GAPPED).isna().any().any()
    lv = np.log(lv)
    lv[GAPPED] = lv[GAPPED].interpolate(limit=1, limit_area="inside")
    d = 100 * lv.diff().dropna()
    d["up"], d["dn"] = d.diesel.clip(lower=0), d.diesel.clip(upper=0)
    return lv, d


def adf_table(lv, d):
    p = lambda s: adfuller(s)[1]
    return pd.DataFrame({"p_levels": lv.apply(p), "p_diffs": d[lv.columns].apply(p)})