import numpy as np, pandas as pd
from lp import run_lp


def sim(T=3000, up=0.3, dn=0.3, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=T)
    e = np.zeros(T)
    for t in range(1, T):
        e[t] = rho * e[t - 1] + rng.normal()
    y = up * x.clip(min=0) + dn * x.clip(max=0) + e
    idx = pd.period_range("2000-01", periods=T, freq="M")
    d = pd.DataFrame({"x": x, "y": y, "z": rng.normal(size=T)}, index=idx)
    d["up"], d["dn"] = d.x.clip(lower=0), d.x.clip(upper=0)
    return d


def test_recovers_passthrough():
    out = run_lp(sim(), "y", ["x"], ["z"], [0, 6, 12])
    assert ((out.coef - 0.3).abs() < 4 * out.se).all()


def test_asymmetry():
    out = run_lp(sim(up=0.4, dn=0.1), "y", ["up", "dn"], ["z"], [0, 12], test="up = dn")
    for s, true in [("up", 0.4), ("dn", 0.1)]:
        o = out[out.shock == s]
        assert ((o.coef - true).abs() < 4 * o.se).all()
    assert (out[out.h == 0].wald_p < 0.01).all()


def test_n_and_mask():
    T, h, lags, m = 120, 12, 6, 60
    d = sim(T=T)
    assert run_lp(d, "y", ["x"], ["z"], [h], lags=lags).n.iloc[0] == T - h - lags
    ex = [(str(d.index[m]),) * 2]
    masked = run_lp(d, "y", ["x"], ["z"], [h], lags=lags, exclude=ex)
    assert masked.n.iloc[0] == sum(not (t - lags <= m <= t + h) for t in range(lags, T - h))