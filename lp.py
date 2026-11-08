import numpy as np, pandas as pd, statsmodels.api as sm


def make_X(d, y, shocks, controls, contemp, lags, months):
    X = d[list(shocks) + list(contemp)].copy()
    for c in dict.fromkeys([y, *shocks, *controls]):
        for L in range(1, lags + 1):
            X[f"{c}_L{L}"] = d[c].shift(L)
    if months:
        m = pd.Series(d.index.month, index=d.index)
        X = X.join(pd.get_dummies(m, prefix="m", drop_first=True, dtype=float))
    return X


def flag_months(index, exclude):
    flag = pd.Series(0.0, index=index)
    for a, b in exclude or []:
        flag[(index >= pd.Period(a, "M")) & (index <= pd.Period(b, "M"))] = 1.0
    return flag


def run_lp(d, y, shocks, controls, horizons, lags=6, contemp=(),
           months=True, test=None, exclude=None):
    assert len(d) == (d.index[-1] - d.index[0]).n + 1, "gaps in monthly index"
    X = make_X(d, y, shocks, controls, contemp, lags, months)
    flag = flag_months(d.index, exclude)
    rows = []
    for h in horizons:
        dep = d[y].rolling(h + 1).sum().shift(-h)          # log y(t+h) - log y(t-1)
        data = pd.concat([dep.rename("dep"), X], axis=1).dropna()
        touch = flag.rolling(h + lags + 1, min_periods=1).max().shift(-h)
        data = data[touch.loc[data.index] == 0]            # window t-lags .. t+h
        Z = sm.add_constant(data.drop(columns="dep"))
        res = sm.OLS(data["dep"], Z).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        ci = res.conf_int()
        p = float(res.wald_test(test, use_f=False, scalar=True).pvalue) if test else np.nan
        for s in shocks:
            rows.append(dict(h=h, shock=s, coef=res.params[s], se=res.bse[s],
                             lo=ci.loc[s, 0], hi=ci.loc[s, 1], wald_p=p, n=int(res.nobs)))
    return pd.DataFrame(rows)