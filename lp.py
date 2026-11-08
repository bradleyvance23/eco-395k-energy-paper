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


def run_lp(d, y, shocks, controls, horizons, lags=6, contemp=(),
           months=True, test=None, exclude=()):
    X = make_X(d, y, shocks, controls, contemp, lags, months)
    flag = pd.Series(0.0, index=d.index)
    for a, b in exclude:
        flag.loc[a:b] = 1.0
    rows = []
    for h in horizons:
        dep = d[y].rolling(h + 1).sum().shift(-h)          # log y(t+h) - log y(t-1)
        df = pd.concat([dep.rename("dep"), X], axis=1).dropna()
        touch = flag.rolling(h + lags + 1, min_periods=1).max().shift(-h)
        df = df[touch[df.index] == 0]                      # window t-lags .. t+h
        res = sm.OLS(df["dep"], sm.add_constant(df.drop(columns="dep"))).fit(
            cov_type="HAC", cov_kwds={"maxlags": h + 1})
        ci = res.conf_int()
        p = float(res.wald_test(test, use_f=False, scalar=True).pvalue) if test else np.nan
        for s in shocks:
            rows.append(dict(h=h, shock=s, coef=res.params[s], se=res.bse[s],
                             lo=ci.loc[s, 0], hi=ci.loc[s, 1], wald_p=p, n=int(res.nobs)))
    return pd.DataFrame(rows)