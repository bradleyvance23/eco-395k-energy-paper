import pandas as pd
from config import OUT

rows = [("Brent to diesel", "crude_diesel"), ("Diesel to PPI food inputs", "upstream"),
        ("Diesel to food at home", "fah"), ("Diesel to food away from home", "baseline")]
H = [3, 6, 12, 24]

t, n = {}, {}
for name, f in rows:
    r = pd.read_csv(OUT / f"{f}.csv").set_index("h").loc[H]
    t[name] = [f"{c:.3f} [{l:.3f}, {u:.3f}]" for c, l, u in zip(r.coef, r.lo, r.hi)]
    n[name] = list(r.n)

tab = pd.DataFrame(t, index=[f"h={h}" for h in H]).T
tab.to_csv(OUT / "table1.csv")
print(tab.to_string())
print(pd.DataFrame(n, index=[f"h={h}" for h in H]).T)