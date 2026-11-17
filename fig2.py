import pandas as pd
import matplotlib.pyplot as plt
from config import OUT

panels = [("crude_diesel", "Brent to diesel"), ("upstream", "Diesel to PPI food inputs"),
          ("fah", "Diesel to food at home"), ("baseline", "Diesel to FAFH")]

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
fig, ax = plt.subplots(1, 4, figsize=(6.5, 2.1))
ax[2].sharey(ax[1])
ax[3].sharey(ax[1])

for a, (f, title) in zip(ax, panels):
    r = pd.read_csv(OUT / f"{f}.csv")
    a.fill_between(r.h, r.lo, r.hi, color="0.8")
    a.plot(r.h, r.coef, color="black", lw=1.2)
    a.axhline(0, color="0.4", lw=0.5)
    a.set_title(title, fontsize=7)
    a.set_xlabel("Months after shock", fontsize=7)
    a.tick_params(labelsize=7)

ax[0].axhline(1, color="0.4", lw=0.8, ls="--")
ax[0].set_ylabel("% change per 1% shock", fontsize=7)
plt.setp(ax[2].get_yticklabels(), visible=False)
plt.setp(ax[3].get_yticklabels(), visible=False)
plt.tight_layout()
fig.savefig(OUT / "fig2.png", dpi=300)