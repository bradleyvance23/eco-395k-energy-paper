import pandas as pd
import matplotlib.pyplot as plt
from config import OUT

rows = [("Brent to diesel", "crude_diesel"), ("Diesel to PPI food inputs", "upstream"),
        ("Diesel to food at home", "fah"), ("Diesel to food away from home", "baseline")]

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(6.5, 2.4))
fig.subplots_adjust(left=0.29, right=0.76, bottom=0.2, top=0.98)

for i, (name, f) in enumerate(rows):
    r = pd.read_csv(OUT / f"{f}.csv").set_index("h")
    for h, off, fill in [(12, -0.2, "black"), (24, 0.2, "white")]:
        c, lo, hi = r.loc[h, ["coef", "lo", "hi"]]
        ax.errorbar(c, i + off, xerr=[[c - lo], [hi - c]], fmt="o", color="black",
                    mfc=fill, ms=4, capsize=2, lw=1, label=f"h = {h}" if i == 0 else None)
        ax.text(1.03, i + off, f"{c:.3f} [{lo:.3f}, {hi:.3f}]", transform=ax.get_yaxis_transform(),
                fontsize=8, va="center", ha="left")

ax.axvline(0, color="0.4", lw=0.5)
ax.axvline(1, color="0.4", lw=0.8, ls="--")
ax.set_xlim(-0.15, 1.25)
ax.set_ylim(3.5, -0.5)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([n for n, _ in rows], fontsize=8)
ax.set_xlabel("% change per 1% shock (95% band)", fontsize=8)
ax.tick_params(axis="x", labelsize=8)
ax.legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(0.80, 0.25))
fig.savefig(OUT / "fig3.png", dpi=300)