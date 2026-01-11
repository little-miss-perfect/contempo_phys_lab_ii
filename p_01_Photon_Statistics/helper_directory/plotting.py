import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
from scipy.stats import poisson


# nos piden obtener el histograma asociado a cada dataframe,
# para luego normalizarlo y obtener la densidad
# (que debe distribuirse como Poisson de parámetro "promedio de fotones")

def fdp_histograma(
    df, *, ax=None, title=None, save_path=None,
    error_bars=True,
    overlay_poisson=False,
    overlay_connect=True,
    poisson_mu=None,
    hist_color="darkmagenta",
    overlay_color="tab:orange",
    errorbar_color="black",
):
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(int)

    # empirical PMF: p_k
    counts = s.value_counts().sort_index()     # n_k
    N = int(s.size)
    fdp = (counts / N)                         # p_k

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # plot PMF as bars
    x = fdp.index.to_numpy()
    y = fdp.to_numpy()
    ax.bar(x, y, width=1.0, align="center", edgecolor="black",
       color=hist_color, alpha=0.7)

    # error bars on p_k
    if error_bars:
        yerr = np.sqrt(counts.to_numpy()) / N   # sqrt(n_k)/N
        ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, color=errorbar_color)

    # --- NEW: Poisson overlay using sample mean ---
    if overlay_poisson:
        mu = float(s.mean()) if poisson_mu is None else float(poisson_mu)  # to use the sample means or a given set of parameters
        k = np.arange(0, int(s.max()) + 1)  # nice full discrete range
        pk = poisson.pmf(k, mu=mu)
        linestyle = "-" if overlay_connect else "none"

        ax.plot(
            k, pk,
            marker="o",
            linestyle=linestyle,
            linewidth=1.2,
            color=overlay_color,
            label=f"Poisson(μ={mu:.3g})",
        )

        ax.legend()

    ax.set_xlabel("número de fotones")
    ax.set_ylabel("probabilidad")
    if title:
        ax.set_title(title)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fdp


# y para graficar de forma bonita, damos estas condiciones
def best_grid(n: int, *, max_cols: int = 6) -> tuple[int, int]:
    """
    Choose (rows, cols) for n plots.
    - Forces 1×n for n<=3
    - Otherwise picks a grid that balances:
      (a) few empty cells
      (b) not overly wide
      (c) fairly square when possible
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n <= 3:
        return 1, n

    best = None
    best_score = float("inf")

    # Try reasonable column counts
    for cols in range(2, min(max_cols, n) + 1):
        rows = math.ceil(n / cols)
        empty = rows * cols - n
        aspect = cols / rows  # >= ~1 tends wide, <1 tends tall (but cols>=2 so often >=1)

        # Score: prioritize few empties, then avoid extreme wide layouts, then prefer near-square
        score = (
            empty * 5.0 +                 # empties are annoying, but not as bad as insane aspect ratios
            (aspect - 1.3) ** 2 * 4.0 +    # prefer slightly wide grids (1.3-ish) for your figsize choice
            (cols - rows) ** 2 * 0.3       # mild preference for square-ish
        )

        # Tie-breaks: fewer empty, then fewer rows (less tall)
        if score < best_score or (abs(score - best_score) < 1e-12 and (empty, rows) < (best[0], best[1])):
            best_score = score
            best = (rows, cols, empty)

    return best[0], best[1]
