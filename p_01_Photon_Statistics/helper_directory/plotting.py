import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

# nos piden obtener el histograma asociado a cada dataframe,
# para luego normalizarlo y obtener la densidad
# (que debe distribuirse como Poisson de parámetro "promedio de fotones")

# def fdp_histograma(df, *, ax=None, title=None, save_path=None):
#     """
#     dado un dataframe (de una columna), grafica la función de densidad mediante un histograma normalizado,
#     y regresa la función de densidad como una serie de Pandas donde
#     los índices son los valores que toma la variable aleatoria discreta y
#     los valores de la serie son las probabilidades asociadas a esos valores.
#     """
#     # redefinimos el dataframe en una serie (ya evitando un par de problemas)
#     s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(int)  # "df.iloc[:, 0]" agarra todas las filas de la columna "0"
#                                                                             # 'errors="coerce"' es para evitar mini irregularidades (cualquier irregularidad las vuelve "NaN")
#                                                                             # ".dropna()" justo quita esas filas dadas por 'errors="coerce"'
#                                                                             # ".astype(int)" lo tenemos por si algo se pasa como un "float" (porque los conteos son, pues, enteros)
#
#     # "relative frequencies (probabilities), sorted by value"
#     fdp = s.value_counts(normalize=True).sort_index()
#                                                        # "value_counts()" cuenta las veces que una entrada distinta aparece (obtiene las frequencias). la documentación dice: 'Return a Series containing the frequency of each distinct row in the Dataframe.'
#                                                        #
#                                                        # "normalize" divide las entradas por el número total de observaciones. la documentación dice: 'Return proportions rather than frequencies.'
#                                                        #
#                                                        # "sort_index" lo tenemos para reordenar la serie en términos de los índices del dataframe. la documentación dice: 'Returns a new Series sorted by label if inplace argument is False, otherwise updates the original series and returns None.'
#
#     # esto ya es para hacer los histogramas
#     if ax is None:
#         fig, ax = plt.subplots()  # como cuando graficamos
#     else:
#         fig = ax.figure
#
#     # elegimos bordes de bins centrados en enteros: [..., k-0.5, k+0.5, ...] para todos los k observados.
#     # con bin width = 1 y density=True, la altura de cada barra es (conteo/N)/1 = probabilidad.
#     kmin = int(s.min())
#     kmax = int(s.max())
#     edges = np.arange(kmin - 0.5, kmax + 1.5, 1)  # bordes con paso 1 (ancho de bin = 1)
#
#     # histograma normalizado (PDF) pero con bins de ancho 1 donde las alturas corresponden a la densidad de probabilidad
#     ax.hist(s, bins=edges, density=True, edgecolor="black", align="mid", color="darkmagenta", alpha=0.7)
#
#     ax.set_xlabel("número de fotones")  # lo de "fdp.index.values"
#     ax.set_ylabel("probabilidad")       # alturas = probabilidades porque width=1
#     if title:  # por si a esta función sí le damos como argumento un título
#         ax.set_title(title)
#
#     if save_path is not None:
#         save_path = Path(save_path)
#         save_path.parent.mkdir(parents=True, exist_ok=True)  # que crea el directorio si no existe
#         fig.savefig(save_path, dpi=300, bbox_inches="tight")
#
#     return fdp  # regresa series con probabilidades por "k" bins

# pero demos el histograma con barras de error

def fdp_histograma(df, *, ax=None, title=None, save_path=None, error_bars=True):
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(int)

    # probabilities p_k (this is your empirical PMF)
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
    ax.bar(x, y, width=1.0, align="center", edgecolor="black", alpha=0.7)

    # error bars on p_k
    if error_bars:
        yerr = np.sqrt(counts.to_numpy()) / N   # sqrt(n_k)/N
        ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3)

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
