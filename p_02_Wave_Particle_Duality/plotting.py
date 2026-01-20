from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIG_DPI, PLOT_GRID, ERRORBAR_CAPSIZE


def plot_g2_summaries(
    df_summary: pd.DataFrame,
    out_path: Path,
    title: str,
    y_col: str,
    yerr_col: str,
    y_label: str = "g2(0)",
) -> None:
    """
    Plots mean g2 per session with error bars (SEM by default).

    y_col / yerr_col refer to dataframe column names.
    y_label controls what appears on the plot axis.
    """
    if df_summary.empty:
        raise ValueError("Summary dataframe is empty; nothing to plot.")

    x_labels = df_summary["session_name"].astype(str).tolist()
    y = df_summary[y_col].to_numpy(dtype=float)
    yerr = df_summary[yerr_col].to_numpy(dtype=float)

    plt.figure()
    plt.errorbar(
        range(len(x_labels)),
        y,
        yerr=yerr,
        fmt="o",
        capsize=ERRORBAR_CAPSIZE,
        ecolor="black",   # error bars
        color="black",    # markers/line
    )

    plt.xticks(range(len(x_labels)), x_labels, rotation=30, ha="right")
    plt.ylabel(y_label)
    plt.title(title)

    if PLOT_GRID:
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


def plot_g2_histogram(
    g2_values,
    out_path: Path,
    title: str,
    bins: int = 30,
    density: bool = True,
    show_mean_sem: bool = True,
    x_label: str = "g2(0)",
) -> None:
    """
    Histogram of g2(0) values across rows/runs.

    Error bars per-bin are NOT supported because we don't have per-sample uncertainties.
    But we *can* show mean and mean ± SEM (from the sample spread).
    """
    import numpy as np

    g2 = np.asarray(g2_values, dtype=float)
    g2 = g2[np.isfinite(g2)]

    if g2.size == 0:
        raise ValueError("No finite g2 values to histogram.")

    mean = float(np.mean(g2))
    std = float(np.std(g2, ddof=1)) if g2.size >= 2 else 0.0
    sem = float(std / np.sqrt(g2.size)) if g2.size >= 2 else 0.0

    plt.figure()
    plt.hist(g2, bins=bins, density=density)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Densidad de probabilidad" if density else "Cuentas")

    if PLOT_GRID:
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

    if show_mean_sem:
        # Mean line (black)
        plt.axvline(mean, linestyle="--", linewidth=1, color="black")

        # Mean ± SEM band (subtle gray)
        if sem > 0:
            plt.axvspan(mean - sem, mean + sem, alpha=0.2, color="gray")

        # Stats box
        txt = (
            f"n={g2.size}\n"
            f"media muestral={mean:.6g}\n"
            f"desviación estándar muestral={std:.6g}\n"
            f"SEM={sem:.6g}"
        )
        ax = plt.gca()
        ax.text(
            0.98,
            0.98,  # <-- inside the axes (top-right corner)
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,  # <-- smaller text so it fits inside
            bbox=dict(boxstyle="round,pad=0.20", alpha=0.18),
            clip_on=True,  # <-- keep it inside the axes
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()
