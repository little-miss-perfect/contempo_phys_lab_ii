from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from config import FIG_DPI, PLOT_GRID, ERRORBAR_CAPSIZE




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
    y_label: str = "g2",
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
    plt.errorbar(range(len(x_labels)), y, yerr=yerr, fmt="o", capsize=ERRORBAR_CAPSIZE)

    plt.xticks(range(len(x_labels)), x_labels, rotation=30, ha="right")
    plt.ylabel(y_label)
    plt.title(title)

    if PLOT_GRID:
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()
