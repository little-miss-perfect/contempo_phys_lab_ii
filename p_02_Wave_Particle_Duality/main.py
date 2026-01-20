from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from config import (
    SAMPLES_DIR,
    OUTPUTS_DIR,
    FILENAME_2D,
    FILENAME_3D,
    INFO_FILENAME,
)
from io_utils import read_counts_csv, read_measurement_info
from g2 import summarize_session, compute_g2_2d, compute_g2_3d
from plotting import plot_g2_summaries, plot_g2_histogram


def _sanitize_filename_piece(s: str) -> str:
    """
    Make a safe filename fragment from a session folder name.
    """
    s = s.strip()
    s = s.replace(" ", "_")
    # Replace characters that can be problematic on Windows filenames
    s = re.sub(r'[<>:"/\\|?*]+', "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def find_sessions(samples_dir: Path) -> list[Path]:
    """
    A session is any subfolder of samples/ that contains either HBT_2D.csv or HBT_3D.csv.
    """
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    sessions: list[Path] = []
    for p in sorted(samples_dir.glob("*")):
        if not p.is_dir():
            continue
        if (p / FILENAME_2D).exists() or (p / FILENAME_3D).exists():
            sessions.append(p)
    return sessions


def build_summary_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = find_sessions(SAMPLES_DIR)

    rows_2d = []
    rows_3d = []

    for session_dir in sessions:
        session_name = session_dir.name
        info = read_measurement_info(session_dir / INFO_FILENAME)

        path_2d = session_dir / FILENAME_2D
        if path_2d.exists():
            df = read_counts_csv(path_2d)
            s = summarize_session(session_name=session_name, mode="2D", df=df, info=info)
            rows_2d.append(s.__dict__)

        path_3d = session_dir / FILENAME_3D
        if path_3d.exists():
            df = read_counts_csv(path_3d)
            s = summarize_session(session_name=session_name, mode="3D", df=df, info=info)
            rows_3d.append(s.__dict__)

    df2 = pd.DataFrame(rows_2d).sort_values("session_name") if rows_2d else pd.DataFrame()
    df3 = pd.DataFrame(rows_3d).sort_values("session_name") if rows_3d else pd.DataFrame()
    return df2, df3


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Summary tables + point plots
    # -----------------------------
    df2, df3 = build_summary_tables()

    if not df2.empty:
        (OUTPUTS_DIR / "summary_2D.csv").parent.mkdir(parents=True, exist_ok=True)
        df2.to_csv(OUTPUTS_DIR / "summary_2D.csv", index=False)

    if not df3.empty:
        (OUTPUTS_DIR / "summary_3D.csv").parent.mkdir(parents=True, exist_ok=True)
        df3.to_csv(OUTPUTS_DIR / "summary_3D.csv", index=False)

    # Point plots: counts-based g2 with SEM error bars
    if not df2.empty:
        plot_g2_summaries(
            df_summary=df2,
            out_path=OUTPUTS_DIR / "plot_g2_2D_counts.png",
            title="g2(0) calculada (2 detectores)",
            y_col="g2_counts_mean",
            yerr_col="g2_counts_sem",
        )

    if not df3.empty:
        plot_g2_summaries(
            df_summary=df3,
            out_path=OUTPUTS_DIR / "plot_g2_3D_counts.png",
            title="g2(0) calculada (3 detectores)",
            y_col="g2_counts_mean",
            yerr_col="g2_counts_sem",
        )

    # Optional: also plot the g2(0) column from file (if present)
    if not df2.empty and "g2_file_mean" in df2.columns and df2["g2_file_mean"].notna().any():
        df2_file = df2[df2["g2_file_mean"].notna()].copy()
        plot_g2_summaries(
            df_summary=df2_file,
            out_path=OUTPUTS_DIR / "plot_g2_2D_file.png",
            title="g2(0) from file column (2 detectors)",
            y_col="g2_file_mean",
            yerr_col="g2_file_sem",
        )

    if not df3.empty and "g2_file_mean" in df3.columns and df3["g2_file_mean"].notna().any():
        df3_file = df3[df3["g2_file_mean"].notna()].copy()
        plot_g2_summaries(
            df_summary=df3_file,
            out_path=OUTPUTS_DIR / "plot_g2_3D_file.png",
            title="g2(0) from file column (3 detectors)",
            y_col="g2_file_mean",
            yerr_col="g2_file_sem",
        )

    # -----------------------------
    # Histograms of g2(0) (COMPUTED ONLY)
    # -----------------------------
    all_g2_2d: list[np.ndarray] = []
    all_g2_3d: list[np.ndarray] = []

    sessions = find_sessions(SAMPLES_DIR)

    for session_dir in sessions:
        session_name = session_dir.name
        safe_name = _sanitize_filename_piece(session_name)
        info = read_measurement_info(session_dir / INFO_FILENAME)

        # 2D
        path_2d = session_dir / FILENAME_2D
        if path_2d.exists():
            df = read_counts_csv(path_2d)
            g2_counts, _g2_file = compute_g2_2d(df, info)

            values = np.asarray(g2_counts, dtype=float)
            values = values[np.isfinite(values)]

            if values.size > 0:
                all_g2_2d.append(values)

                suffix = "" if session_name.strip().lower() == "2d" else f"_{safe_name}"
                plot_g2_histogram(
                    g2_values=values,
                    out_path=OUTPUTS_DIR / f"hist_g2_2D{suffix}.png",
                    # title=f"Histograma de g2(0) calculada — 2 detectores — {session_name}",
                    title=f"Histograma de g2(0) calculada — 2 detectores",
                    bins=30,
                    density=True,
                    show_mean_sem=True,
                    x_label="g2(0)",
                )

        # 3D
        path_3d = session_dir / FILENAME_3D
        if path_3d.exists():
            df = read_counts_csv(path_3d)
            g2_counts, _g2_file = compute_g2_3d(df, info)

            values = np.asarray(g2_counts, dtype=float)
            values = values[np.isfinite(values)]

            if values.size > 0:
                all_g2_3d.append(values)

                suffix = "" if session_name.strip().lower() == "3d" else f"_{safe_name}"
                plot_g2_histogram(
                    g2_values=values,
                    out_path=OUTPUTS_DIR / f"hist_g2_3D{suffix}.png",
                    #title=f"Histogram of computed g2(0) — 3D — {session_name}",
                    title=f"Histograma de g2(0) calculada — 3 detectores",
                    bins=30,
                    density=True,
                    show_mean_sem=True,
                    x_label="g2(0)",
                )

    # Combined histograms (all sessions merged)
    if len(all_g2_2d) > 0:
        combined_2d = np.concatenate(all_g2_2d)
        plot_g2_histogram(
            g2_values=combined_2d,
            out_path=OUTPUTS_DIR / "hist_g2_2D_ALL.png",
            title="Histogram of computed g2(0) — 2D — ALL sessions",
            bins=30,
            density=True,
            show_mean_sem=True,
            x_label="g2(0)",
        )

    if len(all_g2_3d) > 0:
        combined_3d = np.concatenate(all_g2_3d)
        plot_g2_histogram(
            g2_values=combined_3d,
            out_path=OUTPUTS_DIR / "hist_g2_3D_ALL.png",
            title="Histogram of computed g2(0) — 3D — ALL sessions",
            bins=30,
            density=True,
            show_mean_sem=True,
            x_label="g2(0)",
        )


if __name__ == "__main__":
    main()
