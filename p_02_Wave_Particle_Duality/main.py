from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import (
    SAMPLES_DIR,
    OUTPUTS_DIR,
    FILENAME_2D,
    FILENAME_3D,
    INFO_FILENAME,
)
from io_utils import read_counts_csv, read_measurement_info
from g2 import summarize_session
from plotting import plot_g2_summaries


def find_sessions(samples_dir: Path) -> list[Path]:
    """
    A session is any subfolder of samples/ that contains either HBT_2D.csv or HBT_3D.csv.
    """
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

    sessions = []
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

    df2, df3 = build_summary_tables()

    # Save CSVs
    if not df2.empty:
        out2 = OUTPUTS_DIR / "summary_2D.csv"
        df2.to_csv(out2, index=False)

    if not df3.empty:
        out3 = OUTPUTS_DIR / "summary_3D.csv"
        df3.to_csv(out3, index=False)

    # Plots: counts-based g2 with SEM error bars
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
    if not df2.empty and df2["g2_file_mean"].notna().any():
        df2_file = df2[df2["g2_file_mean"].notna()].copy()
        plot_g2_summaries(
            df_summary=df2_file,
            out_path=OUTPUTS_DIR / "plot_g2_2D_file.png",
            title="g2(0) del archivo (2 detectores)",
            y_col="g2_file_mean",
            yerr_col="g2_file_sem",
        )

    if not df3.empty and df3["g2_file_mean"].notna().any():
        df3_file = df3[df3["g2_file_mean"].notna()].copy()
        plot_g2_summaries(
            df_summary=df3_file,
            out_path=OUTPUTS_DIR / "plot_g2_3D_file.png",
            title="g2(0) del archivo (3 detectores)",
            y_col="g2_file_mean",
            yerr_col="g2_file_sem",
        )


if __name__ == "__main__":
    main()
