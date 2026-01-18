from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from io_utils import MeasurementInfo


@dataclass(frozen=True)
class G2Summary:
    session_name: str
    mode: str  # "2D" or "3D"
    n_rows: int

    # From file column g2(0), if present
    g2_file_mean: float | None
    g2_file_std: float | None
    g2_file_sem: float | None

    # From recomputation using counts (+ T/tau only for 2D)
    g2_counts_mean: float
    g2_counts_std: float
    g2_counts_sem: float

    # Meta
    test_time_us: float
    coincidence_window_ns: float


def _col(df: pd.DataFrame, name: str) -> str:
    """
    Case-insensitive column resolver.
    """
    name_low = name.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == name_low:
            return c
    raise KeyError(f"Missing required column '{name}'. Available: {list(df.columns)}")


def _maybe_g2_col(df: pd.DataFrame) -> str | None:
    """
    Finds a column that looks like g2(0) or contains 'g2'.
    Returns None if not found.
    """
    candidates = []
    for c in df.columns:
        s = str(c).strip().lower()
        if "g2" in s:
            candidates.append(c)
    if not candidates:
        return None
    # Prefer exact-looking "g2(0)" if present
    for c in candidates:
        if str(c).strip().lower().replace(" ", "") in {"g2(0)", "g2(0.0)"}:
            return c
    return candidates[0]


def compute_g2_2d(df: pd.DataFrame, info: MeasurementInfo) -> tuple[np.ndarray, np.ndarray | None]:
    """
    2D formula (your notes): g2 = (NTR / (NT*NR)) * (T / tau)
    where T = test time, tau = coincidence window.
    """
    nt = df[_col(df, "NT")].to_numpy(dtype=float)
    nr = df[_col(df, "NR")].to_numpy(dtype=float)
    ntr = df[_col(df, "NTR")].to_numpy(dtype=float)

    denom = nt * nr
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.where(denom > 0, ntr / denom, np.nan)

    # T/tau must be unit-consistent:
    # T is in microseconds, tau is in nanoseconds -> convert tau to microseconds
    tau_us = info.coincidence_window_ns * 1e-3
    factor = info.test_time_us / tau_us if tau_us > 0 else np.nan

    g2_counts = base * factor

    g2c = _maybe_g2_col(df)
    g2_file = df[g2c].to_numpy(dtype=float) if g2c is not None else None
    return g2_counts, g2_file


def compute_g2_3d(df: pd.DataFrame, info: MeasurementInfo) -> tuple[np.ndarray, np.ndarray | None]:
    """
    3D formula: g2 = (NGTR * NG) / (NGT * NGR)
    (No extra T/tau factor per your notes.)
    """
    ng = df[_col(df, "NG")].to_numpy(dtype=float)
    ngt = df[_col(df, "NGT")].to_numpy(dtype=float)
    ngr = df[_col(df, "NGR")].to_numpy(dtype=float)
    ngtr = df[_col(df, "NGTR")].to_numpy(dtype=float)

    denom = ngt * ngr
    with np.errstate(divide="ignore", invalid="ignore"):
        g2_counts = np.where(denom > 0, (ngtr * ng) / denom, np.nan)

    g2c = _maybe_g2_col(df)
    g2_file = df[g2c].to_numpy(dtype=float) if g2c is not None else None
    return g2_counts, g2_file


def summarize_session(session_name: str, mode: str, df: pd.DataFrame, info: MeasurementInfo) -> G2Summary:
    if mode == "2D":
        g2_counts, g2_file = compute_g2_2d(df, info)
    elif mode == "3D":
        g2_counts, g2_file = compute_g2_3d(df, info)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    g2_counts = g2_counts[np.isfinite(g2_counts)]
    if g2_counts.size == 0:
        raise ValueError(f"No finite g2_counts values for session '{session_name}' mode {mode}")

    counts_mean = float(np.mean(g2_counts))
    counts_std = float(np.std(g2_counts, ddof=1)) if g2_counts.size >= 2 else 0.0
    counts_sem = float(counts_std / np.sqrt(g2_counts.size)) if g2_counts.size >= 2 else 0.0

    if g2_file is not None:
        g2_file = g2_file[np.isfinite(g2_file)]
        if g2_file.size >= 1:
            file_mean = float(np.mean(g2_file))
            file_std = float(np.std(g2_file, ddof=1)) if g2_file.size >= 2 else 0.0
            file_sem = float(file_std / np.sqrt(g2_file.size)) if g2_file.size >= 2 else 0.0
        else:
            file_mean = file_std = file_sem = None
    else:
        file_mean = file_std = file_sem = None

    return G2Summary(
        session_name=session_name,
        mode=mode,
        n_rows=int(len(df)),
        g2_file_mean=file_mean,
        g2_file_std=file_std,
        g2_file_sem=file_sem,
        g2_counts_mean=counts_mean,
        g2_counts_std=counts_std,
        g2_counts_sem=counts_sem,
        test_time_us=float(info.test_time_us),
        coincidence_window_ns=float(info.coincidence_window_ns),
    )
