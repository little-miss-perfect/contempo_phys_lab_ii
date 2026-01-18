from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd

from config import DEFAULT_TEST_TIME_US, DEFAULT_COINC_WINDOW_NS


@dataclass(frozen=True)
class MeasurementInfo:
    test_time_us: float
    coincidence_window_ns: float


_TIME_RE = re.compile(r"Tiempo\s*de\s*Prueba\s*:\s*([0-9]*\.?[0-9]+)\s*us", re.IGNORECASE)
_WIN_RE = re.compile(r"Ventana\s*de\s*Coincidencia\s*:\s*([0-9]*\.?[0-9]+)\s*ns", re.IGNORECASE)


def read_measurement_info(info_path: Path) -> MeasurementInfo:
    """
    Parses infoMedicion.txt like:
      Tiempo de Prueba        : 1000000.0 us
      Ventana de Coincidencia : 5 ns
    """
    if not info_path.exists():
        return MeasurementInfo(DEFAULT_TEST_TIME_US, DEFAULT_COINC_WINDOW_NS)

    text = info_path.read_text(encoding="utf-8", errors="ignore")

    t_match = _TIME_RE.search(text)
    w_match = _WIN_RE.search(text)

    test_time_us = float(t_match.group(1)) if t_match else DEFAULT_TEST_TIME_US
    coincidence_window_ns = float(w_match.group(1)) if w_match else DEFAULT_COINC_WINDOW_NS

    return MeasurementInfo(test_time_us=test_time_us, coincidence_window_ns=coincidence_window_ns)


def read_counts_csv(csv_path: Path) -> pd.DataFrame:
    """
    Reads CSV and keeps column names exactly as they appear.
    We will normalize column *lookup* in g2.py (case-insensitive).
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df
