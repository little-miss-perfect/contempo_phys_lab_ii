from pathlib import Path

# Project root = folder where this config.py lives
PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLES_DIR = PROJECT_ROOT / "samples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Input filenames expected inside each session folder under samples/
FILENAME_2D = "HBT_2D.csv"
FILENAME_3D = "HBT_3D.csv"
INFO_FILENAME = "infoMedicion.txt"

# Plot settings
FIG_DPI = 200
PLOT_GRID = True
ERRORBAR_CAPSIZE = 4

# Normalization toggle (kept for later expansion when you have g2(tau) vs tau)
# For g2(0) only, this will effectively do nothing.
ENABLE_NORMALIZATION = False
NORMALIZATION_TARGET = 1.0
NORMALIZATION_TAIL_K = 3  # used only if we have multiple tau points

# When infoMedicion.txt is missing or unparsable, fall back to these:
DEFAULT_TEST_TIME_US = 1_000_000.0  # microseconds
DEFAULT_COINC_WINDOW_NS = 5.0       # nanoseconds
