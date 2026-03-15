'''
Apply RealTimeBandpassFilter to Hip_x for every CSV file in a folder tree
and write the filtered values back as a new column 'filter_hip_x'.
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from definitions import BANDPASS_HIGHCUT, BANDPASS_LOWCUT, BANDPASS_ORDER, SAMPLING_FREQ

# --- Folder containing the CSV files (searched recursively) ---
INPUT_FOLDER = Path(__file__).resolve().parent / "dataset" / "test"


class RealTimeBandpassFilter:
    """Causal bandpass filter for real-time, sample-by-sample processing."""

    def __init__(self, lowcut: float, highcut: float, fs: int, order: int = 3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        coeffs = butter(order, [low, high], btype="band")
        assert coeffs is not None
        self.b: np.ndarray = np.asarray(coeffs[0])
        self.a: np.ndarray = np.asarray(coeffs[1])
        self.zi = lfilter_zi(self.b, self.a)
        self.is_initialized = False

    def process_point(self, new_value: float) -> float:
        """Filter a single new data point and return the filtered value."""
        if not self.is_initialized:
            self.zi = self.zi * new_value
            self.is_initialized = True
        filtered_array, self.zi = lfilter(self.b, self.a, [new_value], zi=self.zi)
        return float(filtered_array[0])


csvs = sorted(INPUT_FOLDER.rglob("*.csv"))
if not csvs:
    print(f"No CSV files found in: {INPUT_FOLDER}")
else:
    for csv_path in csvs:
        df = pd.read_csv(csv_path, sep=";")
        if "Hip_x" not in df.columns:
            print(f"Skipping {csv_path}: no 'Hip_x' column")
            continue

        raw_hip_angle = df["Hip_x"].to_numpy(dtype=float)
        rt_filter = RealTimeBandpassFilter(
            BANDPASS_LOWCUT, BANDPASS_HIGHCUT, SAMPLING_FREQ, 1
        )
        df["filter_hip_x"] = np.array(
            [rt_filter.process_point(x) for x in raw_hip_angle]
        )
        df.to_csv(csv_path, sep=";", index=False)
        print(f"Saved: {csv_path}")

