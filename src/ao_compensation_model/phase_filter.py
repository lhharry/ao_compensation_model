"""
Real-time causal filter for GP_Model phase signal.

Removes noise/jitter at zero crossings using phase-unwrap + exponential
moving average (EMA). Designed to run sample-by-sample on a Raspberry Pi.

Usage:
    # Real-time (sample-by-sample):
    filt = PhaseFilter(alpha=0.15)
    for sample in stream:
        filtered = filt.update(sample)

    # Batch (from CSV):
    python phase_filter.py input.csv output.csv --alpha 0.15
"""

import math
import csv
import argparse


class PhaseFilter:
    """Causal low-pass filter for wrapped phase signals [-pi, pi].

    Internally unwraps the phase to avoid discontinuities at +-pi,
    applies an EMA, then wraps the result back to [-pi, pi].

    Computational cost per sample: ~5 floating-point ops.
    """

    def __init__(self, alpha: float = 0.15):
        """
        Args:
            alpha: EMA smoothing factor in (0, 1].
                   Smaller = more smoothing (more lag).
                   Larger  = less smoothing (less lag).
                   0.15 works well for ~100 Hz sampling with ~10-sample noise bursts.
        """
        self.alpha = alpha
        self._unwrapped = None  # cumulative unwrapped phase
        self._filtered = None   # filtered unwrapped phase
        self._prev_raw = None   # previous raw sample (for unwrapping)

    def update(self, raw: float) -> float:
        """Process one new phase sample and return the filtered value.

        Args:
            raw: current GP_Model value in [-pi, pi]

        Returns:
            Filtered phase value in [-pi, pi]
        """
        if self._prev_raw is None:
            # First sample — initialise
            self._prev_raw = raw
            self._unwrapped = raw
            self._filtered = raw
            return raw

        # --- Phase unwrap ---
        diff = raw - self._prev_raw
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi
        self._unwrapped += diff
        self._prev_raw = raw

        # --- EMA on unwrapped phase ---
        self._filtered += self.alpha * (self._unwrapped - self._filtered)

        # --- Wrap back to [-pi, pi] ---
        out = self._filtered % (2 * math.pi)
        if out > math.pi:
            out -= 2 * math.pi
        return out

    def reset(self):
        """Reset filter state (e.g. between trials)."""
        self._unwrapped = None
        self._filtered = None
        self._prev_raw = None


def filter_csv(input_path: str, output_path: str, alpha: float = 0.15):
    """Read a semicolon-delimited CSV, filter GP_Model, write result."""
    filt = PhaseFilter(alpha=alpha)

    with open(input_path, "r", newline="") as fin, \
         open(output_path, "w", newline="") as fout:
        reader = csv.DictReader(fin, delimiter=";")
        fieldnames = reader.fieldnames + ["GP_Model_filtered"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        for row in reader:
            raw = float(row["GP_Model"])
            filtered = filt.update(raw)
            row["GP_Model_filtered"] = f"{filtered:.8f}"
            writer.writerow(row)

    print(f"Filtered {input_path} -> {output_path}  (alpha={alpha})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter GP_Model phase signal to remove zero-crossing noise."
    )
    parser.add_argument("input", help="Input CSV path (semicolon-delimited)")
    parser.add_argument("output", help="Output CSV path")
    parser.add_argument(
        "--alpha", type=float, default=0.15,
        help="EMA smoothing factor (default: 0.15). Smaller = smoother."
    )
    args = parser.parse_args()
    filter_csv(args.input, args.output, args.alpha)
