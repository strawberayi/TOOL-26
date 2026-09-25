from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from clahe_calibration import CLAHECalibrationConfig, select_beta  # noqa: E402


def summary(betas, gains, acceptance):
    return pd.DataFrame({
        "bracket": "Test",
        "beta": betas,
        "mean_contrast_gain": gains,
        "mean_noise": [0.01 * b for b in betas],
        "acceptance_rate": acceptance,
    })


class BetaSelectionTests(unittest.TestCase):
    BETAS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

    def test_max_gain_picks_largest_admissible_beta(self):
        # Monotonic gain: the legacy rule always lands on the noise ceiling.
        frame = summary(self.BETAS, [1, 2, 3, 4, 5, 6, 7, 8], [1] * 7 + [0.5])
        config = CLAHECalibrationConfig(selection_rule="max_gain")
        self.assertEqual(select_beta(frame, config), 5.0)

    def test_knee_finds_diminishing_returns(self):
        # Gain saturates after beta 3.0.
        gains = [0.0, 6.0, 9.0, 9.5, 9.8, 9.9, 10.0, 10.0]
        frame = summary(self.BETAS, gains, [1] * 8)
        config = CLAHECalibrationConfig(selection_rule="knee")
        self.assertEqual(select_beta(frame, config), 3.0)

    def test_knee_differs_between_brackets_with_same_ceiling(self):
        config = CLAHECalibrationConfig(selection_rule="knee")
        early = summary(self.BETAS, [0, 8, 9, 9.5, 9.7, 9.8, 9.9, 10], [1] * 8)
        late = summary(self.BETAS, [0, 0.5, 1, 2, 5, 8, 9.5, 10], [1] * 8)
        self.assertLess(select_beta(early, config), select_beta(late, config))

    def test_flat_curve_uses_gentlest_beta(self):
        frame = summary(self.BETAS, [3.0] * 8, [1] * 8)
        self.assertEqual(select_beta(frame, CLAHECalibrationConfig()), 2.0)

    def test_no_admissible_beta_raises(self):
        frame = summary(self.BETAS, list(range(8)), [0.1] * 8)
        with self.assertRaises(RuntimeError):
            select_beta(frame, CLAHECalibrationConfig())


if __name__ == "__main__":
    unittest.main()
