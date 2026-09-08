"""
analytics.py — Traffic density and congestion estimation.

IMPORTANT — Limitations
-----------------------
These are ESTIMATES based on computer-vision measurements:
  * vehicle count in the current frame
  * a configurable maximum expected count used for normalisation

They are NOT scientifically calibrated measurements.  Accuracy depends on:
  * camera position and angle
  * video resolution and quality
  * YOLO detection accuracy
  * the max_vehicles calibration value

Speed is NOT used because reliable speed estimation requires camera
calibration (focal length, mount height, tilt angle) that is unavailable
from a generic uploaded video.
"""

from config import (
    DENSITY_MAX_VEHICLES,
    DENSITY_LOW_MAX, DENSITY_MEDIUM_MAX,
    CONGESTION_LOW_MAX, CONGESTION_MODERATE_MAX, CONGESTION_HIGH_MAX,
    CONGESTION_DENSITY_WEIGHT, CONGESTION_COUNT_WEIGHT,
)


class TrafficAnalytics:
    """Compute per-frame density and congestion estimates."""

    # ------------------------------------------------------------------
    def compute_density(
        self,
        vehicle_count: int,
        max_vehicles: int = DENSITY_MAX_VEHICLES,
    ) -> dict:
        """Return a density score (0-100) and categorical level.

        Parameters
        ----------
        vehicle_count : number of vehicles detected in the current frame
        max_vehicles  : calibration value — vehicle count that equals 100 % density
        """
        max_vehicles = max(max_vehicles, 1)
        score = min(100.0, (vehicle_count / max_vehicles) * 100.0)
        level = self._density_level(score)
        return {
            "density_score":  round(score, 1),
            "density_level":  level,
            "vehicle_count":  vehicle_count,
        }

    # ------------------------------------------------------------------
    def compute_congestion(
        self,
        density_score: float,
        vehicle_count: int,
        max_vehicles: int = DENSITY_MAX_VEHICLES,
    ) -> dict:
        """Return a congestion score (0-100) and categorical level.

        Formula (configurable weights):
            congestion = density_weight * density_score
                       + count_weight  * normalised_count * 100

        Neither value is speed — see module docstring for explanation.
        """
        max_vehicles       = max(max_vehicles, 1)
        normalised_count   = min(1.0, vehicle_count / max_vehicles)
        score = (
            CONGESTION_DENSITY_WEIGHT * density_score
            + CONGESTION_COUNT_WEIGHT  * normalised_count * 100.0
        )
        score = min(100.0, max(0.0, score))
        level = self._congestion_level(score)
        return {
            "congestion_score": round(score, 1),
            "congestion_level": level,
        }

    # ------------------------------------------------------------------
    def analyse_frame(
        self,
        vehicle_count: int,
        max_vehicles: int = DENSITY_MAX_VEHICLES,
    ) -> dict:
        """Convenience method: compute both density and congestion at once."""
        density    = self.compute_density(vehicle_count, max_vehicles)
        congestion = self.compute_congestion(
            density["density_score"], vehicle_count, max_vehicles
        )
        return {**density, **congestion}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _density_level(score: float) -> str:
        if score <= DENSITY_LOW_MAX:
            return "LOW"
        if score <= DENSITY_MEDIUM_MAX:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def _congestion_level(score: float) -> str:
        if score <= CONGESTION_LOW_MAX:
            return "LOW"
        if score <= CONGESTION_MODERATE_MAX:
            return "MODERATE"
        if score <= CONGESTION_HIGH_MAX:
            return "HIGH"
        return "SEVERE"
