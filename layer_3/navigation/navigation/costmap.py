"""Geometric global cost map from the HiRISE DEM (slope + roughness)."""
from __future__ import annotations
import numpy as np


def _slope_deg(Z: np.ndarray, res: float) -> np.ndarray:
    """Per-cell terrain slope (degrees) from the elevation gradient."""
    dzdy, dzdx = np.gradient(Z, res)
    grad = np.sqrt(dzdx**2 + dzdy**2)
    return np.degrees(np.arctan(grad))


def _roughness(Z: np.ndarray, radius: int) -> np.ndarray:
    """Local elevation std over a (2r+1) window — proxy for rocks/steps."""
    import warnings
    from scipy.ndimage import generic_filter
    size = 2 * radius + 1
    # all-NaN windows -> NaN (caller's lethal mask handles them)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return generic_filter(Z, np.nanstd, size=size, mode="nearest")


def build_costmap(Z: np.ndarray, res: float,
                  slope_max_deg: float = 25.0,
                  rough_radius: int = 2,
                  rough_max: float = 0.15,
                  w_slope: float = 0.6,
                  w_rough: float = 0.4):
    """DEM elevation grid -> (cost[0..1], lethal mask).

    cost = w_slope * (slope/slope_max) + w_rough * (rough/rough_max), clipped to 1.
    Cells at/over slope_max, or with no DEM data, are lethal.
    """
    slope = _slope_deg(Z, res)
    rough = _roughness(Z, rough_radius)

    c_slope = np.clip(slope / slope_max_deg, 0.0, 1.0)
    c_rough = np.clip(rough / rough_max, 0.0, 1.0)
    cost = w_slope * c_slope + w_rough * c_rough

    lethal = (~np.isfinite(Z)) | (slope >= slope_max_deg)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost[lethal] = 1.0
    return cost.astype(np.float32), lethal


def to_occupancy(cost: np.ndarray, lethal: np.ndarray) -> np.ndarray:
    """Map cost[0..1] -> OccupancyGrid int8 0..100; lethal -> 100, no-data -> -1."""
    occ = np.clip(cost * 100.0, 0, 99).astype(np.int8)
    occ[lethal] = 100
    return occ
