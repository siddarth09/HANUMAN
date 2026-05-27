# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Raster / array utilities shared across map generators."""

from __future__ import annotations

import numpy as np


def normalize_array(data: np.ndarray) -> np.ndarray:
    """Min-max normalize *data* to [0, 1]. NaN values become 0.

    Returns a float64 array of the same shape as *data*.
    """
    data = np.nan_to_num(
        data,
        nan=float(np.nanmin(data)) if not np.all(np.isnan(data)) else 0.0,
    )
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    if vmax > vmin:
        return (data - vmin) / (vmax - vmin)
    return np.zeros_like(data, dtype=np.float64)
