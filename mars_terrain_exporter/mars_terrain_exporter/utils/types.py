# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Data types for Mars terrain generation configuration.

Key difference from the lunar counterpart: MarsSite stores its DEM URL
directly in the catalog (HiRISE DTMs don't share a common URL pattern),
and BoundingBox has no south-pole latitude restriction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


@dataclass
class BoundingBox:
    """Geographic bounding box defined by center (Mars lat/lon) and dimensions.

    lat/lon use planetocentric coordinates (IAU / Mars 2000 convention),
    i.e. the same coordinate system as the HiRISE product headers.
    """

    lat: float        # degrees, −90 … +90
    lon: float        # degrees, 0 … 360 east (or −180 … +180)
    width_km: float = 10.0
    height_km: float = 10.0

    def validate(self) -> None:
        if not (-90.0 <= self.lat <= 90.0):
            raise ValueError(f"lat must be in −90 … +90 (got: {self.lat})")
        if self.width_km <= 0:
            raise ValueError(f"width_km must be > 0 (got: {self.width_km})")
        if self.height_km <= 0:
            raise ValueError(f"height_km must be > 0 (got: {self.height_km})")


@dataclass
class ROI:
    """Defines how much of a DEM to use: full tile or a bounding-box crop."""

    use_full: bool = True
    bounding_box: BoundingBox | None = None

    def validate(self) -> None:
        if not self.use_full:
            if self.bounding_box is None:
                raise ValueError("bounding_box is required when use_full is False")
            self.bounding_box.validate()


@dataclass
class MarsSite:
    """A single Mars terrain site with its HiRISE DTM source and ROI.

    Unlike the lunar counterpart (which constructs DEM URLs from a site_code
    pattern), each Mars site stores its full URL explicitly because HiRISE
    product URLs are unique per scene.
    """

    site_code: str
    name: str
    dem_url: str
    description: str = ""
    roi: ROI = field(default_factory=ROI)

    @classmethod
    def from_catalog(cls, identifier: str, roi: ROI | None = None) -> "MarsSite":
        """Look up a site by name or code and return a MarsSite instance."""
        from .site_catalog import get_site

        entry = get_site(identifier)
        site = cls(
            site_code=entry["site_code"],
            name=entry["site_name"],
            dem_url=entry["dem_url"],
            description=entry["description"],
            roi=roi if roi is not None else ROI(use_full=True),
        )
        site.validate()
        return site

    def validate(self) -> None:
        if not self.name or not _VALID_NAME_RE.match(self.name):
            raise ValueError(
                f"name must be non-empty and contain only alphanumeric, "
                f"hyphens, or underscores (got: {self.name!r})"
            )
        if not self.site_code:
            raise ValueError("site_code must be a non-empty string")
        if not self.dem_url:
            raise ValueError("dem_url must be a non-empty string")
        self.roi.validate()
