# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mars terrain generation tool for Gazebo Harmonic + ROS2 Jazzy.

Downloads NASA HiRISE DTMs (projected ESRI:103885, 1 m/px), processes elevation
data, and exports heightmap-based Gazebo SDF terrain models with Bullet
Featherstone physics worlds.
"""

from .utils.types import BoundingBox, ROI, MarsSite
from .utils.site_catalog import list_sites, get_site
from .mars_terrain_exporter import MarsTerrainExporter

__all__ = [
    "BoundingBox",
    "ROI",
    "MarsSite",
    "list_sites",
    "get_site",
    "MarsTerrainExporter",
]
