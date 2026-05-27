# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Allows the package to be executed directly from the ROS package root:
#
#   python3 mars_terrain_exporter site jezero_c --output-dir ./models
#   python3 -m mars_terrain_exporter site jezero_c --output-dir ./models
#
# When invoked as `python3 mars_terrain_exporter`, Python runs this file as
# __main__ and the grandparent directory (the ROS package root) may not yet
# be on sys.path.  We add it explicitly so absolute imports resolve correctly.

import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent   # …/mars_terrain_exporter/ (ROS pkg)
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from mars_terrain_exporter.cli import main

if __name__ == "__main__":
    main()
