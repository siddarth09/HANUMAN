"""HANUMAN layer_1.

Two-system locomotion stack:
  - system1/  RL high-level policy ("System 1") — registers the mjlab tasks.
  - system0/  MPC executor ("System 0", copied from BHEEMA) — used in the ROS hybrid.

The mjlab task registry symlink (~/mjlab/src/mjlab/tasks/hanuman -> layer_1) imports
this package, so importing system1 here keeps `Hanuman-Mars-v0` registered.
"""

from . import system1  # noqa: F401  (registers Hanuman-Mars-v0 on import)
