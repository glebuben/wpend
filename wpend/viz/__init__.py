"""Визуализация. Читает только Trajectory / TrajectoryBatch.

Ядро (wpend/) не импортирует ничего отсюда и не зависит от pygame:
    pip install "wpend[viz]"
"""

from .grid import FELL_BACKWARD, FELL_FORWARD, HELD, GridSpec, classify

__all__ = ["GridSpec", "classify", "HELD", "FELL_FORWARD", "FELL_BACKWARD"]
