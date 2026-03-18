from __future__ import annotations

try:
    from . import _speedups as speedups
except Exception:
    speedups = None

try:
    from .board import NativeBoard, NativeMove, make_board
except Exception:
    NativeBoard = None
    NativeMove = None
    make_board = None
