from __future__ import annotations

import os


_NATIVE_SPEEDUPS = None
_NATIVE_IMPORT_ERROR = None
_NATIVE_BOARD = None


def requested_board_backend():
    return os.environ.get("TEENYZERO_BOARD_BACKEND", "auto").strip().lower() or "auto"


def native_speedups_module():
    global _NATIVE_SPEEDUPS, _NATIVE_IMPORT_ERROR
    if _NATIVE_SPEEDUPS is not None:
        return _NATIVE_SPEEDUPS
    if _NATIVE_IMPORT_ERROR is not None:
        return None

    try:
        from teenyzero.native import speedups
    except Exception as exc:
        _NATIVE_IMPORT_ERROR = exc
        return None

    _NATIVE_SPEEDUPS = speedups
    return _NATIVE_SPEEDUPS


def native_board_factory():
    global _NATIVE_BOARD
    if _NATIVE_BOARD is not None:
        return _NATIVE_BOARD

    try:
        from teenyzero.native import make_board, speedups
    except Exception:
        return None
    if make_board is None or speedups is None or not hasattr(speedups, "board_new"):
        return None

    _NATIVE_BOARD = make_board
    return _NATIVE_BOARD


def native_speedups_available():
    return native_speedups_module() is not None


def native_board_available():
    return native_board_factory() is not None


def resolve_board_backend_name():
    requested = requested_board_backend()
    if requested == "python":
        return "python"
    if requested == "native" and native_speedups_available() and native_board_available():
        return "native"
    if requested == "native":
        return "python"
    return "native" if native_speedups_available() and native_board_available() else "python"


def create_board(fen=None):
    if resolve_board_backend_name() == "native":
        factory = native_board_factory()
        if factory is not None:
            return factory(fen=fen)
    import chess

    return chess.Board(fen) if fen is not None else chess.Board()


def move_from_uci(uci):
    if resolve_board_backend_name() == "native":
        try:
            from teenyzero.native import NativeMove
        except Exception:
            NativeMove = None
        if NativeMove is not None:
            return NativeMove.from_uci(uci)
    import chess

    return chess.Move.from_uci(uci)


def board_backend_payload():
    return {
        "requested_board_backend": requested_board_backend(),
        "resolved_board_backend": resolve_board_backend_name(),
        "native_speedups_available": native_speedups_available(),
        "native_board_available": native_board_available(),
    }
