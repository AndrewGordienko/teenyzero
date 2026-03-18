from __future__ import annotations

import argparse
import os
import sys


def _normalized_profile_name(profile_name: str) -> str:
    return (profile_name or "").strip().lower()


def _normalized_device_name(device_name: str) -> str:
    value = (device_name or "").strip().lower()
    if value in {"h100", "h200"}:
        return "cuda"
    if value in {"gpu", ""}:
        return "auto"
    return value


def bootstrap_runtime_cli(argv=None):
    source_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--board-backend", default=None)
    parser.add_argument("--runtime-root", default=None)
    parser.add_argument("--tmpdir", default=None)
    args, remaining = parser.parse_known_args(source_argv)

    profile_name = _normalized_profile_name(args.profile)
    device_name = _normalized_device_name(args.device)
    board_backend = (args.board_backend or "").strip().lower()

    if profile_name and profile_name != "auto":
        os.environ["TEENYZERO_PROFILE"] = profile_name
    elif profile_name == "auto":
        os.environ.pop("TEENYZERO_PROFILE", None)

    if device_name and device_name != "auto":
        os.environ["TEENYZERO_DEVICE"] = device_name
    elif device_name == "auto":
        os.environ.pop("TEENYZERO_DEVICE", None)

    if args.device and args.device.strip().lower() in {"h100", "h200"} and not profile_name:
        os.environ["TEENYZERO_PROFILE"] = args.device.strip().lower()

    if board_backend and board_backend != "auto":
        os.environ["TEENYZERO_BOARD_BACKEND"] = board_backend
    elif board_backend == "auto":
        os.environ.pop("TEENYZERO_BOARD_BACKEND", None)

    runtime_root = (args.runtime_root or "").strip()
    if runtime_root:
        os.environ["TEENYZERO_RUNTIME_ROOT"] = runtime_root

    tmpdir = (args.tmpdir or "").strip()
    if tmpdir:
        os.environ["TMPDIR"] = tmpdir

    if argv is None:
        sys.argv[:] = [sys.argv[0], *remaining]

    return args
