from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from any4hdmi.core.format import repo_root
from any4hdmi.core.model import load_model
from any4hdmi.utils.mjcf import (
    DEFAULT_MJCF_PATH,
    DEFAULT_MJCF_REPO_ID,
    DEFAULT_MJCF_REVISION,
    MjcfInput,
    resolve_mjcf_path,
)


def add_csv_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_csv_dir: Path,
    default_out_dir: str,
    default_fps: float,
) -> None:
    parser.add_argument(
        "--csv-dir",
        default=str(default_csv_dir),
        help="Directory containing source CSV files.",
    )
    parser.add_argument(
        "--out-dir", default=default_out_dir, help="Output dataset root for converted motions."
    )
    parser.add_argument("--fps", type=float, default=default_fps, help="Source frame rate.")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern relative to --csv-dir.")


def add_hf_mjcf_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mjcf-repo",
        default=DEFAULT_MJCF_REPO_ID,
        help="Hugging Face repo id that stores the MJCF and mesh assets.",
    )
    parser.add_argument(
        "--mjcf-path",
        default=DEFAULT_MJCF_PATH,
        help="Path to the MJCF file within --mjcf-repo.",
    )
    parser.add_argument(
        "--mjcf-revision",
        default=DEFAULT_MJCF_REVISION,
        help="Revision passed to Hugging Face snapshot_download.",
    )


def resolve_output_dir(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    return out_dir.resolve()


def load_conversion_model(mjcf: MjcfInput) -> mujoco.MjModel:
    print(f"Loading robot MJCF: {mjcf}", flush=True)
    if isinstance(mjcf, str) and mjcf.startswith("hf://"):
        print(
            "Fetching the Hugging Face repository containing robot models and mesh assets "
            "(reuses local cache; downloads missing files).",
            flush=True,
        )
    return load_model(resolve_mjcf_path(mjcf))


def slice_frames(motion: np.ndarray, start: int, end: int, stride: int) -> np.ndarray:
    resolved_end = end if end >= 0 else motion.shape[0]
    return motion[start : min(resolved_end, motion.shape[0]) : max(1, stride)]
