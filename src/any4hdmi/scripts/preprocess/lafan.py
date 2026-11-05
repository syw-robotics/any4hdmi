from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from any4hdmi.core.format import MOTION_DTYPE, MOTIONS_SUBDIR, repo_root, save_motion, write_manifest
from any4hdmi.core.model import G1_JOINT_ORDER, base_qpos_adr, joint_qpos_adrs
from any4hdmi.scripts.preprocess.common import (
    add_csv_arguments,
    add_hf_mjcf_arguments,
    load_conversion_model,
    resolve_output_dir,
    slice_frames,
)
from any4hdmi.utils.mjcf import (
    build_hf_mjcf_reference,
    qpos_names_from_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LAFAN retargeted G1 CSV files into the any4hdmi qpos format."
    )
    add_csv_arguments(
        parser,
        default_csv_dir=repo_root().parent / "lafan-process" / "LAFAN1_Retargeting_Dataset" / "g1",
        default_out_dir="output/lafan",
        default_fps=30.0,
    )
    add_hf_mjcf_arguments(parser)
    return parser.parse_args()


def _load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    return np.asarray(data, dtype=MOTION_DTYPE)


def _build_qpos_sequence(model, motion: np.ndarray) -> np.ndarray:
    expected_width = 7 + len(G1_JOINT_ORDER)
    if motion.shape[1] != expected_width:
        raise ValueError(
            f"Expected {expected_width} columns (7 root + {len(G1_JOINT_ORDER)} joints), "
            f"got {motion.shape[1]}."
        )

    qpos = np.zeros((motion.shape[0], model.nq), dtype=MOTION_DTYPE)
    base_adr = base_qpos_adr(model)
    hinge_qpos_adrs = joint_qpos_adrs(model, G1_JOINT_ORDER)

    qpos[:, base_adr : base_adr + 3] = motion[:, 0:3]
    qpos[:, base_adr + 3 : base_adr + 7] = motion[:, [6, 3, 4, 5]]
    qpos[:, hinge_qpos_adrs] = motion[:, 7:]
    return qpos


def main() -> None:
    args = _parse_args()

    csv_dir = Path(args.csv_dir).expanduser().resolve()
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    out_dir = resolve_output_dir(args.out_dir)

    mjcf_reference = build_hf_mjcf_reference(
        repo_id=args.mjcf_repo,
        path=args.mjcf_path,
        revision=args.mjcf_revision,
    )
    model = load_conversion_model(mjcf_reference)
    qpos_names = qpos_names_from_model(model)

    csv_files = sorted(csv_dir.rglob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched {args.pattern!r} under {csv_dir}")

    num_motions = 0
    total_frames = 0
    for csv_path in tqdm(csv_files, desc="Converting LAFAN", unit="file"):
        motion = _load_csv(csv_path)
        motion = slice_frames(motion, args.start, args.end, args.stride)
        qpos = _build_qpos_sequence(model, motion)

        rel_path = csv_path.relative_to(csv_dir).with_suffix(".npz")
        save_motion(out_dir / MOTIONS_SUBDIR / rel_path, qpos)
        num_motions += 1
        total_frames += int(qpos.shape[0])

    write_manifest(
        out_dir,
        dataset_name="lafan",
        mjcf=mjcf_reference,
        timestep=1.0 / args.fps,
        qpos_names=qpos_names,
        num_motions=num_motions,
        total_hours=total_frames / args.fps / 3600.0,
        source={
            "csv_dir": str(csv_dir),
            "pattern": args.pattern,
            "fps": args.fps,
            "root_representation": "xyz + qx qy qz qw",
            "joint_unit": "rad",
        },
    )


if __name__ == "__main__":
    main()
