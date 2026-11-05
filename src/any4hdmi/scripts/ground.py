from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from any4hdmi.core.format import MOTIONS_SUBDIR, find_dataset_root, load_manifest, load_motion, save_motion
from any4hdmi.core.grounding import DEFAULT_FOOT_BODIES, correct_foot_penetration
from any4hdmi.core.model import load_model
from any4hdmi.dataset.loading import resolve_input_paths


def correct_dataset(
    source: Path,
    output_root: Path,
    *,
    foot_bodies: tuple[str, ...] = DEFAULT_FOOT_BODIES,
    ground_z: float = 0.0,
    clearance: float = 0.001,
    max_correction_speed: float = 0.2,
) -> Path:
    source = source.expanduser().resolve()
    manifest = load_manifest(find_dataset_root(source))
    motions_dir = (manifest.root / manifest.payload.get("motions_subdir", MOTIONS_SUBDIR)).resolve()
    if source.is_file():
        paths = [source] if source.suffix == ".npz" and source.is_relative_to(motions_dir) else []
    elif source == manifest.root or source.is_relative_to(motions_dir):
        paths = sorted((motions_dir if source == manifest.root else source).rglob("*.npz"))
    else:
        paths = []
    if not paths:
        raise ValueError("Input must be a dataset root, a motion .npz, or a nonempty directory under motions_subdir")
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Output dataset directory already exists: {output_root}")
    if output_root.is_relative_to(manifest.root):
        raise ValueError("Output dataset directory must be outside the source dataset")
    model = load_model(manifest.mjcf_path)
    settings = {
        "scope": "whole_robot",
        "foot_bodies": list(foot_bodies), "ground_z": ground_z,
        "clearance": clearance, "max_correction_speed": max_correction_speed,
    }
    reports = {}
    for path in tqdm(paths, desc="Correcting ground penetration", unit="file"):
        corrected, report = correct_foot_penetration(
            model, load_motion(path), timestep=manifest.timestep,
            foot_bodies=foot_bodies, ground_z=ground_z,
            clearance=clearance, max_correction_speed=max_correction_speed,
        )
        # Create the destination only after the first motion has been validated.
        if not reports:
            output_root.mkdir(parents=True, exist_ok=False)
        relative_path = path.relative_to(motions_dir)
        save_motion(output_root / MOTIONS_SUBDIR / relative_path, corrected)
        reports[relative_path.as_posix()] = report

    summary = {
        "num_motions": len(reports), "max_lift_m": max(r["max_lift_m"] for r in reports.values()),
        "body_names": next(iter(reports.values()))["body_names"],
    }
    for phase in ("before", "after"):
        frames = sum(r[phase]["frames"] for r in reports.values())
        penetrating = sum(r[phase]["penetrating_frames"] for r in reports.values())
        summary[phase] = {
            "frames": frames, "penetrating_frames": penetrating,
            "penetrating_frame_ratio": penetrating / frames,
            "max_penetration_m": max(r[phase]["max_penetration_m"] for r in reports.values()),
            "min_body_z_m": np.min([r[phase]["min_body_z_m"] for r in reports.values()], axis=0).tolist(),
            "min_foot_z_m": np.min([r[phase]["min_foot_z_m"] for r in reports.values()], axis=0).tolist(),
        }
    payload = dict(manifest.payload)
    if isinstance(manifest.mjcf, Path):
        payload["mjcf"] = str(manifest.mjcf)
    payload.update(
        dataset_name=f"{manifest.dataset_name}_grounded", motions_subdir=MOTIONS_SUBDIR,
        num_motions=len(reports), total_hours=summary["after"]["frames"] * manifest.timestep / 3600,
        source={
            "dataset": str(manifest.root), "selection": str(source),
            "original_source": payload.get("source", {}), "foot_grounding": settings,
        },
    )
    report_path = output_root / "grounding_report.json"
    report_path.write_text(json.dumps({"settings": settings, "summary": summary, "motions": reports}, indent=2) + "\n")
    # Publish the manifest last so an interrupted output is not a valid dataset.
    (output_root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct whole-robot ground penetration on flat ground by lifting root Z; export a new dataset.")
    parser.add_argument("--motion", required=True, help="Dataset root, motion subdirectory, or .npz; also accepts hf:// dataset references.")
    parser.add_argument("--output-root", required=True, help="New dataset directory outside the source dataset; must not exist.")
    parser.add_argument("--foot-bodies", nargs="+", default=DEFAULT_FOOT_BODIES, help="Foot bodies for per-foot diagnostics (default: G1 ankle-roll links). Correction always checks the whole robot.")
    parser.add_argument("--ground-z", type=float, default=0.0, help="Horizontal floor world Z in metres (default: 0).")
    parser.add_argument("--clearance", type=float, default=.001, help="Nonnegative clearance for every robot surface in metres (default: .001).")
    parser.add_argument("--max-correction-speed", type=float, default=.2, help="Positive limit on added root Z change in m/s (default: .2); uses offline anticipation.")
    args = parser.parse_args()
    source = resolve_input_paths(Path.cwd(), args.motion)[0]
    report_path = correct_dataset(
        source, Path(args.output_root), foot_bodies=tuple(args.foot_bodies),
        ground_z=args.ground_z, clearance=args.clearance,
        max_correction_speed=args.max_correction_speed,
    )
    summary = json.loads(report_path.read_text())["summary"]
    print(f"Penetrating frames: {summary['before']['penetrating_frames']} -> {summary['after']['penetrating_frames']}; max lift: {summary['max_lift_m']:.4f} m")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
