from __future__ import annotations

import json
from pathlib import Path

from any4hdmi.core.format import find_dataset_root, load_manifest, load_motion, save_motion


def export_motion_clip(source: Path, output_root: Path, start: int, end: int) -> Path:
    """Export inclusive, zero-based source frames as a new standalone dataset."""
    qpos = load_motion(source)
    if not 0 <= start <= end < len(qpos):
        raise ValueError(f"Expected 0 <= start <= end < {len(qpos)}, got {start}..{end}")
    manifest = load_manifest(find_dataset_root(source))
    payload = dict(manifest.payload)
    mjcf = manifest.mjcf
    if isinstance(mjcf, Path):
        payload["mjcf"] = str(mjcf.absolute())
    payload.update(
        dataset_name=f"{payload['dataset_name']}_edited",
        motions_subdir="motions",
        num_motions=1,
        total_hours=(end - start + 1) * manifest.timestep / 3600.0,
        source={
            "dataset": str(manifest.root),
            "motion": source.absolute().relative_to(find_dataset_root(source)).as_posix(),
            "start_frame": start,
            "end_frame_inclusive": end,
            "original_source": payload.get("source", {}),
        },
    )
    output_root = output_root.expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    output_path = save_motion(output_root / "motions" / source.name, qpos[start : end + 1])
    (output_root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
