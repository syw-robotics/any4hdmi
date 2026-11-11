from __future__ import annotations

from pathlib import Path

import mujoco

DEFAULT_BASE_JOINT_NAME = "floating_base_joint"
DEFAULT_MJCF_REPO_ID = "elijahgalahad/g1_xmls"
DEFAULT_MJCF_PATH = "g1-mode_13_15.xml"
DEFAULT_MJCF_REVISION = "main"
BUNDLED_MJCF_REVISION = "dfa690ab283816a180f66ce892541ad99bf81d7d"
BUNDLED_MJCF_DIR = Path(__file__).resolve().parents[1] / "assets" / "g1"
MjcfInput = str | Path


def build_hf_mjcf_reference(
    *,
    repo_id: str = DEFAULT_MJCF_REPO_ID,
    path: str = DEFAULT_MJCF_PATH,
    revision: str = DEFAULT_MJCF_REVISION,
) -> str:
    return f"hf://{repo_id}@{revision}/{path}"


def resolve_mjcf_path(mjcf: MjcfInput, *, dataset_root: str | Path | None = None) -> Path:
    normalized = normalize_mjcf_reference(mjcf, dataset_root=dataset_root)
    if isinstance(normalized, Path):
        return normalized
    # Existing manifests use these references as stable model identifiers.
    prefixes = (
        f"hf://{DEFAULT_MJCF_REPO_ID}/",
        f"hf://{DEFAULT_MJCF_REPO_ID}@{DEFAULT_MJCF_REVISION}/",
        f"hf://{DEFAULT_MJCF_REPO_ID}@{BUNDLED_MJCF_REVISION}/",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            path = (BUNDLED_MJCF_DIR / normalized[len(prefix):]).resolve()
            if path.parent == BUNDLED_MJCF_DIR and path.suffix == ".xml" and path.is_file():
                return path
    raise FileNotFoundError(
        f"Model assets are local-only. No bundled model matches {normalized!r}; "
        "provide a local XML path with its mesh assets."
    )


def normalize_mjcf_reference(
    mjcf: MjcfInput,
    *,
    dataset_root: str | Path | None = None,
) -> MjcfInput:
    if isinstance(mjcf, Path):
        return mjcf.expanduser().absolute()
    if isinstance(mjcf, str):
        if mjcf.startswith("hf://"):
            return mjcf
        base_dir = Path(dataset_root).expanduser().absolute() if dataset_root else Path.cwd()
        return (base_dir / mjcf).absolute()
    raise TypeError(f"Unsupported mjcf payload type: {type(mjcf)!r}")


def qpos_names_from_model(
    model: mujoco.MjModel, base_joint_name: str = DEFAULT_BASE_JOINT_NAME
) -> list[str]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            if joint_name == base_joint_name:
                names.extend(
                    [
                        "root_tx",
                        "root_ty",
                        "root_tz",
                        "root_qw",
                        "root_qx",
                        "root_qy",
                        "root_qz",
                    ]
                )
            else:
                names.extend(
                    [
                        f"{joint_name}_tx",
                        f"{joint_name}_ty",
                        f"{joint_name}_tz",
                        f"{joint_name}_qw",
                        f"{joint_name}_qx",
                        f"{joint_name}_qy",
                        f"{joint_name}_qz",
                    ]
                )
        elif joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            names.append(str(joint_name))
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            names.extend(
                [f"{joint_name}_qw", f"{joint_name}_qx", f"{joint_name}_qy", f"{joint_name}_qz"]
            )
        else:
            raise ValueError(f"Unsupported joint type {joint_type} for {joint_name}")
    if len(names) != model.nq:
        raise ValueError(f"Expected {model.nq} qpos names, got {len(names)}")
    return names
