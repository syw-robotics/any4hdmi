from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
from mjhub import temp_mjcf_with_floor
from tqdm import tqdm

from any4hdmi.core.format import MOTIONS_SUBDIR, find_dataset_root, load_manifest, load_motion
from any4hdmi.dataset.loading import resolve_input_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay qpos-only any4hdmi motions with MuJoCo.")
    parser.add_argument(
        "--motion",
        required=True,
        help=(
            "A motion .npz, a dataset root that contains manifest.json, or "
            "hf://<namespace>/<repo>[@revision]/<path>. Interactive playback can switch "
            "among every motion in that dataset."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS override. Defaults to 1 / manifest.timestep.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a viewer window.")
    parser.add_argument("--port", type=int, default=8080, help="mjviser server port.")
    return parser.parse_args()


def _iter_frame_indices(length: int, start: int, end: int, stride: int) -> range:
    resolved_end = end if end >= 0 else length
    return range(start, min(length, resolved_end), max(1, stride))


def _apply_qpos_frame(data: mujoco.MjData, qpos_frame) -> None:
    data.qpos[:] = qpos_frame
    data.qvel[:] = 0.0


def _resolve_motion_path(motion: str) -> Path:
    return resolve_input_paths(Path.cwd(), motion)[0]


def dataset_motions(dataset_root: Path, manifest) -> list[tuple[str, Path]]:
    motions_dir = dataset_root / manifest.payload.get("motions_subdir", MOTIONS_SUBDIR)
    motions = sorted(path for path in motions_dir.rglob("*.npz") if path.is_file())
    if not motions:
        raise FileNotFoundError(f"No motion .npz files found under {motions_dir}")
    return [(path.relative_to(motions_dir).as_posix(), path) for path in motions]


def initial_motion_label(motion_path: Path, motions: list[tuple[str, Path]]) -> str:
    if motion_path.is_dir():
        return motions[0][0]
    wanted = {motion_path.absolute(), motion_path.resolve()}
    for label, path in motions:
        if path.absolute() in wanted or path.resolve() in wanted:
            return label
    raise FileNotFoundError(f"Motion file is not part of the dataset: {motion_path}")


def _run_mjviser(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motions: list[tuple[str, Path]],
    initial_label: str,
    *,
    start: int,
    end: int,
    stride: int,
    frame_dt: float,
    loop: bool,
    port: int,
) -> None:
    try:
        import viser
        from mjviser import ViserMujocoScene
    except ImportError as exc:
        raise ImportError(
            "any4hdmi-view now uses mjviser for interactive playback. "
            "Install the project dependencies with `uv sync` or run through `uv run`."
        ) from exc

    motion_by_label = dict(motions)
    labels = list(motion_by_label)
    server = viser.ViserServer(port=port, label="any4hdmi-view")
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = True
    scene.create_scene_gui()

    state = {
        "label": None,
        "qpos": None,
        "frame_indices": [],
        "playing": True,
        "cursor": 0,
        "pending_seek": None,
        "ignore_slider_value": None,
        "syncing_slider": False,
        "next_time": time.monotonic(),
        "speed": 1.0,
    }

    with server.gui.add_folder("Motions", order=0):
        motion_dropdown = server.gui.add_dropdown("Motion", labels, initial_value=initial_label, order=0)
        prev_button = server.gui.add_button("Previous", order=1)
        next_button = server.gui.add_button("Next", order=2)

    with server.gui.add_folder("Playback", order=1):
        play_pause_button = server.gui.add_button("Start / Pause", color="green", order=0)
        status_indicator = server.gui.add_text("Status", initial_value="Playing", disabled=True, order=1)
        frame_slider = server.gui.add_slider("Frame", min=0, max=0, step=1, initial_value=0, order=2)
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0, order=3)

    def effective_frame_dt() -> float:
        return frame_dt / max(float(state["speed"]), 1e-6)

    def update_indicator() -> None:
        status_indicator.value = "Playing" if state["playing"] else "Paused"

    def apply_cursor(cursor: int, *, sync_slider: bool) -> None:
        frame_indices = state["frame_indices"]
        cursor = max(0, min(cursor, len(frame_indices) - 1))
        state["cursor"] = cursor
        _apply_qpos_frame(data, state["qpos"][frame_indices[cursor]])
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)
        if sync_slider:
            state["syncing_slider"] = True
            state["ignore_slider_value"] = cursor
            frame_slider.value = cursor
            state["syncing_slider"] = False

    def select_motion(label: str) -> None:
        if state["label"] == label:
            return
        qpos = load_motion(motion_by_label[label])
        if qpos.shape[1] != model.nq:
            raise ValueError(f"Motion qpos width {qpos.shape[1]} does not match model.nq={model.nq}")
        frame_indices = list(_iter_frame_indices(int(qpos.shape[0]), start, end, stride))
        if not frame_indices:
            raise ValueError("No frames selected. Check --start/--end/--stride.")

        state["label"] = label
        state["qpos"] = qpos
        state["frame_indices"] = frame_indices
        state["cursor"] = 0
        state["playing"] = True
        state["pending_seek"] = None
        state["next_time"] = time.monotonic()
        frame_slider.max = len(frame_indices) - 1
        state["syncing_slider"] = True
        state["ignore_slider_value"] = 0
        frame_slider.value = 0
        state["syncing_slider"] = False
        update_indicator()
        apply_cursor(0, sync_slider=False)

    def show_motion(label: str) -> None:
        if motion_dropdown.value != label:
            motion_dropdown.value = label
        select_motion(label)

    @motion_dropdown.on_update
    def _(_) -> None:
        show_motion(str(motion_dropdown.value))

    def shift_motion(delta: int) -> None:
        index = labels.index(str(motion_dropdown.value))
        show_motion(labels[(index + delta) % len(labels)])

    @prev_button.on_click
    def _(_) -> None:
        shift_motion(-1)

    @next_button.on_click
    def _(_) -> None:
        shift_motion(1)

    @play_pause_button.on_click
    def _(_) -> None:
        should_play = not state["playing"]
        if should_play and state["cursor"] >= len(state["frame_indices"]) - 1 and not loop:
            state["cursor"] = 0
        state["playing"] = should_play
        if should_play:
            state["next_time"] = time.monotonic()
        update_indicator()

    @frame_slider.on_update
    def _(_) -> None:
        slider_value = int(frame_slider.value)
        if state["syncing_slider"] or state["ignore_slider_value"] == slider_value:
            state["ignore_slider_value"] = None
            return
        state["pending_seek"] = slider_value
        state["playing"] = False
        update_indicator()

    @speed_slider.on_update
    def _(_) -> None:
        state["speed"] = float(speed_slider.value)
        if state["playing"]:
            state["next_time"] = time.monotonic()

    show_motion(initial_label)
    print(f"[any4hdmi-view] {len(labels)} motions. Use the Motion dropdown, Previous, or Next.")
    print(f"[any4hdmi-view] mjviser server: http://localhost:{port}")
    print("[any4hdmi-view] Open the URL in a browser. Press Ctrl+C to quit.")

    while True:
        pending_seek = state["pending_seek"]
        if pending_seek is not None:
            state["pending_seek"] = None
            apply_cursor(int(pending_seek), sync_slider=False)
            state["next_time"] = time.monotonic() + effective_frame_dt()

        if state["playing"] and time.monotonic() >= state["next_time"]:
            next_cursor = state["cursor"] + 1
            if next_cursor >= len(state["frame_indices"]):
                if loop:
                    next_cursor = 0
                else:
                    next_cursor = len(state["frame_indices"]) - 1
                    state["playing"] = False
                    update_indicator()
            apply_cursor(next_cursor, sync_slider=True)
            state["next_time"] = time.monotonic() + effective_frame_dt()

        time.sleep(min(0.01, max(effective_frame_dt() / 4.0, 0.001)))


def main() -> None:
    args = _parse_args()
    motion_path = _resolve_motion_path(args.motion)
    dataset_root = find_dataset_root(motion_path)
    manifest = load_manifest(dataset_root)
    motions = dataset_motions(dataset_root, manifest)
    initial_label = initial_motion_label(motion_path, motions)

    with temp_mjcf_with_floor(manifest.mjcf_path) as viewer_mjcf_path:
        model = mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))
    data = mujoco.MjData(model)
    qpos = load_motion(dict(motions)[initial_label])
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Motion qpos width {qpos.shape[1]} does not match model.nq={model.nq}")

    frame_indices = list(_iter_frame_indices(int(qpos.shape[0]), args.start, args.end, args.stride))
    if not frame_indices:
        raise ValueError("No frames selected. Check --start/--end/--stride.")

    if args.headless:
        for frame_idx in tqdm(frame_indices, desc="Playing", unit="frame"):
            _apply_qpos_frame(data, qpos[frame_idx])
            mujoco.mj_forward(model, data)
        return

    fps = float(args.fps) if args.fps is not None else 1.0 / manifest.timestep
    _run_mjviser(
        model,
        data,
        motions,
        initial_label,
        start=args.start,
        end=args.end,
        stride=args.stride,
        frame_dt=1.0 / fps,
        loop=args.loop,
        port=args.port,
    )


if __name__ == "__main__":
    main()
