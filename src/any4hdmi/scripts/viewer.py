from __future__ import annotations

import argparse
import time
from bisect import bisect_left
from functools import wraps
from pathlib import Path
from threading import RLock

import mujoco
from mjhub import temp_mjcf_with_floor
from tqdm import tqdm

from any4hdmi.core.editing import export_motion_clip
from any4hdmi.core.format import MOTIONS_SUBDIR, find_dataset_root, load_manifest, load_motion
from any4hdmi.dataset.loading import resolve_input_paths
from any4hdmi.scripts.viewer_contacts import ContactHighlights
from any4hdmi.scripts.viewer_comparison import ComparisonView
from any4hdmi.scripts.viewer_issues import IssueNavigator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay and trim qpos-only any4hdmi motions with MuJoCo.")
    parser.add_argument(
        "--motion",
        required=True,
        help=(
            "One any4hdmi .npz file, dataset root, or directory within a dataset; "
            "also accepts hf://<namespace>/<repo>[@revision][/<path>] (default revision: main). "
            "Requires manifest.json in the directory or an ancestor. A file selects the "
            "initial motion. A directory under motions_subdir limits the GUI to motions "
            "in that directory (recursively), with names relative to it. A dataset root "
            "or file lists all dataset motions."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS override. Defaults to 1 / manifest.timestep.",
    )
    parser.add_argument("--start", type=int, default=0, help="Inclusive, zero-based source start frame (default: 0).")
    parser.add_argument(
        "--end", type=int, default=-1,
        help="Exclusive source end frame, capped at motion length. Any negative value means the full length.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a viewer window.")
    parser.add_argument("--port", type=int, default=8080, help="mjviser server port.")
    parser.add_argument("--compare", default=None, help="Original dataset root or motion .npz; grounded datasets auto-discover their original through the manifest.")
    return parser.parse_args()


def _iter_frame_indices(length: int, start: int, end: int, stride: int) -> range:
    resolved_end = end if end >= 0 else length
    return range(start, min(length, resolved_end), max(1, stride))


def _apply_qpos_frame(data: mujoco.MjData, qpos_frame) -> None:
    data.qpos[:] = qpos_frame
    data.qvel[:] = 0.0


def _resolve_motion_path(motion: str) -> Path:
    return resolve_input_paths(Path.cwd(), motion)[0]


def dataset_motions(dataset_root: Path, manifest, motion_path: Path) -> list[tuple[str, Path]]:
    motions_dir = dataset_root / manifest.payload.get("motions_subdir", MOTIONS_SUBDIR)
    if motion_path.is_dir() and motion_path.resolve().is_relative_to(motions_dir.resolve()):
        motions_dir = motion_path
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


def _create_motion_scene(server, model: mujoco.MjModel):
    import io

    import trimesh
    from mjviser import ViserMujocoScene

    # Model vendors use different geom groups. Keep world geometry separate
    # and classify robot geoms by their actual MuJoCo collision flags.
    robot = (model.geom_bodyid != 0) & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    collision = robot & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    visual = robot & ~collision
    model.geom_group[:] = 2
    model.geom_group[visual] = 0
    model.geom_group[collision] = 1
    model.geom_matid[collision] = -1
    model.geom_rgba[collision] = (40 / 255, 200 / 255, 90 / 255, 120 / 255)

    scene = ViserMujocoScene(server, model, num_envs=1)
    # As in LocoLab, add a BLEND material: vertex alpha alone is not enough
    # for mjviser's exported collision meshes to render transparently.
    handles = [group.handle for group in scene._mesh_groups if group.group_id == 1]
    handles += [handle for (_, group, _), handle in scene._fixed_geom_handles.items() if group == 1]
    for handle in handles:
        meshes = trimesh.load(io.BytesIO(handle.glb_data), file_type="glb", force="scene")
        for mesh in meshes.geometry.values():
            material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=mesh.visual.vertex_colors[0], alphaMode="BLEND", doubleSided=True
            )
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
        handle.glb_data = meshes.export(file_type="glb")

    has_visual = bool(visual.any())
    scene.geom_groups_visible = [has_visual, not has_visual and bool(collision.any()), True, False, False, False]
    scene._sync_visibilities()
    with server.gui.add_folder("Robot"):
        visual_checkbox = server.gui.add_checkbox("Visual", initial_value=scene.geom_groups_visible[0])
        collision_checkbox = server.gui.add_checkbox("Collision", initial_value=scene.geom_groups_visible[1])

        def update_layers(_) -> None:
            scene.geom_groups_visible[0] = bool(visual_checkbox.value)
            scene.geom_groups_visible[1] = bool(collision_checkbox.value)
            scene._sync_visibilities()
            scene.refresh_visualization()

        visual_checkbox.on_update(update_layers)
        collision_checkbox.on_update(update_layers)

    return scene


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
    compare: Path | None = None,
) -> None:
    try:
        import viser
    except ImportError as exc:
        raise ImportError(
            "any4hdmi-view now uses mjviser for interactive playback. "
            "Install the project dependencies with `uv sync` or run through `uv run`."
        ) from exc

    motion_by_label = dict(motions)
    labels = list(motion_by_label)
    server = viser.ViserServer(port=port, label="any4hdmi-view")
    server.gui.configure_theme(control_width="large")
    scene = _create_motion_scene(server, model)
    scene.camera_tracking_enabled = True
    scene.create_scene_gui()
    contact_highlights = ContactHighlights(server, model, data, scene)
    comparison = ComparisonView(server, model, scene, contact_highlights, compare)

    state = {
        "label": None,
        "qpos": None,
        "frame_indices": [],
        "playback_bounds": (0, 0),
        "playing": True,
        "cursor": 0,
        "pending_seek": None,
        "pending_source_seek": None,
        "next_time": time.monotonic(),
        "speed": 1.0,
    }
    state_lock = RLock()

    def synchronized(callback):
        @wraps(callback)
        def call(*args, **kwargs):
            with state_lock:
                return callback(*args, **kwargs)
        return call

    def seek_source(frame: int) -> None:
        state["pending_source_seek"] = frame
        state["pending_seek"] = None
        state["playing"] = False

    issues = IssueNavigator(server, model, contact_highlights, seek_source, synchronize=synchronized)

    with server.gui.add_folder("Motions", order=0):
        motion_dropdown = server.gui.add_dropdown("Motion", labels, initial_value=initial_label, order=0)
        prev_button = server.gui.add_button("Previous", order=1)
        next_button = server.gui.add_button("Next", order=2)

    with server.gui.add_folder("Playback", order=1):
        play_pause_button = server.gui.add_button("Start / Pause", color="green", hint="Shortcut: Space", order=0)
        status_indicator = server.gui.add_text("Status", initial_value="Playing", disabled=True, order=1)
        # viser clamps step to max - min at creation; a zero range loses step=1.
        frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0, order=2)
        speed_slider = server.gui.add_slider("Speed", min=0.5, max=3.0, step=0.5, initial_value=1.0, order=3)
        source_frame = server.gui.add_number("Source frame (0-based)", initial_value=0, disabled=True, order=4)

    with server.gui.add_folder("Trim motion", order=5, expand_by_default=False):
        trim_start = server.gui.add_number("Start frame (inclusive)", initial_value=0, min=0, step=1)
        trim_end = server.gui.add_number("End frame (inclusive)", initial_value=0, min=0, step=1)
        set_start_button = server.gui.add_button("Set start to current frame")
        set_end_button = server.gui.add_button("Set end to current frame")
        preview_button = server.gui.add_button("Preview selection")
        reset_button = server.gui.add_button("Reset selection")
        output_directory = server.gui.add_text("New dataset directory", initial_value="")
        export_button = server.gui.add_button("Export clip", color="green")
        edit_status = server.gui.add_text("Edit status", initial_value="", disabled=True)

    def effective_frame_dt() -> float:
        return frame_dt / max(float(state["speed"]), 1e-6)

    def update_indicator() -> None:
        status_indicator.value = "Playing" if state["playing"] else "Paused"

    def apply_cursor(cursor: int, *, sync_slider: bool) -> None:
        frame_indices = state["frame_indices"]
        cursor = max(0, min(cursor, len(frame_indices) - 1))
        state["cursor"] = cursor
        source_frame.value = frame_indices[cursor]
        _apply_qpos_frame(data, state["qpos"][frame_indices[cursor]])
        mujoco.mj_forward(model, data)
        with server.atomic():
            scene.update_from_mjdata(data)
            contact_highlights.update(force=True, frame=frame_indices[cursor])
            comparison.update(frame_indices[cursor])
        issues.set_frame(frame_indices[cursor], *state["playback_bounds"])
        if sync_slider:
            frame_slider.value = cursor

    @synchronized
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
        motion_dropdown.label = f"Motion ({labels.index(label) + 1}/{len(labels)})"
        state["qpos"] = qpos
        contact_highlights.load_motion(qpos, load_manifest(find_dataset_root(motion_by_label[label])).timestep)
        comparison.load(motion_by_label[label], qpos)
        issues.load(qpos, comparison.qpos)
        state["frame_indices"] = frame_indices
        state["playback_bounds"] = (frame_indices[0], min(end if end >= 0 else len(qpos), len(qpos)) - 1)
        state["cursor"] = 0
        state["playing"] = True
        state["pending_seek"] = None
        state["pending_source_seek"] = None
        state["next_time"] = time.monotonic()
        trim_end.max = len(qpos) - 1
        trim_start.max = len(qpos) - 1
        trim_start.value = frame_indices[0]
        trim_end.value = frame_indices[-1]
        output_directory.value = str(Path("output/motion_edits") / Path(label).with_suffix(""))
        edit_status.value = "Select source frames; both endpoints are included."
        frame_slider.max = max(1, len(frame_indices) - 1)
        frame_slider.disabled = len(frame_indices) == 1
        frame_slider.value = 0
        update_indicator()
        apply_cursor(0, sync_slider=False)

    def show_motion(label: str) -> None:
        if motion_dropdown.value != label:
            motion_dropdown.value = label
        select_motion(label)

    @motion_dropdown.on_update
    @synchronized
    def _(_) -> None:
        show_motion(str(motion_dropdown.value))

    def shift_motion(delta: int) -> None:
        index = labels.index(str(motion_dropdown.value))
        show_motion(labels[(index + delta) % len(labels)])

    @prev_button.on_click
    @synchronized
    def _(_) -> None:
        shift_motion(-1)

    @next_button.on_click
    @synchronized
    def _(_) -> None:
        shift_motion(1)

    @synchronized
    def toggle_playback(_) -> None:
        should_play = not state["playing"]
        if should_play and state["cursor"] >= len(state["frame_indices"]) - 1 and not loop:
            state["cursor"] = 0
        state["playing"] = should_play
        if should_play:
            state["next_time"] = time.monotonic()
        update_indicator()

    play_pause_button.on_click(toggle_playback)
    server.gui.add_command("Start / Pause", hotkey="space").on_trigger(toggle_playback)

    @frame_slider.on_update
    @synchronized
    def _(event) -> None:
        # Programmatic playback updates have no client; only user seeks pause.
        if event.client_id is None:
            return
        state["pending_seek"] = int(event.target.value)
        state["playing"] = False
        update_indicator()

    @speed_slider.on_update
    @synchronized
    def _(_) -> None:
        state["speed"] = float(speed_slider.value)
        if state["playing"]:
            state["next_time"] = time.monotonic()

    def selected_bounds() -> tuple[int, int]:
        first, last = int(trim_start.value), int(trim_end.value)
        if not 0 <= first <= last < len(state["qpos"]):
            raise ValueError(f"Expected 0 <= start <= end < {len(state['qpos'])}")
        return first, last

    @set_start_button.on_click
    @synchronized
    def _(_) -> None:
        trim_start.value = state["frame_indices"][state["cursor"]]

    @set_end_button.on_click
    @synchronized
    def _(_) -> None:
        trim_end.value = state["frame_indices"][state["cursor"]]

    @preview_button.on_click
    @synchronized
    def _(_) -> None:
        try:
            first, last = selected_bounds()
        except ValueError as exc:
            edit_status.value = str(exc)
            return
        state["frame_indices"] = list(range(first, last + 1, max(1, stride)))
        state["playback_bounds"] = (first, last)
        state["pending_seek"] = None
        frame_slider.max = max(1, len(state["frame_indices"]) - 1)
        frame_slider.disabled = len(state["frame_indices"]) == 1
        apply_cursor(0, sync_slider=True)
        state["playing"] = True
        state["next_time"] = time.monotonic() + effective_frame_dt()
        update_indicator()
        edit_status.value = f"Previewing frames {first}..{last} ({last - first + 1} source frames)."

    @reset_button.on_click
    @synchronized
    def _(_) -> None:
        trim_start.value = 0
        trim_end.value = len(state["qpos"]) - 1
        # Restore the full source timeline without changing the source motion.
        state["frame_indices"] = list(range(0, len(state["qpos"]), max(1, stride)))
        state["playback_bounds"] = (0, len(state["qpos"]) - 1)
        state["pending_seek"] = None
        frame_slider.max = max(1, len(state["frame_indices"]) - 1)
        frame_slider.disabled = len(state["frame_indices"]) == 1
        apply_cursor(0, sync_slider=True)
        state["playing"] = False
        update_indicator()
        edit_status.value = "Selection reset to the full motion."

    @export_button.on_click
    @synchronized
    def _(_) -> None:
        try:
            first, last = selected_bounds()
            if not output_directory.value.strip():
                raise ValueError("Enter a new dataset directory.")
            output_path = export_motion_clip(
                motion_by_label[state["label"]], Path(output_directory.value), first, last
            )
        except (OSError, ValueError) as exc:
            edit_status.value = f"Export failed: {exc}"
            return
        edit_status.value = f"Saved {last - first + 1} frames to {output_path}"

    show_motion(initial_label)
    print(f"[any4hdmi-view] {len(labels)} motions. Use the Motion dropdown, Previous, or Next.")
    print(f"[any4hdmi-view] mjviser server: http://localhost:{port}")
    print("[any4hdmi-view] Open the URL in a browser. Press Ctrl+C to quit.")

    while True:
        # Viser callbacks run in worker threads. A clip and its comparison,
        # indices and scan must change together between playback ticks.
        with state_lock:
            # Keep exact issue frames reachable even if --stride skipped them.
            pending_source = state["pending_source_seek"]
            if pending_source is not None:
                state["pending_source_seek"] = None
                indices = state["frame_indices"]
                cursor = bisect_left(indices, pending_source)
                if cursor == len(indices) or indices[cursor] != pending_source:
                    indices.insert(cursor, pending_source)
                frame_slider.max = max(1, len(indices) - 1)
                apply_cursor(cursor, sync_slider=True)
                update_indicator()

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

            with server.atomic():
                contact_highlights.update()
                comparison.update(state["frame_indices"][state["cursor"]])
            issues.advance()
        time.sleep(min(0.01, max(effective_frame_dt() / 4.0, 0.001)))


def main() -> None:
    args = _parse_args()
    motion_path = _resolve_motion_path(args.motion)
    dataset_root = find_dataset_root(motion_path)
    manifest = load_manifest(dataset_root)
    motions = dataset_motions(dataset_root, manifest, motion_path)
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
        compare=resolve_input_paths(Path.cwd(), args.compare)[0] if args.compare else None,
    )


if __name__ == "__main__":
    main()
