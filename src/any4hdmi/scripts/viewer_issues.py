from __future__ import annotations

from bisect import bisect_left, bisect_right

import mujoco
import numpy as np

from any4hdmi.scripts.viewer_contacts import ground_link_distances, self_collision_bodies


def issue_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1).tolist(), (np.flatnonzero(changes == -1) - 1).tolist()))


def next_issue_frame(intervals: list[tuple[int, int]], current: int, direction: int) -> int | None:
    if not intervals:
        return None
    starts = [first for first, _ in intervals]
    index = bisect_right(starts, current) if direction > 0 else bisect_left(starts, current) - 1
    return starts[index % len(starts)]


class MotionIssueScan:
    """Incremental full-resolution scan, isolated from the displayed MjData."""

    def __init__(self, model: mujoco.MjModel, qpos: np.ndarray, floor_id: int, *, feet_only: bool) -> None:
        self.model, self.qpos, self.floor_id, self.feet_only = model, qpos, floor_id, feet_only
        self.data = mujoco.MjData(model)
        self.completed = 0
        self.depth_m = np.zeros(len(qpos))
        self.ground_links = [()] * len(qpos)
        self.self_links = [()] * len(qpos)

    @property
    def done(self) -> bool:
        return self.completed == len(self.qpos)

    def advance(self, count: int = 128) -> None:
        for frame in range(self.completed, min(len(self.qpos), self.completed + count)):
            self.data.qpos[:] = self.qpos[frame]
            mujoco.mj_forward(self.model, self.data)
            distances = ground_link_distances(self.model, self.data, self.floor_id, feet_only=self.feet_only) if self.floor_id >= 0 else {}
            self.depth_m[frame] = max(0., -min(distances.values(), default=0.))
            self.ground_links[frame] = tuple(body for body, distance in distances.items() if distance < -.001)
            self.self_links[frame] = tuple(sorted(self_collision_bodies(self.model, self.data)))
            self.completed = frame + 1

    def mask(self, kind: str) -> np.ndarray:
        ground = self.depth_m > .001
        collision = np.array([bool(links) for links in self.self_links])
        return ground if kind == "Ground penetration" else collision if kind == "Self collision" else ground | collision

    def details(self, frame: int) -> str:
        parts = []
        for name, bodies in (("Ground", self.ground_links[frame]), ("Self collision", self.self_links[frame])):
            if bodies:
                names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body_{body}" for body in bodies]
                parts.append(f"{name}: {', '.join(names)}")
        if self.depth_m[frame] > .001:
            parts.append(f"penetration {self.depth_m[frame] * 1000:.1f} mm")
        return " | ".join(parts) or "No issue at this frame."


def timeline_image(scan: MotionIssueScan, frame: int) -> np.ndarray:
    image = np.full((24, 400, 3), 65, dtype=np.uint8)
    for row, kind, color in ((0, "Ground penetration", (255, 205, 40)), (13, "Self collision", (240, 60, 60))):
        # Map intervals, not downsampled points: preserve even a one-frame issue.
        for first, last in issue_intervals(scan.mask(kind)):
            left, right = first * 400 // len(scan.qpos), min(400, last * 400 // len(scan.qpos) + 1)
            image[row:row + 11, left:right] = color
    column = min(399, frame * 400 // len(scan.qpos))
    image[:, column:column + 1] = 255
    return image


class IssueNavigator:
    def __init__(self, server, model, contacts, seek, *, synchronize=lambda callback: callback) -> None:
        self.model, self.contacts, self.seek = model, contacts, seek
        self.scans = {}
        self.qpos = {}
        self.frame = 0
        self.bounds = (0, 0)
        self.intervals = []
        self.requested = False
        self.scope_value = contacts.scope.value
        with server.gui.add_folder("Problem frames", order=4, expand_by_default=False):
            self.scan_button = server.gui.add_button("Scan current clip")
            self.source = server.gui.add_dropdown("Issue source", ("Current", "Original"), initial_value="Current")
            self.kind = server.gui.add_dropdown("Issue type", ("Either", "Ground penetration", "Self collision"), initial_value="Either")
            self.previous = server.gui.add_button("Previous issue", disabled=True)
            self.next = server.gui.add_button("Next issue", disabled=True)
            self.selection = server.gui.add_dropdown("Issue interval", ("Not scanned",), initial_value="Not scanned", disabled=True)
            self.timeline = server.gui.add_image(np.full((24, 400, 3), 65, dtype=np.uint8), label="Ground penetration / Self collision")
            self.status = server.gui.add_text("Scan status", initial_value="Not scanned", disabled=True)
            self.details = server.gui.add_text("Frame issues", initial_value="", disabled=True)
            server.gui.add_markdown("Scan checks every source frame. Ground penetration >1 mm uses Ground links; self collision uses MJCF exclusions. Yellow/red strips show ground/self issues.")

        def request_scan(_) -> None:
            self.requested = True

        def select_interval(event) -> None:
            if event.client_id is not None and self.intervals:
                self.seek(self.intervals[self.selection.options.index(self.selection.value)][0])

        self.scan_button.on_click(synchronize(request_scan))
        self.previous.on_click(synchronize(lambda _: self.seek_next(-1)))
        self.next.on_click(synchronize(lambda _: self.seek_next(1)))
        self.selection.on_update(synchronize(select_interval))
        self.kind.on_update(synchronize(lambda _: self.refresh()))
        self.source.on_update(synchronize(lambda _: self.refresh()))

    def load(self, qpos: np.ndarray, original: np.ndarray | None) -> None:
        self.qpos = {"Current": qpos}
        if original is not None:
            self.qpos["Original"] = original
        self.scans = {}
        self.requested = False
        self.scope_value = self.contacts.scope.value
        self.source.options = tuple(self.qpos)
        self.source.value = "Current"
        self.status.value = "Not scanned"
        self.refresh()

    def set_frame(self, frame: int, first: int, last: int) -> None:
        bounds_changed = self.bounds != (first, last)
        self.frame, self.bounds = frame, (first, last)
        if bounds_changed:
            self.refresh()
        scan = self.scans.get(self.source.value)
        if scan is not None and scan.done:
            self.timeline.image = timeline_image(scan, frame)
            self.details.value = scan.details(frame)
            for first, last in self.intervals:
                if first <= frame <= last:
                    self.selection.value = f"{first}..{last}"
                    break

    def refresh(self) -> None:
        scan = self.scans.get(self.source.value)
        self.intervals = []
        if scan is not None and scan.done:
            mask = scan.mask(self.kind.value).copy()
            mask[:self.bounds[0]] = False
            mask[self.bounds[1] + 1:] = False
            self.intervals = issue_intervals(mask)
            self.status.value = f"{self.source.value}: {int(mask.sum())} issue frames / {len(self.intervals)} intervals"
            self.timeline.image = timeline_image(scan, self.frame)
            self.details.value = scan.details(self.frame)
        else:
            self.timeline.image = np.full((24, 400, 3), 65, dtype=np.uint8)
            self.details.value = "Not scanned"
        options = tuple(f"{first}..{last}" for first, last in self.intervals) or ("No intervals",)
        self.selection.options = options
        self.selection.value = options[0]
        self.selection.disabled = self.previous.disabled = self.next.disabled = not self.intervals

    def seek_next(self, direction: int) -> None:
        frame = next_issue_frame(self.intervals, self.frame, direction)
        if frame is not None:
            self.seek(frame)

    def advance(self) -> None:
        if self.contacts.scope.value != self.scope_value:
            self.load(self.qpos["Current"], self.qpos.get("Original"))
        if self.requested:
            self.requested = False
            self.scans = {name: MotionIssueScan(self.model, qpos, self.contacts.floor_id, feet_only=self.scope_value == "Feet") for name, qpos in self.qpos.items()}
            self.refresh()
        for name, scan in self.scans.items():
            if not scan.done:
                scan.advance()
                self.status.value = f"Scanning {name}: {scan.completed}/{len(scan.qpos)}"
                if all(item.done for item in self.scans.values()):
                    self.refresh()
                break
