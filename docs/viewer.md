# Motion Viewer

[Back to README](../README.md) · [Ground penetration correction](grounding.md)

## Open and play motions

`--motion` accepts one path or Hugging Face dataset URI. It determines the dataset
and initial clip. Passing a folder under `motions` limits browsing to that folder.

| Input | Initial clip | Motions available in the GUI |
| --- | --- | --- |
| A `.npz` under the dataset's motion directory | The specified file | All dataset motions |
| A dataset root containing `manifest.json` | First motion in sorted path order | All dataset motions |
| A directory under the manifest's motion directory | First motion in sorted path order within that directory | Only motions in that directory, recursively; names are relative to that directory |
| Another directory within the dataset | First motion in sorted path order | All dataset motions |
| `hf://<namespace>/<repo>[@revision][/<path>]` | Same rules after resolving the path | Same file/directory rules as local paths |

The viewer searches the input directory (or file's parent) and its ancestors for
the nearest `manifest.json`, then recursively lists `.npz` files under that
manifest's `motions_subdir` (default: `motions`), or just the supplied directory
when it is inside `motions_subdir`. Local paths can be absolute,
relative to the current working directory, or start with `~`.

An isolated `.npz` without a dataset manifest is insufficient: the manifest supplies
the timestep and MJCF model, and `qpos` must match the model's `nq`. Raw CSV/BVH,
legacy `motion.npz` + `meta.json` datasets, a parent directory containing several
datasets, glob patterns, multiple paths, and HTTPS URLs are not supported by this
argument. Convert raw data first; pass one dataset or a motion inside it.

Open a specific initial motion, or browse a whole dataset:

```bash
uv run any4hdmi-view --motion output/lafan/motions/dance1_subject2.npz
uv run any4hdmi-view --motion output/lafan
uv run any4hdmi-view --motion ../HumanoidMotionDataset/any4hdmi-converted/any4hdmi-g1-100style/motions/Akimbo/
```

Open a hosted dataset or a specific hosted motion (`@revision` defaults to `main`):

```bash
uv run any4hdmi-view --motion hf://elijahgalahad/any4hdmi-lafan
uv run any4hdmi-view --motion hf://elijahgalahad/any4hdmi-lafan@main/motions/dance1_subject2.npz
```

Hugging Face inputs use a cached dataset repository snapshot; specifying one motion
does not restrict the download to that file. The MJCF reference in the manifest may
also need to be downloaded. Open the Viser URL printed by the command (default:
`http://localhost:8080`). Use the **Motion** dropdown or **Previous / Next** to switch
clips. With `--headless`, only the initial clip is checked and no GUI is started.

Press **Space** to pause/resume playback, or drag **Frame** to seek and pause.
**Speed** ranges from 0.5× to 3.0× in steps of 0.5×.

In the **Robot** panel, toggle **Visual** and **Collision** independently. Collision
geometry is shown in translucent green, using the MJCF's `contype` / `conaffinity`
flags rather than vendor-specific geom groups. Visual is shown by default; models
with only collision geometry show Collision by default. The ground stays visible.

In **Contacts**, enable **Stable foot support** to color inferred supporting feet
yellow. Fixed sole samples near the heel and forefoot must stay near the floor
and move slowly in world space; low swings, sliding and penetration alone do not
qualify. A stable toe or heel can support while the ankle moves. This is a
kinematic estimate, not measured contact force or ground-truth contact labels.
Initial thresholds enter at -10 to +20 mm and <= 0.15 m/s, hold at -15 to +30 mm
and <= 0.25 m/s, and require 60 ms of qualifying samples (at least two frames).
Single-frame clips cannot establish support. Very slow sliding below the speed
tolerance can still qualify; these defaults need calibration for noisy retargeted
motions. G1 ankle-roll links and their descendants are colored
together; models without these named feet have no inferred highlights.
The full source clip is analyzed on load using the manifest timestep, independent
of playback FPS, speed, stride or seek order. Original and current clips are
analyzed separately. Continuous support keeps the same visible mesh across frames.
**Penetration scan links** selects **Feet** or **All links** for the problem-frame
scan; it does not change the yellow support highlights.
Enable **Self collision** to color both links in a self collision red, according to
the MJCF collision masks and exclusions. Scene-object collisions are excluded. Red
takes priority when a link also touches the ground. Unaffected links keep their
original appearance. Colors follow the Visual/Collision layer toggles and update
during playback, seeking, and motion changes, including toggling while paused.

CLI playback bounds use zero-based source frames: `--start` is inclusive and
`--end` is exclusive; any negative `--end` (default: `-1`) means the motion length.
An end beyond the motion length is capped at that length. For example, this selects
every second frame in the range 100 through 199: frames 100, 102, ..., 198.

```bash
uv run any4hdmi-view --motion output/lafan --start 100 --end 200 --stride 2
```

These playback bounds apply again when switching clips, so each clip must have at
least one frame in the requested range. The playback **Frame** slider indexes the
selected playback sequence; **Source frame (0-based)** shows the original frame
number to use for trimming.

## Locate problem frames

Expand **Problem frames** and click **Scan current clip** to locate ground
penetration deeper than 1 mm and self collisions. The scan checks every source
frame, including those skipped by `--stride`, and runs incrementally so the viewer
stays responsive. **Penetration scan links** selects the scan's ground scope; changing it
clears the scan so you can rescan. Self collisions always use the whole robot.
Choose **Issue type** and **Issue source** (Current or Original), then use
**Previous issue**, **Next issue**, or **Issue interval** to pause and seek both
copies to an exact source frame. Navigation groups consecutive issue frames and
wraps at the ends, within the current playback/trim bounds. A skipped issue frame
is inserted into the playback sequence when selected. The two timeline strips
show full-clip ground penetration (yellow) and self collision (red); the white
marker is the current source frame. **Frame issues** lists affected links and
penetration depth. Switching clips discards that clip's scan.

## Compare original and corrected motions

Opening a dataset produced by `any4hdmi-ground` automatically finds its original
dataset through `manifest.json`. Enable/disable the reference with **Comparison →
Show original**. Both robots share playback, pause, seeking, speed, and the source
frame number. The reference is labelled **Original** and shifted 1.4 metres along
world Y; **Current** is the opened motion. A common camera translation preserves
height differences. Contact coloring and layer toggles apply to both copies.

For other paired datasets, explicitly specify an original dataset root or one
motion file:

```bash
uv run any4hdmi-view --motion output/lafan_grounded --compare output/lafan
```

Dataset pairs match by the motion path relative to each manifest's motion
directory. They must use the same MJCF, qpos schema, timestep and frame count;
there is no automatic time or height alignment. A missing or mismatched pair
hides the original copy and displays the reason in **Original motion**, while
the current motion remains usable. **Issue source → Original** scans/navigates
the original even if the correction has removed all penetration.

## Trim and export

The Viser viewer supports trimming in the **Trim motion** panel (collapsed by default).
Unlike CLI `--end`, the panel's end frame is **inclusive**:

1. Pause or scrub playback, then use **Set start/end to current frame**, or enter
   source frame numbers directly. Frames are zero-based and both endpoints are included.
2. Click **Preview selection** to play the selected range. **Reset selection** restores
   the full motion timeline.
3. Choose a **New dataset directory** (default: `output/motion_edits/<motion path>`),
   then click **Export clip**. The directory must not already exist.

Export creates a `manifest.json` and `motions/<original filename>.npz`, preserving
the original timestep and all source frames in the selection. Playback speed,
`--fps`, and `--stride` do not resample the export. Source files are preserved.
The status field shows the saved path or an error. Open the exported dataset with
`uv run any4hdmi-view --motion <new dataset directory>`.
