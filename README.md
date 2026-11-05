# any4hdmi

`any4hdmi` defines one simple `qpos`-based motion format for HDMI-related datasets.

- one dataset-level `manifest.json`
- one `motions/**/*.npz` file per motion
- each motion file stores only `qpos`
- each manifest stores an MJCF reference on Hugging Face, not a vendored local XML/STL tree

The motion plus the dataset timestep from `manifest.json` is enough to replay the clip in MuJoCo.

## Datasets

Clone the source datasets from Hugging Face:

```bash
git clone https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset data/LAFAN1_Retargeting_Dataset
git clone https://huggingface.co/datasets/bones-studio/seed data/seed

tar xzf data/seed/g1.tar -C data/seed/g1
```

## Commands

Convert LAFAN:

```bash
uv run any4hdmi-convert-lafan \
  --csv-dir data/LAFAN1_Retargeting_Dataset/g1 \
  --out-dir output/lafan
```

Convert SONIC:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir data/seed/g1/csv \
  --out-dir output/sonic
```

Convert 100STYLE from the Axellwppr `MotionDataset` tarball:

```bash
uv run any4hdmi-convert-axellwppr \
  --input /home/elijah/Downloads/100style.tar \
  --out-dir output/100style
```

Override the MJCF reference if needed:

```bash
uv run any4hdmi-convert-sonic \
  --csv-dir data/seed/g1/csv \
  --out-dir output/sonic \
  --mjcf-repo elijahgalahad/g1_xmls \
  --mjcf-path g1-mode_13_15.xml \
  --mjcf-revision main
```

### View and trim motions

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

Runtime loading also accepts a hosted dataset root:

```python
load_any4hdmi_dataset(
  root_path="hf://elijahgalahad/any4hdmi-lafan",
  target_fps=50,
  base_dir=Path.cwd(),
  num_envs=1,
  full_motion=True,
  shard=False,
)
```

Upload a converted dataset folder to Hugging Face:

```bash
uv run any4hdmi-upload output/lafan elijahgalahad/any4hdmi-lafan
```

Headless check:

```bash
uv run any4hdmi-view \
  --motion output/sonic/motions/230322/reach_jump_R_001__A299_M.npz \
  --headless
```

Pipeline details live in [docs/pipeline.md](docs/pipeline.md).
Dataset format details live in [docs/dataset_format.md](docs/dataset_format.md).
