# any4hdmi

Convert, view, and load humanoid motion datasets in a unified MuJoCo `qpos` format.
Supports LAFAN, SONIC, and 100STYLE conversion, with local or Hugging Face dataset loading.

Each dataset contains a `manifest.json` (model reference, timestep, and joint schema)
and one `motions/**/*.npz` per clip storing `qpos`. G1 MJCF models and meshes are
bundled in [`src/any4hdmi/assets/g1/`](src/any4hdmi/assets/g1/) and loaded locally.
Model loading does not check Hugging Face or download assets. Existing G1
`hf://elijahgalahad/g1_xmls...` model references resolve to these bundled files;
custom models must use a local XML path with their mesh assets.

## Install

Requires Python 3.10+ and `uv`. From the repository root:

```bash
uv sync
```

## Find a feature

| Feature | Entry point | Guide |
| --- | --- | --- |
| Convert source datasets | `any4hdmi-convert-{lafan,sonic,axellwppr}` | [Conversion pipeline](docs/pipeline.md) |
| Play, trim, compare, and inspect motions | `any4hdmi-view` | [Viewer](docs/viewer.md) |
| Screen motions and export a whitelist | `any4hdmi-select` | [Interactive selection](docs/selection.md) |
| Repair ground penetration and self collision | `any4hdmi-repair` or the viewer's Repair panel | [Motion repair](docs/repair.md) |
| Correct foot penetration | `any4hdmi-ground` | [Grounding](docs/grounding.md) |
| Filter a dataset by motion checks | `any4hdmi-filter-dataset` | [Filtering](docs/pipeline.md#typical-commands) |
| Load motions for training | `load_any4hdmi_dataset()` | [Dataset runtime](docs/dataset.md) |
| Score quality, train HME, and explore embeddings | `any4hdmi-limmt-*` | [LIMMT](src/any4hdmi/limmt/README.md) |
| Publish to Hugging Face | `any4hdmi-upload` | [Publishing](#publish-a-dataset) |

## Usage

Run these commands from the repository root. Replace the example input paths
with your own data. Append `--help` to any command to see all options.

### View a dataset

```bash
uv run any4hdmi-view --motion hf://elijahgalahad/any4hdmi-lafan
# Or open a local converted dataset:
uv run any4hdmi-view --motion output/lafan
```

Open the printed browser URL (default: `http://localhost:8080`). Select a motion,
press **Space** to pause/resume, and drag **Frame** to seek. The first hosted load
may download the motion dataset snapshot. Robot XML and meshes are already local.

### Convert source data

```bash
# LAFAN
uv run any4hdmi-convert-lafan \
  --csv-dir data/LAFAN1_Retargeting_Dataset/g1 \
  --out-dir output/lafan

# SONIC
uv run any4hdmi-convert-sonic \
  --csv-dir data/g1_sonic/complete/g1/csv \
  --out-dir output/sonic

# 100STYLE / Axellwppr
uv run any4hdmi-convert-axellwppr \
  --input data/100style.tar \
  --out-dir output/100style
```

See the [conversion guide](docs/pipeline.md) for source downloads and examples.

### Select motions

```bash
uv run any4hdmi-select \
  --input-path output/lafan \
  --project-path output/lafan_selection
```

Open the printed URL (default: `http://127.0.0.1:8081`), batch filter the dataset,
review decisions, and export a whitelist. The project stores rules, cached
features, and review state separately from the source.
See [interactive selection](docs/selection.md) for the workflow and loader integration.

### Repair motions

```bash
# Inspect without writing a dataset.
uv run any4hdmi-repair --motion output/lafan --dry-run

# Repair and export to a new dataset.
uv run any4hdmi-repair \
  --motion output/lafan \
  --output-root output/lafan_repaired
```

The viewer also exposes the same pipeline in its **Repair** panel.
See [motion repair](docs/repair.md) for preview, acceptance criteria, and export policies.

### Correct ground penetration

```bash
uv run any4hdmi-ground \
  --motion output/lafan \
  --output-root output/lafan_grounded
```

Correct penetration by adjusting root height and export to a new dataset.
See [grounding](docs/grounding.md) for floor height and correction limits.

### Filter a dataset

```bash
uv run any4hdmi-filter-dataset \
  --input-root output/lafan \
  --output-root output/lafan_filtered
```

Apply FK-based motion checks and copy kept clips to the output dataset.
Add `--dry-run` to emit a report without copying motions.
See the [conversion pipeline](docs/pipeline.md#typical-commands) for more examples.

### Load in Python

```python
from pathlib import Path
from any4hdmi.dataset import load_any4hdmi_dataset

dataset = load_any4hdmi_dataset(
    root_path="hf://elijahgalahad/any4hdmi-lafan",  # Or a local dataset root.
    target_fps=50,
    base_dir=Path.cwd(),
    num_envs=1,
    full_motion=True,
)
```

See [dataset runtime](docs/dataset.md) for full/windowed loading and FK caching.

### Score quality and generate subsets with LIMMT

Run these stages in order on an existing converted dataset (here `output/amass`):

```bash
# Score physical quality and collect passing motions.
uv run any4hdmi-limmt-score \
  --input-path output/amass \
  --project-path output/amass_limmt

# Train HME on passing motions, then compute their embeddings.
uv run any4hdmi-limmt-train-hme --project-path output/amass_limmt
uv run any4hdmi-limmt-embed --project-path output/amass_limmt

# Generate subsets at the requested ratios and visualize the results.
uv run any4hdmi-limmt-gqs \
  --project-path output/amass_limmt \
  --ratios 0.04 0.08 0.16 0.32
uv run any4hdmi-limmt-visualize --project-path output/amass_limmt
```

See [LIMMT](src/any4hdmi/limmt/README.md) for GPU acceleration, training settings,
and output layout.

### Publish a dataset

```bash
uv run any4hdmi-upload output/lafan your-namespace/any4hdmi-lafan
```

## Code navigation

Start in [`src/any4hdmi/`](src/any4hdmi). Reusable motion logic lives in
`core/`; command-line entry points and browser UI live in `scripts/`.
Related modules share a feature directory:

```text
src/any4hdmi/
├── assets/g1/          # Bundled G1 XML, meshes, and source provenance
├── core/
│   ├── repair/          # motion.py: solver; geometry.py: checks; dataset.py: batch sessions
│   ├── selection/       # features.py: screening; project.py: cache, decisions, exports
│   ├── editing.py      # Clip export
│   ├── grounding.py    # Foot penetration correction
│   ├── support.py      # Shared foot support inference
│   ├── format.py       # Manifests and motion files
│   └── model.py        # Robot models and joint schema
├── scripts/
│   ├── preprocess/     # Source converters and shared conversion helpers
│   ├── viewer/         # app.py: playback; comparison, contacts, issues, support
│   ├── repair/         # cli.py: batch command; panel.py: viewer controls
│   ├── select.py       # Interactive motion selection
│   ├── ground.py       # Batch foot penetration correction
│   ├── filter.py       # Dataset filtering
│   └── upload.py       # Hugging Face publishing
├── dataset/            # Runtime loading, full/windowed storage, FK caches
├── fk/                 # Forward kinematics runners
├── limmt/              # Physical scores, HME, embeddings, quality selection
└── utils/              # Shared dataset, MJCF, and math helpers
```

For example, start with [`core/repair/motion.py`](src/any4hdmi/core/repair/motion.py)
to change the repair algorithm, or [`scripts/repair/panel.py`](src/any4hdmi/scripts/repair/panel.py)
to change its viewer controls. Single-module features remain files until they
need multiple modules. Regression tests live in [`tests/`](tests), named after
the feature they cover.

## Documentation

- [Dataset format](docs/dataset_format.md)
- [Conversion pipeline](docs/pipeline.md)
- [Runtime loading](docs/dataset.md)
- [Viewer: trimming, diagnostics, and comparison](docs/viewer.md)
- [Ground penetration correction](docs/grounding.md)
- [Motion repair: Viser and batch CLI](docs/repair.md)
- [Interactive motion selection](docs/selection.md)
- [LIMMT: quality scoring and embeddings](src/any4hdmi/limmt/README.md)
