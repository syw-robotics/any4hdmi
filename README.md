# any4hdmi

Convert, view, and load humanoid motion datasets in a unified MuJoCo `qpos` format.
Supports LAFAN, SONIC, and 100STYLE conversion, with local or Hugging Face dataset loading.

Each dataset contains a `manifest.json` (model reference, timestep, and joint schema)
and one `motions/**/*.npz` per clip storing `qpos`. MJCF models and meshes are
resolved from Hugging Face rather than bundled in this repository.

## Install

Requires Python 3.10+ and `uv`. From the repository root:

```bash
uv sync
```

## View a dataset

```bash
uv run any4hdmi-view --motion hf://elijahgalahad/any4hdmi-lafan
# Or open a local converted dataset:
uv run any4hdmi-view --motion output/lafan
```

Open the printed browser URL (default: `http://localhost:8080`). Select a motion,
press **Space** to pause/resume, and drag **Frame** to seek. The first hosted load
may download the dataset snapshot and model assets.

## Convert source data

```bash
uv run any4hdmi-convert-lafan \
  --csv-dir data/LAFAN1_Retargeting_Dataset/g1 \
  --out-dir output/lafan
```

SONIC and 100STYLE use `any4hdmi-convert-sonic` and `any4hdmi-convert-axellwppr`.
See the [conversion guide](docs/pipeline.md) for source downloads and examples.

## Load in Python

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

## Publish a dataset

```bash
uv run any4hdmi-upload output/lafan your-namespace/any4hdmi-lafan
```

## Documentation

- [Dataset format](docs/dataset_format.md)
- [Conversion pipeline](docs/pipeline.md)
- [Runtime loading](docs/dataset.md)
- [Viewer: trimming, diagnostics, and comparison](docs/viewer.md)
- [Ground penetration correction](docs/grounding.md)
