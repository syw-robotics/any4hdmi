# Interactive motion selection

Launch the independent Viser workspace:

```bash
uv run any4hdmi-select --input-path output/sonic --project-path output/sonic_selection
```

The default URL is `http://127.0.0.1:8081`. A source can also be an `hf://...` dataset root. For an existing environment whose console entry points have not been refreshed:

```bash
.venv/bin/python -m any4hdmi.scripts.select --input-path output/sonic --project-path output/sonic_selection
```

## Workflow

1. Click **Batch filter all motions**. This always scans the whole source dataset, even when the browser list is filtered. Progress and decision counts appear above the rules. **Cancel batch** stops between clips; completed results are saved but a partial run cannot export.
2. Filter the list by decision, reason or path. Lists are paginated at 100 clips. Select an event to loop its source-frame interval with 0.4 seconds of context. Frame intervals are zero-based and end-exclusive.
3. Adjust event actions (`allow`, `review`, `reject`) and thresholds. **Preview current motion** evaluates one clip. **Batch filter all motions** applies the new settings to everything using cached features. Editing rules marks old results pending; they cannot silently be exported.
4. Use **Manually keep**, **Manually reject**, and **Clear manual override** for exceptions. Set the note before clicking an override. Overrides survive recomputation and reopening, but expire when that motion's content changes. Invalid input files cannot be manually kept.
5. Click **Export manifest + whitelist** after a complete current batch. The export directory contains `selection_manifest.json` (provenance, thresholds, evidence, automatic and final decisions, overrides) and `filenames.txt` (only final `keep` paths relative to `motions_subdir`). Each export uses a separate directory under `PROJECT/exports/`. Source motions are never copied or edited.

An empty keep set cannot export: the training loader rejects empty whitelist files. Reopening restores decisions and rules; batch-filter again to verify source contents before exporting. The project directory should be owned by one selector process at a time.

## Training

Pass the exported whitelist as an absolute path to the existing loader:

```python
from pathlib import Path
from any4hdmi.dataset import load_any4hdmi_dataset

dataset = load_any4hdmi_dataset(
    root_path="output/sonic",
    filenames_path="/absolute/path/to/sonic_selection/exports/<id>/filenames.txt",
    target_fps=50,
    base_dir=Path.cwd(),
    num_envs=1,
    full_motion=True,
)
```

## Rules and limitations

The default policy rejects sustained body inversion and coherent near-ground rolls. Airborne motion, low posture, speed, and hand contact alone do not reject a clip; jumps and crawling are allowed.

Inversion uses the greater pelvis/torso tilt from world +Z (default 130 degrees for at least 0.08 seconds). Tilt within 20 degrees below that threshold is borderline and enters `review`. Ground roll integrates signed world-horizontal torso rotation in a bounded 1.5-second window (default 150 degrees, 75% directional coherence, at least 80% of the window near ground). Rotation above 80% of the configured limit is borderline. This also catches sideways rolls while the body longitudinal axis remains horizontal; yaw and alternating rocking do not accumulate unboundedly. Actions set to `allow` remain visible as detected events without affecting the decision.

These defaults are **uncalibrated screening heuristics**, not a learned action classifier or a guarantee of RL trackability. Extreme speed/undersampling, unusual body frames and long slow rolls can require threshold adjustments and manual review. One event rejects the whole clip; there is no safe-subclip export. There is no automatic dynamics-quality filtering in this tool.

Models need exactly one free root. The root is found from the joint schema, not hard-coded qpos columns. Torso defaults to `torso_link` when present, otherwise the free-root body; `--torso-body NAME` overrides it. Root and torso local +Z must point upright. Near-ground height is the root-origin world Z minus **Ground Z**, not sole height; set it for the robot and source coordinates. A flat horizontal ground reference is assumed. Bad files/mappings become `review` with an error message.

Features are CPU MuJoCo kinematics, computed once per clip, then compressed to disk (two quaternions and one height per frame). Cache identity includes motion bytes, the compiled model (including assets), manifest, torso mapping and feature version. No full training dataset or GPU window pools are allocated. Filtering checkpoints every 100 clips and on completion/cancellation. First-pass speed has not been benchmarked on full datasets; subsequent threshold changes avoid FK but still read source files to verify content.
