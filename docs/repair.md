# Motion repair

Repair existing robot qpos with the same pipeline from the Viser viewer or CLI.
The source dataset is preserved. Grounding and self-collision correction are
jointly optimized with pose preservation and temporal regularization.

## Viser

```sh
uv run any4hdmi-view --motion /path/to/dataset
```

Expand **Repair**. Choose **Current motion**, **Selected motions**, or
**All loaded motions**. For a custom batch, use **Selection motion** and
**Add to selection / Remove from selection**. A subdirectory opened with
`--motion` limits All loaded motions to that directory. Opening a single file
still lists its dataset as usual; the default repair scope is Current motion.

**Check selected** reports problems without solving. **Repair selected** includes
checking, so a separate check is optional. Jobs snapshot their selection/settings,
process clips sequentially in the background, and leave playback usable.
**Cancel repair** stops at a scan/solver checkpoint and retains completed results.
A failed clip does not abort the rest. Results are staged on disk in a session
scratch directory rather than retaining all candidates in RAM. Starting a new job
replaces the previous job/results; export any results you want to keep first.

Use **Result filter**, **Repair result**, and **Previous/Next result** to inspect
individual results from a batch. **Preview repair** displays the candidate with
its job-input original using synchronized comparison and issue navigation.
Completion never changes the displayed pose automatically. **Restore source**
returns to ordinary playback. Switching clips also returns to source playback.
The Trim export button is disabled during repair preview; use Repair export.

Repair always uses full source frames and the manifest timestep. Playback FPS,
speed, stride, and Trim selection do not affect repair. To repair a shorter clip,
first export it using Trim and open that dataset.

## Outcomes and export

- **Unchanged**: no correction needed and checks pass.
- **Passed**: candidate meets the recorded acceptance criteria.
- **Needs review**: residual penetration/limits, support drift, excessive correction,
  or unavailable support validation. Inspect reasons and before/after metrics.
- **Failed**: unreadable/invalid motion or solver failure.

Select an export scope: all completed results, current result, or the custom
selected motions. Select a policy:

- **Preserve selection**: passed candidates; original motions for all other outcomes.
- **Passed / unchanged only**: exclude unresolved and failed motions, recording omissions.

The panel shows inclusion, unresolved-original, exclusion and unfinished counts.
Invalid files cannot be copied as valid originals; exclude them or use passed-only
export. Check-only results and unfinished jobs are not exportable. After cancelling,
you can export completed results. Choose a new directory outside the input dataset,
then click **Export repair**. Export runs in the background. No source is overwritten.

Output contains `manifest.json`, `repair_report.json`, and the selected
`motions/**/*.npz` with their original relative paths, frame counts and timestep.
The report distinguishes candidate diagnostics from the actual exported version.
The manifest is written last; incomplete exports are not loadable datasets.
Opening a repaired dataset automatically finds its input for comparison through
manifest provenance. Derived velocities and FK caches are rebuilt downstream.

## CLI

```sh
uv run any4hdmi-repair --motion /path/to/input --output-root /path/to/repaired
uv run any4hdmi-repair --motion /path/to/input --dry-run
uv run any4hdmi-repair --motion /path/to/input --output-root /path/to/passed --passed-only
```

The CLI shows completed/total motions, the current motion path, and elapsed time.
Long clips can keep the completed count unchanged while CPU geometry checks or
sparse least-squares solving are running. Export is announced separately.

Input may be a dataset, motion subdirectory, or one `.npz`; `hf://` references
are supported. A CLI file processes only that file. `--dry-run` prints findings
without writing. `--no-ground` and `--no-self-collision` disable the corresponding
repair target, while measurement and no-new-violation checks remain enabled.
`--ground-z` sets horizontal floor height (default 0 m).

Exit status is 0 for a fully accepted run, 1 if unresolved/failed motions remain,
and 130 for interruption. A fallback dataset may still be exported when status
is 1. Malformed sources require passed-only export or removal from the input scope.

## Solver and limits

For isolated ground penetration, the pipeline first tries a rate-limited root lift
and accepts it only after the same support, pose and temporal checks pass.
Otherwise, it uses MuJoCo geometry/contact Jacobians and SciPy sparse bounded
least squares over all affected frames and their transition neighborhoods.
Quadratic elastic distance penalties handle conflicting geometric constraints.
It preserves the original pose and penalizes the first/second temporal differences
of the *correction*, not simply slow robot motion. Root updates use MuJoCo
configuration integration so quaternions remain normalized. No post-smoothing or
unconditional final root lift is applied.

Supported models have one world-attached floating base and hinge/slide joints.
Ground checks include all visual/collision surfaces of that robot; self collision
uses MuJoCo's contact rules and exclusions. Repair assumes a horizontal floor at
`ground_z`; it does not model terrain, scene obstacles or dynamic balance.
G1 support protection uses fixed heel/forefoot samples inferred on the original
clip. Each stable sample follows its original trajectory, preventing added drift
without forcing whole-foot orientation or pulling airborne feet down. This does
not remove pre-existing slow sliding. Models without both expected G1 foot names
cannot receive a fully validated modified result and are marked Needs review.

Defaults: 1 mm geometric tolerance, 1 mm target clearance, 0.3 s transition
neighborhood, 0.15 m maximum per-axis root translation correction, 0.35 rad/m
maximum joint correction (also used for root rotation components), 5 mm support
drift. Maximum componentwise correction speed/acceleration are 2 and 30 in the
corresponding tangent-coordinate units per second/second squared. Settings are
recorded per result. These are conservative starting values, not physical
feasibility guarantees; extreme motions may need review. Acceptance rechecks the
actual exported float32 candidate, including every frame outside edited intervals.
