# Ground Penetration Correction

[Back to README](../README.md) · [Viewer and comparison](viewer.md)

Export a new dataset with every robot surface above a horizontal floor:

```bash
uv run any4hdmi-ground \
  --motion output/lafan \
  --output-root output/lafan_grounded
uv run any4hdmi-view --motion output/lafan_grounded
```

`--motion` accepts a dataset root, a directory under its motion directory, or one
motion `.npz`; `hf://` dataset references are also supported. A file processes only
that clip. Relative motion paths are preserved in the output. The output directory
must be new and outside the source dataset. Source motions are preserved.

The command uses the manifest's MJCF and timestep. At every frame it measures
the lowest point across the floating-base robot's visual **and** collision
geometry. Knees, hands, pelvis, torso and head can determine the lift when they
are below the feet. World geometry and unrelated scene bodies are excluded.
The model must have exactly one floating-base free joint. `--foot-bodies` selects
feet for the per-foot diagnostics (default: G1 ankle-roll links); it does not
restrict which robot geometry is corrected.

Only root Z changes. Joint angles, root orientation, XY trajectory, frame count,
and timestep stay unchanged. Corrections lift the body; they never pull airborne
feet down to the floor. `--ground-z` defaults to `0` metres, and `--clearance` to
`0.001` metres. `--max-correction-speed` defaults to `0.2` m/s and limits changes in
the **added height**, not the original motion's vertical speed. Offline forward
and backward passes spread correction changes before/after penetration while
keeping every frame above its required clearance (within `1e-6` m).

`grounding_report.json` contains per-motion and aggregate penetration counts and
depths across the whole robot, per-body and per-foot minimum heights, and per-motion
lift/speed/vertical acceleration diagnostics. `body_names` gives the order of
`min_body_z_m`; settings record `scope: whole_robot`. Review clips with large lifts:
moving the whole body can leave the other support foot floating. The rate limit
does not guarantee smooth acceleration.
This operation addresses robot/floor penetration on flat ground; it does not solve
foot sliding, self collisions, or terrain contact. It does not change runtime
full/windowed dataset loading; FK caches are built from the exported motions.

## 实现记录与后续计划（2026-11-05）

### 当前方法

- **Feet contact 可视化**：当前黄色表示 `Stable foot support`，已替代早期仅按离地 5 mm 判定的方式。
  从 G1 脚底 mesh 选取固定脚跟／前掌采样点，在地面坐标系中计算高度和相邻帧速度。
  进入条件为高度 −10～20 mm、速度 ≤0.15 m/s，连续满足 60 ms（至少两帧）；
  保持条件放宽至 −15～30 mm、≤0.25 m/s，减少闪烁。每只脚任一点满足即可，整段预计算，
  与播放速度和 stride 无关。这是运动学推断，缓慢滑动仍可能误判，并非接触力测量。
  实现：[viewer_support.py](../src/any4hdmi/scripts/viewer_support.py)、[viewer_contacts.py](../src/any4hdmi/scripts/viewer_contacts.py)。
- **Ground 修正**：逐帧 FK，检查浮动机器人全部 visual/collision 几何的最低点（不只脚部）。
  必需抬升量为 `max(0, ground_z + clearance - min_z)`；默认 clearance 为 1 mm。
  前后两遍传播抬升量，限制新增高度变化速度为 0.2 m/s，同时保证不穿地。只修改 root Z，导出新数据集。
  支撑识别尚未接入修正：不主动压低腾空动作，但平滑抬升可能延伸到腾空帧，也可能让支撑脚悬空。
  实现：[grounding.py](../src/any4hdmi/core/grounding.py)、[ground.py](../src/any4hdmi/scripts/ground.py)。
- **Self collision 当前仅可视化**：按 MuJoCo 碰撞 mask/exclude 检查同一机器人不同 link 的接触，
  穿透／接触 link 标红，优先于黄色；整体 root 平移不能修复自碰撞。

### 参考来源

参考本机 `/home/syw/.gitrepos/human-humanoid-tools` 的以下实现；不是直接移植：

- `hhtools/retarget/newton_basic/feet_stabilizer.py::_apply_foot_planting`：低高度、低移动量判断 planted 状态。
  当前 Viewer 使用实际 timestep 换算速度，并增加固定脚底采样、持续时间和进入／退出阈值。
- `hhtools/retarget/newton_basic/pipeline.py::_clamp_solved_foot_heights`：参考全身最低几何点抬升 root 的思路。
  `hhtools/retarget/clip_ground_snap.py::snap_joint_q_clip_floor` 是整段常量高度对齐，未作为当前逐帧修正算法。
- `hhtools/robot/foot_geometry.py::clamp_joint_q_foot_lateral_clearance`：调整髋外展角缓解双脚互穿／交叉，
  可供未来局部修正参考；不等于全身自碰撞优化。
- `hhtools/retarget/interaction_mesh/collision.py::compute_nonpenetration_constraints`：距离与解析 Jacobian 线性约束可借鉴，
  但其候选筛选只包含机器人与场景，不能直接用于机器人自身 link 对。

### 自碰撞修复计划（未实现）

1. **建立基准**：收集双脚互穿、膝盖互撞、手臂穿躯干，以及合法交叉腿／贴身动作；记录 link 对、深度和问题区间。
2. **离线局部优化**：按模型碰撞规则筛选自碰撞对，以距离 Jacobian 约束迭代调整相关关节，
   同时限制关节范围、单步幅度、原姿态偏差和时间变化；参考双脚局部修正，但不把合法交叉腿一律展开。
3. **联合保护动作**：加入地面不穿透和可信支撑脚的位置约束；腾空段不施加贴地约束。
   冲突或未收敛帧明确报告，不宣称全部修好。
4. **验收与导出**：导出新数据集并同步对比；验证碰撞帧数／最大深度下降，且无新增穿地、关节越界、明显滑步或抖动。
   覆盖无碰撞动作不变及腾空动作保留；报告残余问题和最大姿态改变量。
