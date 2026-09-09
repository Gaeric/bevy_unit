# bevy-tnua 使用 Wiki（bevy_waltz）

> 目的：把本项目当前如何接入/使用 `bevy-tnua`（v0.32，配合 avian3d）写清楚——概念、数据流、当前配置、正确喂入方式、已知坑——方便后续决策：是继续调 tnua，还是换一个控制插件。换插件前先读第 7 节。

## 0. 版本与环境

| 项 | 值 | 备注 |
|---|---|---|
| bevy-tnua | `0.32.0` | crates.io；本地源码镜像：`../bevy-tnua` |
| bevy-tnua-avian3d | `0.12.0` | 本地：`../bevy-tnua/avian3d` |
| avian3d | `0.7.0` | 本地：`../avian` |
| bevy | `0.19.0`（patch 到 `../bevy_engines/bevy_0.19/`） | root `Cargo.toml` `[patch.crates-io]` |
| bevy_enhanced_input | `0.26.0` | crates.io；本地 git 含 `v0.26.0` tag：`../bevy_enhanced_input` |

声明位置：`bevy_waltz/Cargo.toml`。root `Cargo.toml` 里到本地 tnua/avian 的 path 依赖目前是注释掉的。

构建/检查：

```sh
cargo check -p bevy_waltz
cargo run --example hs2_head   # 或本包 example
```

---

## 1. 概念总览（tnua 术语 → 本项目对应）

| tnua 概念 | 本项目位置 | 说明 |
|---|---|---|
| `TnuaScheme`（derive） | `WaltzTnuaCtrlScheme`（`character/mod.rs`） | 定义“basis + 动作集合”的枚举 |
| `TnuaController<S>` | 玩家实体上的组件 | 主要交互接口：喂 basis/action、查状态 |
| `TnuaConfig<S>` | `TnuaConfig::<WaltzTnuaCtrlScheme>` | 持有 `S::Config` asset（`WaltzTnuaCtrlSchemeConfig`），运行期参数都在这 |
| Basis | `TnuaBuiltinWalk` | 常规移动；v0.32 每帧只喂 `desired_motion` + `desired_forward`，参数全在 config |
| Action | `Jump/Crouch/Dash/Knockback/WallSlide/WallJump/Climb` | 特殊动作，靠“每帧持续喂 or 不喂”控制 |
| 后端集成 | `TnuaAvian3dPlugin::new(FixedUpdate)` | 把 sensor、motor 落到 avian |
| Action 计数 | `WaltzAirActionSlots` + `TnuaActionsCounter` + `TnuaAirActionsPlugin` | 空中动作次数（实现二段跳等） |
| 动画辅助 | `TnuaAnimatingState<AnimationState>` | 每帧喂状态枚举，返回 Maintain/Alter 指令 |

### 1.1 Basis vs Action 的本质区别

- **basis** 每帧必须喂（没输入就喂零向量），它一直在底层跑“浮空 + 弹簧 + 加速度 + 转向”。
- **action** 是“按下喂、松开不喂”的插层命令，同一时刻只有一个生效，由 tnua 内部仲裁谁顶替谁（含中断/延续语义）。

### 1.2 v0.32 的重要 API 变化（相对旧 demo）

旧代码里 `TnuaBuiltinWalk { desired_velocity, float_height, max_slope, ... }` 那种“全塞一个结构体”的写法 **在 0.32 已失效**。现在拆成：

- 每帧输入：`TnuaBuiltinWalk { desired_motion, desired_forward }`
- 参数：`TnuaBuiltinWalkConfig { speed, float_height, spring_*, acceleration, max_slope, ... }`（放 scheme config asset 里）

并且 `apply()` 内部执行：

```rust
let desired_velocity = self.desired_motion * config.speed;
```

→ **`desired_motion` 会被 config 里的 `speed` 再乘一次**。这是本项目目前最直接的坑（见 6.1）。

---

## 2. 文件地图（本项目 tnua 相关）

| 文件 | 作用 |
|---|---|
| `bevy_waltz/src/character/mod.rs` | scheme 定义、动作槽、config 默认值、角色 spawn、插件装配 |
| `bevy_waltz/src/character/config.rs` | `CharacterMotionConfig`（旧版遗留字段）等 |
| `bevy_waltz/src/character/animating.rs` | 读 controller 状态 → 切动画 |
| `bevy_waltz/src/character/sound.rs` | 读跳记忆做音效 |
| `bevy_waltz/src/control/character_ctrl.rs` | **当前喂入控制**（basis/jump 的入口） |
| `bevy_waltz/src/control/demo.rs` | 旧的轮询式喂入示例（未挂载，编译不参与） |
| `bevy_waltz/src/level_switch/*` | 关卡搭建、`Climable` 等 |
| `bevy_waltz/src/lib.rs` | 插件组装 |

角色 spawn 链：

```
setup_demo_player / setup_player
  └─ setup_character_with_entity_cmd
       └─ RigidBody::Dynamic + TnuaController + TnuaConfig(asset)
          + TnuaObstacleRadar + TnuaBlipReuseAvoidance + TnuaToggle
          + TnuaAnimatingState + TnuaGhostOverwrites + TnuaSimpleFallThroughPlatformsHelper
```

> 注意：当前 `Startup` 挂的是 `setup_demo_player`（居中胶囊），`setup_player`（gltf，脚底原点）被注释。

### 2.1 插件装配（调度很关键，见第 5 节）

```rust
PhysicsPlugins::new(FixedPostUpdate);              // avian 物理步进
TnuaAvian3dPlugin::new(FixedUpdate);               // tnua 后端
TnuaControllerPlugin::<WaltzTnuaCtrlScheme>::new(FixedUpdate);
TnuaAirActionsPlugin::<WaltzAirActionSlots>::new(FixedUpdate);
```

全部在同一套固定步调度里（demo `ScheduleToUse::FixedUpdate` 的配置），这是 tnua 官方推荐的组合，**不要改成一个 Update 一个 FixedUpdate**。

---

## 3. 当前方案与参数

### 3.1 `WaltzTnuaCtrlScheme`

```rust
#[derive(TnuaScheme)]
#[scheme(basis = TnuaBuiltinWalk)]
pub enum WaltzTnuaCtrlScheme {
    Jump(TnuaBuiltinJump),
    Crouch(TnuaBuiltinCrouch),
    Dash(TnuaBuiltinDash),
    Knockback(TnuaBuiltinKnockback),
    WallSlide(TnuaBuiltinWallSlide, Entity),
    #[scheme(same_trigger(Jump))]
    WallJump(TnuaBuiltinJump),
    Climb(TnuaBuiltinClimb, Entity, Vector3),
}
```

动作槽：

```rust
#[derive(Debug, TnuaActionSlots)]
#[slots(scheme = WaltzTnuaCtrlScheme, ending(WallSlide, WallJump, Climb))]
pub struct WaltzAirActionSlots {
    #[slots(Jump)]
    jump: usize,
}
```

`ending(...)` = 这些动作发生即“结束计数”（爬墙/蹬墙跳会重置空中次数）。

### 3.2 Config 默认值（`WaltzTnuaCtrlSchemeConfig::default()`）

| 字段 | 值 |
|---|---|
| `basis.speed` | **未设置 = 默认 20.0** ⚠️ |
| `basis.float_height` | `0.01`（只适合脚底在原点模型）⚠️ |
| `basis.headroom` | 有，`distance_to_collider_top = 1.0` |
| `basis.max_slope` | `FRAC_PI_4` |
| `jump.height` | `4.0` |
| `crouch.float_offset` | `-0.9` |
| `dash.horizontal_distance` | `10.0` |
| `wall_slide.maintain_distance` | `0.7` |
| `wall_jump` | height 4.0、takeoff_extra_gravity 90.0、horizontal_distance 2.0 |
| `climb.climb_speed` | `10.0` |

角色侧 `CharacterMotionConfig { speed: 5.0 * 3.0 = 15.0, actions_in_air: 1, ... }`。

---

## 4. 正确的“喂入”模型

### 4.1 轮询式（tnua 推荐，官方 demo 写法）

```rust
// 每个需要控制角色的系统帧：
for (mut controller, input, config) in ... {
    controller.initiate_action_feeding();       // 1. 必须先调用一次
    controller.basis = TnuaBuiltinWalk {
        desired_motion: dir,                     // 2. 喂 basis（无输入喂 ZERO）
        desired_forward: Dir3::new(dir).ok(),    // 3. 可选转向
    };
    if jump_held {
        controller.action(ControlScheme::Jump(TnuaBuiltinJump {
            allow_in_air: air_actions.count_for(...) <= 1,
            ..Default::default()
        }));
    }
}
```

要点：
- **只要用 `action()` 就必须在同一帧先 `initiate_action_feeding()`**，否则 panic。
- Jump/Dash/Crouch 这类“按住有语义”的动作，要 **每帧持续喂**（不是只喂一次）。
- basis 没输入时要喂 `ZERO`，否则旧方向会被 basis 记忆住。

### 4.2 本项目当前做法（事件式，风险见 6.4）

`character_ctrl.rs` 用 enhanced input 的 `On<Fire<Jump>>` observer 喂 jump，`On<Fire<Move>>` 累加移动方向。`Fire` 的语义是“只要 action 处于 Fired 就每帧触发”（不是只触发一次），所以“按住跳”基本能被持续喂到；但：
- 喂 action 发生在 `PreUpdate`（enhanced input `add_input_context` 默认注册在 PreUpdate），而 `initiate_action_feeding()` 在 `Update` 的 `apply_tnua_ctrl` 里，二者跨帧、跨调度；
- 松开输入后没有值=0 的 `Fire`，导致 `last_move` 缓存永不归零（→ 角色不停）。

---

## 5. 调度与数据流

Bevy 0.19 一帧主顺序：

```
First → PreUpdate → RunFixedMainLoop(FixedFirst..FixedLast ×N) → Update → ... → Last
```

本项目配置下 tnua 的流向（每固定步）：

```
TnuaPipelineSystems（FixedUpdate 内，链式）
  Sensors（地面/头顶/障碍雷达，读上一次物理结果）
  → TnuaUserControlsSystems（喂入应发生的位置）
  → Logic（apply_controller_system：消费喂入、仲裁 action、驱动 motor）
  → Motors（把速度/角速度写给 avian）
随后同一固定步的 FixedPostUpdate → avian 真正解算。
```

- `TnuaControllerPlugin` 会把这些 set 链式配置在它注册的调度上（本项 = `FixedUpdate`）。
- **喂入最好放在同调度的 `TnuaUserControlsSystems`**，这样在固定步内排在 Sensors 之后、Logic 之前。
- demo（`../bevy-tnua/demos/src/bin/platformer_3d.rs`）在 FixedUpdate 配置下仍把控制系统加在 `Update` 的 `TnuaUserControlsSystems`，说明“在 Update 读输入、把组件喂好，等 FixedUpdate 来消费”是被支持的，但因此行为有一帧延迟/不等于“步内精确喂入”。

### 5.1 计数（Air Actions）

`TnuaAirActionsPlugin` 会在 `TnuaController<S>` 上自动挂 `TnuaActionsCounter<S>`（required component）。它在固定步内把“当前 basis 是否离地”和“新动作是否属于带槽动作”转成计数：

- 落地/地面时计数重置；
- 起跳本身是“计数前的动作”（号 0）；
- 空中再按 Jump 是号 1、号 2…
- `allow_in_air: count_for(Jump) <= actions_in_air` 即“地面跳 + N 次空中跳”。

---

## 6. 已知坑（症状 → 原因 → 修法）

### 6.1 🔴 速度被乘两遍
- 症状：角色移动极快/失控（期望 15，实际按 15×20≈300 量级）。
- 原因：`apply_tnua_ctrl` 喂 `desired_motion = direction * motion_config.speed`，但 basis config 的 `speed` 默认 `20.0`，`apply()` 会再乘一次（见 1.2）。
- 修法：
  - 方案 A：喂单位/≤1 的方向，把真实速度放进 `TnuaBuiltinWalkConfig { speed }`（推荐，和 demo 一致）；
  - 方案 B：保留 `dir*speed`，但把 basis config 的 `speed` 设 `1.0`。

### 6.2 🔴 松开按键角色不停
- 症状：按过一次方向后松开仍一直走。
- 原因：`AccumulatedInput.last_move` 只在 `Fire<Move>` 时更新，松开走 `Cancel/Complete`，无 0 值事件；`clear_accumulated_input` 未挂载。
- 修法：改成每帧轮询输入值（无输入喂 `ZERO`），或监听 `Cancel/Complete` 清缓存。见第 4 节推荐写法。

### 6.3 🟠 `desired_forward` 符号
- tnua 把角色 **-Z** 转到 `desired_forward`。
- 目前喂 `Dir3::new(-direction)`，而 demo 喂 `Dir3::new(direction)`。若模型正面是 -Z，倒着走；先确认模型正面朝向再定符号。shooter 一般希望面向相机前方，而不是移动方向。

### 6.4 🟠 事件 observer 喂 action 的脆弱性
- `action()` 断言要求本帧调过 `initiate_action_feeding()`。现在喂 action（PreUpdate）与 initiate（Update）不同帧/调度，靠上一帧遗留 flag 兜底；首帧/极端时序可能 panic，`count_for` 的空中判断也可能滞后一拍（计数在 FixedUpdate 才更新）。
- 建议收敛为“单个轮询系统放进 `TnuaUserControlsSystems`”。

### 6.5 🟠 `float_height = 0.01` vs 当前居中胶囊
- `setup_demo_player` 的 collider（`Collider::capsule(0.5, 1.0)`，avian 参数是 `radius, length`）底部在原点下 1.0；0.01 会让角色“被地面顶起”与弹簧打架、cast 距离（≈1.01）临界。
- 用居中胶囊时至少 `float_height ≈ 1.05`；换成脚底在原点模型再回到 0.01 附近。
- 另：视觉 `Capsule3d::new(0.5, 1.0)` 总高 3.0，collider 总高 2.0，不一致会导致视觉沉底 0.5。

### 6.6 🟡 僵尸 trait 实现
- `TnuaAirActionDefinition::is_air_action` 全 false：新计数 API 不读它（只被废弃的 `TnuaSimpleAirActionsCounter` 用），属死代码；
- `TnuaHasTargetEntity` 全 None：目前没人调用 `TnuaBlipReuseAvoidance::update`；以后启用爬墙/蹬墙 + blip reuse 时要对 `WallSlide`/`Climb` 返回 `Some(entity)`。
- 槽里没有 Dash：若启用 Dash 又不加槽，`count_for(Dash)` 恒 0 → 无限 dash。

### 6.7 🟡 未使用的组件/逻辑
- `TnuaGhostOverwrites`、`TnuaSimpleFallThroughPlatformsHelper` 已 insert 但没有任何 ghost platform/one-way 逻辑在喂（仓库关卡目前也没有 ghost platform）。无害但易误导。

---

## 7. 换控制插件的评估清单

### 7.1 先明确“我们到底用了 tnua 的哪些能力”

- 浮空式移动（弹簧、坡度、coyote time）
- 可变高度跳（按住更高）、空中动作计数（二段跳/多段）
- （计划中）爬墙/蹬墙跳/爬行、Dash、Crouch、one-way platform、ghost sensor
- 基于控制器状态的动画切换
- 与 `FixedUpdate` 确定性物理、`TnuaAvian3dSensorShape` 等集成细节

### 7.2 换插件前先回答的问题

1. 固定步调度（FixedUpdate + FixedPostUpdate）是否保留？候选插件是否支持同一定时？
2. 是否必须保留“浮空/弹簧”手感？Kinematic 控制器（更常见）与浮空是不同手感。
3. 空中动作计数、蹬墙/coyote 这些是否候选有现成 API，还是要自己拼？
4. 动画状态机是继续用 tnua 风格（每个动作一个 memory/state）还是换输入枚举？
5. 你正在评估哪些插件？把候选名列出来，我可以帮查 bevy 0.19/avian 0.7 兼容性并写对照。

### 7.3 可选方向（需要按候选版本自行验证）

- **保留 tnua**：按第 4/6 节修正后，问题基本都可控；改动量最小。
- **`bevy_character_controller`**（avian 生态，kinematic/hybrid）：面向 FPS/TPS 角色，社区活跃；但无浮空弹簧，跳跃/空中动作要自己实现。
- **自研轻量控制器 + avian 直接驱动**：如果最终需求只是“走路+跳”，写一套 velocity-based 可能比迁移插件更快。
- 其他 crate：先把名字给我，或去 docs.rs 看其 target bevy/avian 版本。

### 7.4 迁移备忘（概念映射）

| tnua | 替代/自研 |
|---|---|
| `TnuaScheme` | 枚举 + match |
| basis 每帧喂 + `float_height` 弹簧 | 每帧设置水平速度 + 地面探测决定是否施加重力 |
| `TnuaBuiltinJump` 状态机（Starting/Maintaining/Fall） | 自己维护 jump timer/velocity |
| `TnuaActionsCounter`/槽 | 自己记 `grounded`/`air_actions_used` |
| `TnuaControllerPlugin`/`TnuaAvian3dPlugin` | 一个自定义控制 system + avian 物理 |
| `TnuaAnimatingState` | `On<Timer>`/状态机 |

迁移前建议先把现有逻辑抽到一份“期望行为清单”（跳跃高度/时长、coyote、空中次数、爬墙判定），无论换什么都拿它当验收标准。

---

## 8. 参考

- 本地 tnua 源码：`../bevy-tnua`（`src/controller.rs`、`src/builtins/walk.rs`、`src/control_helpers/*`）
- 官方 demo（最接近本项目用途）：`../bevy-tnua/demos/src/bin/platformer_3d.rs`
- 控制/动画示例：`../bevy-tnua/demos/src/character_control_systems/platformer_control_systems.rs`、`platformer_control_scheme.rs`
- Avian collider 参数：`../avian/src/collision/collider/parry/mod.rs`（`capsule(radius, length)`）
- 调度顺序：`../bevy_engines/bevy_0.19/crates/bevy_app/src/main_schedule.rs`
- enhanced input `Fire` 语义：`../bevy_enhanced_input`（git tag `v0.26.0`，`src/action/events.rs`）

> 文档记录日期：本次评审（含 6.1–6.6 结论）。若后续改动 scheme/config/调度，请同步更新第 3、5 节。
