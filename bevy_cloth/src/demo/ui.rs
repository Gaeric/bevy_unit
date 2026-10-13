//! m4 debug panel, built with the engine's `bevy_feathers` widgets (m4-feathers-plan.org).
//!
//! `bevy_ui_render` draws ui for both `Camera2d` and `Camera3d`, so the panel overlays the 3d
//! scene without spawning a second camera.

use bevy::{
    feathers::{
        FeathersPlugins,
        controls::{FeathersCheckbox, FeathersSlider},
        dark_theme::create_dark_theme,
        display::{caption, label},
        theme::{ThemeBackgroundColor, UiTheme},
        tokens,
    },
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
    ui::Checked,
    ui_widgets::{SliderPrecision, SliderStep, SliderValue, ValueChange, checkbox_self_update},
};

use crate::{
    demo::{
        plugin::{SimClock, SimParamsRes, SimStateRes, SolverRes, StatsRes},
        scenes::{self, CurrentScene},
    },
    sim::params::SimParams,
};

/// the telemetry readout the panel keeps updated.
#[derive(Component, Clone, Default)]
struct StatsText;

/// the parameter list; "hide gui" collapses just this part so the toggles stay reachable.
#[derive(Component, Clone, Default)]
struct ParamList;

/// the panel's root node. the whole panel is rebuilt whenever the scene changes, so the widgets
/// pick up the new scene's parameter overrides (the source's ImGui widgets bind the values
/// directly; feathers widgets own their own state).
#[derive(Component, Clone, Default)]
struct PanelRoot;

/// the `SimState` / `SimParams` flag a toggle checkbox mirrors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToggleFlag {
    #[default]
    Pause,
    HideGui,
    Wireframe,
    DrawParticles,
    SelfCollision,
}

/// tags a `FeathersCheckbox` with the flag it drives, so `sync_toggle_checked` can push keyboard
/// changes back into the widget (`input.rs`).
///
/// `bsn!` builds components from `Default` + `Clone`, so the marker derives `Default` even though
/// every spawn site passes a flag.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToggleKind(pub ToggleFlag);

/// feathers ui: theme, panel widgets, and the per-frame readouts.
pub struct ClothUiPlugin;

impl Plugin for ClothUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((FeathersPlugins, WireframePlugin::default()))
            .insert_resource(UiTheme(create_dark_theme()))
            // keeps each checkbox's own `Checked` state in sync with its `ValueChange`
            .add_observer(checkbox_self_update)
            // the panel reads the current scene's parameters, so build it after the scene writes them
            .add_systems(Startup, spawn_panel.after(scenes::setup_initial_scene))
            .add_systems(
                Update,
                (
                    update_stats_text,
                    apply_hide_gui,
                    apply_wireframe,
                    draw_particles,
                    sync_toggle_checked,
                    // `apply_scene_switch` writes the new scene's parameters; rebuilding before it
                    // runs would show the outgoing scene's values
                    rebuild_panel_on_scene_change.after(scenes::apply_scene_switch),
                ),
            );
    }
}

fn spawn_panel(mut commands: Commands, params: Res<SimParamsRes>) {
    build_panel(&mut commands, &params.0);
}

/// a scene switch replaces the parameters, so the panel is thrown away and rebuilt from the current
/// values.
fn rebuild_panel_on_scene_change(
    mut first_run: Local<bool>,
    current: Res<CurrentScene>,
    params: Res<SimParamsRes>,
    roots: Query<Entity, With<PanelRoot>>,
    mut commands: Commands,
) {
    // `CurrentScene` is *added* at startup, so `is_changed()` is true on the first `Update` even
    // though the scene never switched. `spawn_panel` already built the panel from this scene, so
    // rebuilding here would despawn that fresh panel while `sync_toggle_checked` still holds its
    // checkboxes and panic on the stale entities.
    if !std::mem::replace(&mut *first_run, true) {
        return;
    }

    if !current.is_changed() {
        return;
    }

    for entity in &roots {
        commands.entity(entity).despawn();
    }
    build_panel(&mut commands, &params.0);
}

fn build_panel(commands: &mut Commands, p: &SimParams) {
    // slider positions are captured once at spawn; the widgets own their value afterwards
    let substeps = p.num_substeps as f32;
    let iterations = p.num_iterations as f32;
    let interleaved = p.interleaved_hash as f32;
    let max_speed = p.max_speed;
    let damping = p.damping;
    let relaxation = p.relaxation_factor;
    let stretchiness = p.long_range_stretchiness;
    let margin = p.collision_margin;
    let friction = p.friction;
    // the source labels this slider logarithmic; 0..100 maps onto 0..99 here
    let bend = 50.0 * (p.bend_compliance + 1.0).log10();
    // `bsn!` values must be plain idents, so the gravity axes are bound individually
    let gravity_x = p.gravity.x;
    let gravity_y = p.gravity.y;
    let gravity_z = p.gravity.z;

    commands.spawn_scene(bsn! {
        Node {
            position_type: PositionType::Absolute,
            left: px(10),
            top: px(10),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            padding: UiRect::all(px(10)),
            width: px(320),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        PanelRoot
        Children [
            @label("cloth sim")
            --
            @caption("") StatsText
            --
            @FeathersCheckbox { @caption: bsn! { @caption("pause") } }
            ToggleKind(ToggleFlag::Pause)
            on(|change: On<ValueChange<bool>>, mut state: ResMut<SimStateRes>| {
                state.0.pause = change.value;
            })
            --
            @FeathersCheckbox { @caption: bsn! { @caption("hide gui") } }
            ToggleKind(ToggleFlag::HideGui)
            on(|change: On<ValueChange<bool>>, mut state: ResMut<SimStateRes>| {
                state.0.hide_gui = change.value;
            })
            --
            @FeathersCheckbox { @caption: bsn! { @caption("wireframe") } }
            ToggleKind(ToggleFlag::Wireframe)
            on(|change: On<ValueChange<bool>>, mut state: ResMut<SimStateRes>| {
                state.0.render_wireframe = change.value;
            })
            --
            @FeathersCheckbox { @caption: bsn! { @caption("draw particles") } }
            ToggleKind(ToggleFlag::DrawParticles)
            on(|change: On<ValueChange<bool>>, mut state: ResMut<SimStateRes>| {
                state.0.draw_particles = change.value;
            })
            --
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: px(4),
            }
            ParamList
            Children [
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("substeps")
                    --
                    @FeathersSlider { @max: 20.0, @min: 1.0 }
                    SliderValue(substeps)
                    SliderStep(1.0)
                    SliderPrecision(0)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.num_substeps = change.value.round() as u32;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("iterations")
                    --
                    @FeathersSlider { @max: 20.0, @min: 1.0 }
                    SliderValue(iterations)
                    SliderStep(1.0)
                    SliderPrecision(0)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.num_iterations = change.value.round() as u32;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("interleaved")
                    --
                    @FeathersSlider { @max: 10.0, @min: 1.0 }
                    SliderValue(interleaved)
                    SliderStep(1.0)
                    SliderPrecision(0)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.interleaved_hash = change.value.round() as u32;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("max speed")
                    --
                    @FeathersSlider { @max: 100.0, @min: 0.01 }
                    SliderValue(max_speed)
                    SliderPrecision(2)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.max_speed = change.value;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("damping")
                    --
                    @FeathersSlider { @max: 10.0, @min: 0.0 }
                    SliderValue(damping)
                    SliderPrecision(3)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.damping = change.value;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("relaxation")
                    --
                    @FeathersSlider { @max: 3.0, @min: 0.0 }
                    SliderValue(relaxation)
                    SliderPrecision(3)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.relaxation_factor = change.value;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("stretchiness")
                    --
                    @FeathersSlider { @max: 2.0, @min: 1.0 }
                    SliderValue(stretchiness)
                    SliderPrecision(3)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.long_range_stretchiness = change.value;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("margin")
                    --
                    @FeathersSlider { @max: 0.5, @min: 0.0 }
                    SliderValue(margin)
                    SliderPrecision(3)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.collision_margin = change.value;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("friction")
                    --
                    @FeathersSlider { @max: 1.0, @min: 0.0 }
                    SliderValue(friction)
                    SliderPrecision(3)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.friction = change.value;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("bend")
                    --
                    @FeathersSlider { @max: 100.0, @min: 0.0 }
                    SliderValue(bend)
                    SliderPrecision(1)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.bend_compliance = 10f32.powf(change.value / 50.0) - 1.0;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("gravity x")
                    --
                    @FeathersSlider { @max: 50.0, @min: -50.0 }
                    SliderValue(gravity_x)
                    SliderPrecision(2)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.gravity.x = change.value;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("gravity y")
                    --
                    @FeathersSlider { @max: 50.0, @min: -50.0 }
                    SliderValue(gravity_y)
                    SliderPrecision(2)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.gravity.y = change.value;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                Node { align_items: AlignItems::Center, column_gap: px(6) }
                Children [
                    @label("gravity z")
                    --
                    @FeathersSlider { @max: 50.0, @min: -50.0 }
                    SliderValue(gravity_z)
                    SliderPrecision(2)
                    on(|change: On<ValueChange<f32>>, mut params: ResMut<SimParamsRes>, mut commands: Commands| {
                        params.0.gravity.z = change.value;
                        commands.entity(change.source).insert(SliderValue(change.value));
                    })
                ]
                --
                @FeathersCheckbox { @caption: bsn! { @caption("self collision") } }
                ToggleKind(ToggleFlag::SelfCollision)
                on(|change: On<ValueChange<bool>>, mut params: ResMut<SimParamsRes>| {
                    params.0.enable_self_collision = change.value;
                })
            ]
        ]
    });
}

fn update_stats_text(
    solver: Res<SolverRes>,
    clock: Res<SimClock>,
    stats: Res<StatsRes>,
    mut text: Single<&mut Text, With<StatsText>>,
) {
    text.0 = format!(
        "particles {}\nt {:.2}s   step {:.2}ms\nmax|v| {:.2}   clamp {}\nstretch max {:.2e}\navg {:.2e}   pairs {}",
        solver.0.num_particles(),
        clock.0.seconds(),
        stats.0.step_time.as_secs_f64() * 1000.0,
        stats.0.max_speed,
        stats.0.clamped,
        stats.0.max_stretch_error,
        stats.0.avg_stretch_error,
        stats.0.self_collision_candidates,
    );
}

fn apply_hide_gui(state: Res<SimStateRes>, mut list: Query<&mut Node, With<ParamList>>) {
    let display = if state.0.hide_gui {
        Display::None
    } else {
        Display::Flex
    };

    for mut node in &mut list {
        if node.display != display {
            node.display = display;
        }
    }
}

fn apply_wireframe(state: Res<SimStateRes>, mut config: ResMut<WireframeConfig>) {
    if config.global != state.0.render_wireframe {
        config.global = state.0.render_wireframe;
    }
}

/// pushes `SimState` / `SimParams` back into the toggle widgets.
///
/// clicking a checkbox keeps `Checked` in sync through `checkbox_self_update`, but a keyboard
/// toggle (`input.rs`) only writes the resource, so the box has to be corrected here.
fn sync_toggle_checked(
    state: Res<SimStateRes>,
    params: Res<SimParamsRes>,
    mut commands: Commands,
    toggles: Query<(Entity, &ToggleKind, Has<Checked>)>,
) {
    for (entity, kind, checked) in &toggles {
        let want = match kind.0 {
            ToggleFlag::Pause => state.0.pause,
            ToggleFlag::HideGui => state.0.hide_gui,
            ToggleFlag::Wireframe => state.0.render_wireframe,
            ToggleFlag::DrawParticles => state.0.draw_particles,
            ToggleFlag::SelfCollision => params.0.enable_self_collision,
        };

        if want == checked {
            continue;
        }

        // `Checked` is a presence component: insert to tick, remove to clear
        if want {
            commands.entity(entity).insert(Checked);
        } else {
            commands.entity(entity).remove::<Checked>();
        }
    }
}

/// `VtGameState::drawParticles`: a small cross per particle, drawn with gizmos.
fn draw_particles(state: Res<SimStateRes>, solver: Res<SolverRes>, mut gizmos: Gizmos) {
    if !state.0.draw_particles {
        return;
    }

    const R: f32 = 0.012;
    let color = Color::srgb(1.0, 0.45, 0.1);

    for position in &solver.0.positions {
        gizmos.line(*position - Vec3::X * R, *position + Vec3::X * R, color);
        gizmos.line(*position - Vec3::Y * R, *position + Vec3::Y * R, color);
        gizmos.line(*position - Vec3::Z * R, *position + Vec3::Z * R, color);
    }
}
