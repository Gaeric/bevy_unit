//! Scene, solver, step-budget and GPU timing controls for the GLB example.

use bevy::{
    picking::{Pickable, hover::Hovered},
    prelude::*,
    ui_widgets::{
        Activate, Button, Slider, SliderPrecision, SliderRange, SliderThumb, SliderValue,
        TrackClick, ValueChange,
    },
};
use bevy_cloth::{
    ClothSolver,
    debug::timing::{ClothGpuTimings, TimingSnapshot, TimingStatus},
    legacy::demo::LegacySceneId,
};
use std::fmt::Write as _;

use super::{Demo, SceneReady, StepBudget};

const MAX_STEP_COUNT: u32 = 20;
const SLIDER_THUMB_SIZE: f32 = 12.0;
const TIMING_OFF: &str = "GPU Timing: Off";
const TIMING_ON: &str = "GPU Timing: On";

#[derive(Component)]
pub(super) struct Hud;

#[derive(Component)]
pub(super) struct TimingReadout;

#[derive(Component)]
pub(super) struct TimingToggleLabel;

#[derive(Clone, Copy, Component)]
pub(super) enum TimingAction {
    Toggle,
    ResetPeak,
}

#[derive(Component)]
pub(super) struct VbdOptionRow;

#[derive(Clone, Copy)]
enum BudgetField {
    Substeps,
    Sweeps,
}

impl BudgetField {
    fn label(self) -> &'static str {
        match self {
            Self::Substeps => "Substeps / tick",
            Self::Sweeps => "Sweeps / substep",
        }
    }

    fn value(self, budget: StepBudget) -> u32 {
        match self {
            Self::Substeps => budget.substeps,
            Self::Sweeps => budget.sweeps,
        }
    }
}

#[derive(Component)]
pub(super) struct BudgetSlider {
    field: BudgetField,
    label: Entity,
    thumb: Entity,
}

#[derive(Component)]
pub(super) struct ApplyBudget;

#[derive(Clone, Copy, Component)]
pub(super) enum Choice {
    Scene(LegacySceneId),
    Solver(ClothSolver),
    VbdFriction(bool),
    VbdSelfContact(bool),
}

impl Choice {
    fn is_selected(self, demo: &Demo) -> bool {
        match self {
            Self::Scene(scene) => demo.scene == scene,
            Self::Solver(solver) => demo.solver == solver,
            Self::VbdFriction(enabled) => demo.vbd_options.friction == enabled,
            Self::VbdSelfContact(enabled) => demo.vbd_options.self_contact == enabled,
        }
    }
}

pub(super) fn spawn(mut commands: Commands, demo: Res<Demo>) {
    use LegacySceneId::*;

    let mut panel = commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: px(10),
            left: px(10),
            right: px(10),
            padding: UiRect::all(px(10)),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            border_radius: BorderRadius::all(px(8)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.04, 0.06, 0.09, 0.94)),
    ));
    panel.with_children(|panel| {
        for (heading, choices, vbd_only) in [
            (
                "Scene",
                &[
                    ("Attach", Choice::Scene(Attach)),
                    ("Collision", Choice::Scene(Collision)),
                    ("Self Collision", Choice::Scene(SelfCollision)),
                    ("Friction", Choice::Scene(Friction)),
                    ("Multiple", Choice::Scene(Multiple)),
                    ("High Resolution", Choice::Scene(HighResolution)),
                    ("Swirl", Choice::Scene(Swirl)),
                ][..],
                false,
            ),
            (
                "Solver",
                &[
                    ("Colored GS", Choice::Solver(ClothSolver::ColoredGs)),
                    ("Jacobi", Choice::Solver(ClothSolver::Jacobi)),
                    ("VBD", Choice::Solver(ClothSolver::Vbd)),
                ][..],
                false,
            ),
            (
                "VBD friction",
                &[
                    ("Off", Choice::VbdFriction(false)),
                    ("On", Choice::VbdFriction(true)),
                ][..],
                true,
            ),
            (
                "VBD self-contact",
                &[
                    ("Off", Choice::VbdSelfContact(false)),
                    ("On", Choice::VbdSelfContact(true)),
                ][..],
                true,
            ),
        ] {
            let mut row = panel.spawn(Node {
                display: if vbd_only {
                    Display::None
                } else {
                    Display::Flex
                },
                flex_wrap: FlexWrap::Wrap,
                align_items: AlignItems::Center,
                column_gap: px(6),
                row_gap: px(4),
                ..default()
            });
            if vbd_only {
                row.insert(VbdOptionRow);
            }
            row.with_children(|row| {
                row.spawn((
                    Text::new(heading),
                    TextFont::from_font_size(14.0),
                    Node {
                        width: px(if vbd_only { 120 } else { 50 }),
                        ..default()
                    },
                ));
                for &(label, choice) in choices {
                    row.spawn((
                        Button,
                        choice,
                        Hovered::default(),
                        Node {
                            padding: UiRect::axes(px(10), px(5)),
                            border_radius: BorderRadius::all(px(4)),
                            ..default()
                        },
                        BackgroundColor::default(),
                        children![(
                            Text::new(label),
                            TextFont::from_font_size(14.0),
                            Pickable::IGNORE
                        )],
                    ))
                    .observe(select_choice);
                }
            });
        }
        spawn_budget_controls(panel, demo.pending_budget);
        panel.spawn((
            Hud,
            Text::new("Loading GLBs"),
            TextFont::from_font_size(14.0),
        ));
    });
}

fn spawn_budget_controls(panel: &mut ChildSpawnerCommands, budget: StepBudget) {
    panel
        .spawn(Node {
            flex_wrap: FlexWrap::Wrap,
            align_items: AlignItems::Center,
            column_gap: px(12),
            row_gap: px(4),
            ..default()
        })
        .with_children(|row| {
            for field in [BudgetField::Substeps, BudgetField::Sweeps] {
                spawn_budget_slider(row, field, field.value(budget));
            }
            row.spawn((
                Button,
                ApplyBudget,
                Hovered::default(),
                Node {
                    padding: UiRect::axes(px(10), px(5)),
                    border_radius: BorderRadius::all(px(4)),
                    ..default()
                },
                button_color(false, false),
                children![(
                    Text::new("Apply"),
                    TextFont::from_font_size(14.0),
                    Pickable::IGNORE,
                )],
            ))
            .observe(apply_budget);
        });
}

fn spawn_budget_slider(row: &mut ChildSpawnerCommands, field: BudgetField, count: u32) {
    row.spawn(Node {
        align_items: AlignItems::Center,
        column_gap: px(6),
        ..default()
    })
    .with_children(|group| {
        let label = group
            .spawn((
                Text::new(format!("{}: {count}", field.label())),
                TextFont::from_font_size(14.0),
            ))
            .id();
        let range = SliderRange::new(1.0, MAX_STEP_COUNT as f32);
        let mut slider = group.spawn((
            Slider {
                track_click: TrackClick::Snap,
                ..default()
            },
            SliderValue(count as f32),
            range,
            SliderPrecision(0),
            Node {
                width: px(140),
                height: px(18),
                border_radius: BorderRadius::all(px(4)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.14, 0.19, 0.25)),
        ));
        let mut thumb = Entity::PLACEHOLDER;
        // Position the thumb over the same travel distance the widget uses for dragging.
        slider.with_children(|track| {
            track
                .spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(0),
                        right: px(SLIDER_THUMB_SIZE),
                        top: px(0),
                        bottom: px(0),
                        ..default()
                    },
                    Pickable::IGNORE,
                ))
                .with_children(|travel| {
                    thumb = travel
                        .spawn((
                            SliderThumb,
                            Node {
                                position_type: PositionType::Absolute,
                                left: percent(range.thumb_position(count as f32) * 100.0),
                                width: px(SLIDER_THUMB_SIZE),
                                height: px(18),
                                border_radius: BorderRadius::all(px(4)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.28, 0.64, 0.95)),
                        ))
                        .id();
                });
        });
        slider
            .insert(BudgetSlider {
                field,
                label,
                thumb,
            })
            .observe(edit_budget);
    });
}

fn edit_budget(
    event: On<ValueChange<f32>>,
    sliders: Query<(&BudgetSlider, &SliderRange)>,
    mut demo: ResMut<Demo>,
) {
    let Ok((slider, range)) = sliders.get(event.source) else {
        return;
    };
    if !event.value.is_finite() {
        return;
    }
    let count = range.clamp(event.value.round()) as u32;
    match slider.field {
        BudgetField::Substeps => demo.pending_budget.substeps = count,
        BudgetField::Sweeps => demo.pending_budget.sweeps = count,
    }
}

fn apply_budget(_: On<Activate>, mut demo: ResMut<Demo>, mut timings: ResMut<ClothGpuTimings>) {
    if demo.budget != demo.pending_budget {
        demo.budget = demo.pending_budget;
        demo.reset(&mut timings);
    }
}

pub(super) fn update_budget_controls(
    mut commands: Commands,
    demo: Res<Demo>,
    sliders: Query<(Entity, &BudgetSlider, &SliderRange, &SliderValue)>,
    mut labels: Query<&mut Text>,
    mut thumbs: Query<&mut Node, With<SliderThumb>>,
    mut apply: Single<(&Hovered, &mut BackgroundColor), With<ApplyBudget>>,
) {
    for (entity, slider, range, value) in &sliders {
        let count = slider.field.value(demo.pending_budget);
        if value.0 == count as f32 {
            continue;
        }
        commands.entity(entity).insert(SliderValue(count as f32));
        if let Ok(mut label) = labels.get_mut(slider.label) {
            label.0 = format!("{}: {count}", slider.field.label());
        }
        if let Ok(mut thumb) = thumbs.get_mut(slider.thumb) {
            thumb.left = percent(range.thumb_position(count as f32) * 100.0);
        }
    }
    let (hovered, background) = &mut *apply;
    background.set_if_neq(button_color(
        demo.budget != demo.pending_budget,
        hovered.get(),
    ));
}

fn select_choice(
    event: On<Activate>,
    choices: Query<&Choice>,
    mut demo: ResMut<Demo>,
    mut timings: ResMut<ClothGpuTimings>,
) {
    let Ok(&choice) = choices.get(event.entity) else {
        return;
    };
    if choice.is_selected(&demo) {
        return;
    }
    match choice {
        Choice::Scene(scene) => {
            demo.scene = scene;
            demo.budget = StepBudget::for_scene(scene);
        }
        Choice::Solver(solver) => demo.solver = solver,
        Choice::VbdFriction(enabled) if demo.solver == ClothSolver::Vbd => {
            demo.vbd_options.friction = enabled;
        }
        Choice::VbdSelfContact(enabled) if demo.solver == ClothSolver::Vbd => {
            demo.vbd_options.self_contact = enabled;
        }
        Choice::VbdFriction(_) | Choice::VbdSelfContact(_) => return,
    }
    demo.reset(&mut timings);
}

pub(super) fn highlight_selection(
    demo: Res<Demo>,
    mut option_rows: Query<&mut Node, With<VbdOptionRow>>,
    mut buttons: Query<(&Choice, &Hovered, &mut BackgroundColor)>,
) {
    let display = if demo.solver == ClothSolver::Vbd {
        Display::Flex
    } else {
        Display::None
    };
    for mut row in &mut option_rows {
        if row.display != display {
            row.display = display;
        }
    }
    for (choice, hovered, mut background) in &mut buttons {
        background.set_if_neq(button_color(choice.is_selected(&demo), hovered.get()));
    }
}

fn button_color(selected: bool, hovered: bool) -> BackgroundColor {
    BackgroundColor(if selected {
        Color::srgb(0.10, 0.38, 0.65)
    } else if hovered {
        Color::srgb(0.26, 0.32, 0.40)
    } else {
        Color::srgb(0.14, 0.19, 0.25)
    })
}

pub(super) fn spawn_timing_panel(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: px(10),
                left: px(10),
                max_width: percent(95),
                padding: UiRect::all(px(8)),
                flex_direction: FlexDirection::Column,
                row_gap: px(4),
                border_radius: BorderRadius::all(px(8)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.04, 0.06, 0.09, 0.94)),
        ))
        .with_children(|panel| {
            panel
                .spawn(Node {
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: px(6),
                    row_gap: px(4),
                    ..default()
                })
                .with_children(|row| {
                    for (action, label) in [
                        (TimingAction::Toggle, TIMING_OFF),
                        (TimingAction::ResetPeak, "Reset Peak"),
                    ] {
                        row.spawn((
                            Button,
                            action,
                            Hovered::default(),
                            Node {
                                padding: UiRect::axes(px(10), px(5)),
                                border_radius: BorderRadius::all(px(4)),
                                ..default()
                            },
                            button_color(false, false),
                        ))
                        .with_children(|button| {
                            let mut label = button.spawn((
                                Text::new(label),
                                TextFont::from_font_size(14.0),
                                Pickable::IGNORE,
                            ));
                            if matches!(action, TimingAction::Toggle) {
                                label.insert(TimingToggleLabel);
                            }
                        })
                        .observe(apply_timing_action);
                    }
                });
            panel.spawn((
                TimingReadout,
                Text::default(),
                TextFont::from_font_size(14.0),
                Pickable::IGNORE,
                Node {
                    display: Display::None,
                    ..default()
                },
            ));
        });
}

fn apply_timing_action(
    event: On<Activate>,
    actions: Query<&TimingAction>,
    mut demo: ResMut<Demo>,
    scene: Option<Res<SceneReady>>,
    mut timings: ResMut<ClothGpuTimings>,
) {
    let Ok(action) = actions.get(event.entity) else {
        return;
    };
    match action {
        TimingAction::Toggle => {
            demo.timing_enabled = !demo.timing_enabled;
            if demo.timing_enabled && !demo.rebuild {
                timings.set_target(scene.map(|scene| scene.cloth));
            } else {
                timings.set_target(None);
            }
        }
        TimingAction::ResetPeak if timings.target().is_some() => timings.reset_peak(),
        TimingAction::ResetPeak => (),
    }
}

pub(super) fn update_timings(
    demo: Res<Demo>,
    timings: Res<ClothGpuTimings>,
    mut toggle: Single<&mut Text, (With<TimingToggleLabel>, Without<TimingReadout>)>,
    mut readout: Single<(&mut Text, &mut Node), With<TimingReadout>>,
    mut buttons: Query<(&TimingAction, &Hovered, &mut BackgroundColor)>,
) {
    let (text, node) = &mut *readout;
    let (label, display, value) = if demo.timing_enabled {
        (TIMING_ON, Display::Flex, timing_text(&timings.snapshot()))
    } else {
        (TIMING_OFF, Display::None, String::new())
    };
    if toggle.0 != label {
        toggle.0 = label.to_owned();
    }
    if node.display != display {
        node.display = display;
    }
    text.set_if_neq(Text::new(value));
    for (action, hovered, mut background) in &mut buttons {
        let selected = matches!(action, TimingAction::Toggle) && demo.timing_enabled;
        background.set_if_neq(button_color(selected, hovered.get()));
    }
}

fn timing_text(snapshot: &TimingSnapshot) -> String {
    let status = match snapshot.status {
        TimingStatus::Off => return "GPU timing pending: waiting for cloth".to_owned(),
        TimingStatus::Pending => "GPU timing pending: waiting for a timed tick",
        TimingStatus::Unsupported => return "GPU timing unavailable on this device".to_owned(),
        TimingStatus::TotalsOnly => "GPU timing: tick totals only",
        TimingStatus::Detailed => "GPU timing: VBD stage detail at peak",
        TimingStatus::OutputOnly => "GPU timing: output only (tick timing unavailable)",
    };
    let mut text = status.to_owned();
    if snapshot.status != TimingStatus::OutputOnly {
        let mean = snapshot
            .mean_tick_ms
            .map_or_else(|| "—".to_owned(), |ms| format!("{ms:.3} ms"));
        let max = snapshot.peak.as_ref().map_or_else(
            || "—".to_owned(),
            |peak| format!("{:.3} ms (global tick {})", peak.milliseconds, peak.tick),
        );
        let _ = write!(
            text,
            "\nMean: {mean}   Max: {max}\nSamples: {}   Skipped: {}",
            snapshot.samples, snapshot.skipped
        );
        if let Some(peak) = &snapshot.peak {
            for stage in &peak.stages {
                let _ = write!(
                    text,
                    "\n  {}: {:.3} ms ({} calls)",
                    stage.stage.label(),
                    stage.milliseconds,
                    stage.calls
                );
            }
        }
    }
    let output = snapshot
        .output_ms
        .map_or_else(|| "pending".to_owned(), |ms| format!("{ms:.3} ms/frame"));
    let _ = write!(text, "\nLatest mesh output: {output}");
    text
}
