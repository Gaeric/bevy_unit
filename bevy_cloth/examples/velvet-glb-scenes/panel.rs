//! Scene and solver selectors for the GLB example.

use bevy::{
    picking::{Pickable, hover::Hovered},
    prelude::*,
    ui_widgets::{Activate, Button},
};
use bevy_cloth::{ClothSolver, legacy::demo::LegacySceneId};

use super::Demo;

#[derive(Component)]
pub(super) struct Hud;

#[derive(Component)]
pub(super) struct VbdOptionRow;

#[derive(Clone, Copy, Component)]
pub(super) enum Choice {
    Scene(LegacySceneId),
    Solver(ClothSolver),
    VbdFriction(bool),
}

impl Choice {
    fn is_selected(self, demo: &Demo) -> bool {
        match self {
            Self::Scene(scene) => demo.scene == scene,
            Self::Solver(solver) => demo.solver == solver,
            Self::VbdFriction(enabled) => demo.vbd_friction == enabled,
        }
    }
}

pub(super) fn spawn(mut commands: Commands) {
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
        panel.spawn((
            Hud,
            Text::new("Loading GLBs"),
            TextFont::from_font_size(14.0),
        ));
    });
}

fn select_choice(event: On<Activate>, choices: Query<&Choice>, mut demo: ResMut<Demo>) {
    let Ok(&choice) = choices.get(event.entity) else {
        return;
    };
    if choice.is_selected(&demo) {
        return;
    }
    match choice {
        Choice::Scene(scene) => demo.scene = scene,
        Choice::Solver(solver) => demo.solver = solver,
        Choice::VbdFriction(enabled) if demo.solver == ClothSolver::Vbd => {
            demo.vbd_friction = enabled;
        }
        Choice::VbdFriction(_) => return,
    }
    demo.reset();
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
        let color = if choice.is_selected(&demo) {
            Color::srgb(0.10, 0.38, 0.65)
        } else if hovered.get() {
            Color::srgb(0.26, 0.32, 0.40)
        } else {
            Color::srgb(0.14, 0.19, 0.25)
        };
        background.set_if_neq(BackgroundColor(color));
    }
}
