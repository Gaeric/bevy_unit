//! Solver selector for the Attach example.

use bevy::{
    picking::{Pickable, hover::Hovered},
    prelude::*,
    ui_widgets::{Activate, Button},
};
use bevy_cloth::ClothSolver;

use super::Demo;

#[derive(Component)]
pub(super) struct Hud;

#[derive(Clone, Copy, Component)]
pub(super) struct Choice(ClothSolver);

pub(super) fn spawn(mut commands: Commands) {
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
        panel
            .spawn(Node {
                flex_wrap: FlexWrap::Wrap,
                align_items: AlignItems::Center,
                column_gap: px(6),
                row_gap: px(4),
                ..default()
            })
            .with_children(|row| {
                row.spawn((
                    Text::new("Solver"),
                    TextFont::from_font_size(14.0),
                    Node {
                        width: px(50),
                        ..default()
                    },
                ));
                for (label, solver) in [
                    ("Colored GS", ClothSolver::ColoredGs),
                    ("Jacobi", ClothSolver::Jacobi),
                    ("VBD", ClothSolver::Vbd),
                ] {
                    row.spawn((
                        Button,
                        Choice(solver),
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
    if choice.0 == demo.solver {
        return;
    }
    demo.solver = choice.0;
    demo.reset();
}

pub(super) fn highlight_selection(
    demo: Res<Demo>,
    mut buttons: Query<(&Choice, &Hovered, &mut BackgroundColor)>,
) {
    for (choice, hovered, mut background) in &mut buttons {
        let color = if choice.0 == demo.solver {
            Color::srgb(0.10, 0.38, 0.65)
        } else if hovered.get() {
            Color::srgb(0.26, 0.32, 0.40)
        } else {
            Color::srgb(0.14, 0.19, 0.25)
        };
        background.set_if_neq(BackgroundColor(color));
    }
}
