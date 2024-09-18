use bevy::{ecs::system::EntityCommands, prelude::*};
use bevy_egui::egui::widgets;

#[derive(Component)]
pub struct UiCamera;

#[derive(Clone, Copy, Component)]
pub struct RadioButton;

#[derive(Clone, Copy, Component)]
pub struct RadioButtonText;

#[derive(Clone, Component, Deref, DerefMut)]
pub struct WidgetClickSender<T>(T)
where
    T: Clone + Send + Sync + 'static;

/// The type of light presently in the scene: directional, point, or spot.
#[derive(Clone, Copy, Default, PartialEq)]
enum LightType {
    /// A directional light, with a cascaded shadow map.
    #[default]
    Directional,
    /// A point light, with a cube shadow map.
    Point,
    /// A spot light, with a cube shadow map.
    Spot,
}

/// The type of shadow filter.
///
/// Generally, `Gaussian` is preferred when temporal antialiasing isn't in use,
/// while `Temporal` is preferred when TAA is in use. In this example, this
/// setting also turns TAA on and off.
#[derive(Clone, Copy, Default, PartialEq)]
enum ShadowFilter {
    /// The non-temporal Gaussian filter (Castano '13 for directional lights, an
    /// analogous alternative for point and spot lights).
    #[default]
    NonTemporal,
    /// The temporal Gaussian filter (Jimenez '14 for directional lights, an
    /// analogous alternative for point and spot lights).
    Temporal,
}

/// Each example setting that can be toggled in the UI.
#[derive(Clone, Copy, PartialEq)]
enum AppSetting {
    /// The type of light presently in the scene: directional, point, or spot.
    LightType(LightType),
    /// The type of shadow filter.
    ShadowFilter(ShadowFilter),
    /// Whether PCSS is enabled or disabled.
    SoftShadows(bool),
}

pub(crate) fn plugin(app: &mut App) {
    app.add_systems(Startup, setup);
}

fn setup(mut commands: Commands) {
    // let main_camera = commands
    //     .spawn((
    //         Camera3dBundle {
    //             camera: Camera {
    //                 order: 1,
    //                 clear_color: Color::BLACK.into(),
    //                 ..default()
    //             },
    //             transform: Transform::from_translation(Vec3::new(0., 30., 0.))
    //                 .looking_at(Vec3::ZERO, Vec3::Y),
    //             ..default()
    //         },
    //         UiCamera,
    //     ))
    //     .id();

    commands.spawn(Camera2dBundle::default());

    // commands.ui_builder(UiRoot).column(|column| {
    //     column.spawn(NodeBundle {
    //         style: Style {
    //             height: Val::Percent(100.),
    //             flex_direction: FlexDirection::Column,
    //             ..default()
    //         },
    //         background_color: Color::WHITE.into(),
    //         ..default()
    //     });
    // });
    spawn_box(&mut commands);
    spawn_buttons(&mut commands);
}

fn spawn_box(commands: &mut Commands) {
    let container = NodeBundle {
        style: Style {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            ..default()
        },
        ..default()
    };

    let square = NodeBundle {
        style: Style {
            width: Val::Px(20.0),
            border: UiRect::all(Val::Px(2.)),
            left: Val::Px(2.0),
            ..default()
        },
        background_color: Color::srgba(0.65, 0.65, 0.65, 0.50).into(),
        ..default()
    };

    let parent = commands.spawn(container).id();
    let child = commands.spawn(square).id();

    commands.entity(parent).push_children(&[child]);
}

fn spawn_buttons(commands: &mut Commands) {
    commands
        .spawn(NodeBundle {
            style: Style {
                flex_direction: FlexDirection::Column,
                position_type: PositionType::Absolute,
                row_gap: Val::Px(6.0),
                left: Val::Px(10.0),
                bottom: Val::Px(10.0),
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            spawn_option_buttons(
                parent,
                "Light Type",
                &[
                    (AppSetting::LightType(LightType::Directional), "Directional"),
                    (AppSetting::LightType(LightType::Point), "Point"),
                    (AppSetting::LightType(LightType::Spot), "Spot"),
                ],
            );
        });
}

pub fn spawn_option_buttons<T>(parent: &mut ChildBuilder, title: &str, options: &[(T, &str)])
where
    T: Clone + Send + Sync + 'static,
{
    // Add the parent node for the row.
    parent
        .spawn(NodeBundle {
            style: Style {
                align_items: AlignItems::Center,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            spawn_ui_text(parent, title, Color::BLACK).insert(Style {
                width: Val::Px(125.0),
                ..default()
            });

            for (option_index, (option_value, option_name)) in options.iter().cloned().enumerate() {
                spawn_option_button(
                    parent,
                    option_value,
                    option_name,
                    option_index == 0,
                    option_index == 0,
                    option_index == options.len() - 1,
                );
            }
        });
}

pub fn spawn_ui_text<'a>(
    parent: &'a mut ChildBuilder,
    label: &str,
    color: Color,
) -> EntityCommands<'a> {
    parent.spawn(TextBundle::from_section(
        label,
        TextStyle {
            font_size: 18.0,
            color,
            ..default()
        },
    ))
}

pub fn spawn_option_button<T>(
    parent: &mut ChildBuilder,
    option_value: T,
    option_name: &str,
    is_selected: bool,
    is_first: bool,
    is_last: bool,
) where
    T: Clone + Send + Sync + 'static,
{
    let (bg_color, fg_color) = if is_selected {
        (Color::WHITE, Color::BLACK)
    } else {
        (Color::BLACK, Color::WHITE)
    };

    parent
        .spawn(ButtonBundle {
            style: Style {
                border: UiRect::all(Val::Px(1.0)).with_left(if is_first {
                    Val::Px(1.0)
                } else {
                    Val::Px(0.0)
                }),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                padding: UiRect::axes(Val::Px(2.0), Val::Px(6.0)),
                ..default()
            },
            border_color: BorderColor(Color::WHITE),
            border_radius: BorderRadius::ZERO
                .with_left(if is_first { Val::Px(6.0) } else { Val::Px(0.0) })
                .with_right(if is_last { Val::Px(6.0) } else { Val::Px(0.0) }),
            background_color: BackgroundColor(bg_color),
            ..default()
        })
        .insert(RadioButton)
        .insert(WidgetClickSender(option_value.clone()))
        .with_children(|parent| {
            spawn_ui_text(parent, option_name, fg_color)
                .insert(RadioButtonText)
                .insert(WidgetClickSender(option_value));
        });
}
