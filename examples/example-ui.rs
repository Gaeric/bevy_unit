use bevy::{ecs::system::EntityCommands, prelude::*};

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

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, (setup, setup_card_ui))
        .run();
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

    commands.spawn(Camera2d::default());

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
    let container = Node {
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        justify_content: JustifyContent::Center,
        ..default()
    };

    let square = Node {
        width: Val::Px(20.0),
        border: UiRect::all(Val::Px(2.)),
        left: Val::Px(2.0),
        ..default()
    };

    let parent = commands.spawn(container).id();
    let child = commands
        .spawn((
            square,
            BackgroundColor(Color::srgba(0.65, 0.65, 0.65, 0.50)),
        ))
        .id();

    commands.entity(parent).add_child(child);
}

fn spawn_buttons(commands: &mut Commands) {
    commands
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            position_type: PositionType::Absolute,
            row_gap: Val::Px(6.0),
            left: Val::Px(10.0),
            bottom: Val::Px(10.0),
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

pub fn spawn_option_buttons<T>(
    parent: &mut ChildSpawnerCommands,
    title: &str,
    options: &[(T, &str)],
) where
    T: Clone + Send + Sync + 'static,
{
    // Add the parent node for the row.
    parent
        .spawn(Node {
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|parent| {
            spawn_ui_text(parent, title, Color::BLACK).insert(Node {
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
    parent: &'a mut ChildSpawnerCommands,
    label: &str,
    color: Color,
) -> EntityCommands<'a> {
    parent.spawn((
        Text::new(label),
        TextColor(color),
        TextFont {
            font_size: 18.0,
            ..default()
        },
    ))
}

pub fn spawn_option_button<T>(
    parent: &mut ChildSpawnerCommands,
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
        .spawn((
            Button,
            Node {
                border: UiRect::all(Val::Px(1.0)).with_left(if is_first {
                    Val::Px(1.0)
                } else {
                    Val::Px(0.0)
                }),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                padding: UiRect::axes(Val::Px(2.0), Val::Px(6.0)),
                border_radius: BorderRadius::ZERO
                    .with_left(if is_first { Val::Px(6.0) } else { Val::Px(0.0) })
                    .with_right(if is_last { Val::Px(6.0) } else { Val::Px(0.0) }),
                ..default()
            },
            BorderColor::all(Color::WHITE),
            BackgroundColor(bg_color),
        ))
        .insert(RadioButton)
        .insert(WidgetClickSender(option_value.clone()))
        .with_children(|parent| {
            spawn_ui_text(parent, option_name, fg_color)
                .insert(RadioButtonText)
                .insert(WidgetClickSender(option_value));
        });
}

#[derive(Component)]
struct Card;

fn setup_card_ui(mut commands: Commands) {
    // Spawn 2D Camera required for UI rendering

    // 1. Root Card Container
    // We use a Column layout to stack the header, image, and description vertically.
    commands
        .spawn((
            Card,
            Node {
                width: Val::Px(260.0),
                height: Val::Px(380.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                padding: UiRect::all(Val::Px(12.0)),
                border: UiRect::all(Val::Px(3.0)),
                ..default()
            },
            // Ironclad's signature deep red background
            BackgroundColor(Color::srgb(0.35, 0.08, 0.08)),
            // Golden/Bronze border
            BorderColor::all(Color::srgb(0.7, 0.6, 0.3)),
        ))
        .with_children(|parent| {
            // 2. Header Row: Contains Energy Cost and Card Name
            parent
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|header| {
                    // Energy Orb (The blue circle in the top left)
                    header
                        .spawn((
                            Node {
                                width: Val::Px(42.0),
                                height: Val::Px(42.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                // Note: Bevy 0.18 uses BorderRadius for rounding corners
                                border_radius: BorderRadius::all(Val::Percent(50.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.1, 0.3, 0.7)), // Energy Blue
                        ))
                        .with_children(|orb| {
                            orb.spawn((
                                Text::new("1"),
                                TextFont {
                                    font_size: 24.0,
                                    ..default()
                                },
                            ));
                        });

                    // Card Title
                    header.spawn((
                        Text::new("STRIKE"),
                        TextFont {
                            font_size: 20.0,
                            ..default()
                        },
                    ));
                });

            // 3. Illustration Area
            // In a real game, you would spawn an ImageNode here
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(160.0),
                    margin: UiRect::vertical(Val::Px(15.0)),
                    border: UiRect::all(Val::Px(2.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
                BorderColor::all(Color::srgb(0.5, 0.4, 0.2)),
            ));

            // 4. Card Type Banner (Attack)
            parent
                .spawn((
                    Node {
                        padding: UiRect::axes(Val::Px(15.0), Val::Px(2.0)),
                        margin: UiRect::bottom(Val::Px(10.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.25, 0.05, 0.05)),
                ))
                .with_children(|banner| {
                    banner.spawn((
                        Text::new("Attack"),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                    ));
                });

            // 5. Description Area
            parent
                .spawn(Node {
                    flex_grow: 1.0,
                    justify_content: JustifyContent::Center,
                    ..default()
                })
                .with_children(|desc| {
                    desc.spawn((
                        Text::new("Deal 6 damage."),
                        TextLayout::new_with_justify(Justify::Center),
                        TextFont {
                            font_size: 18.0,
                            ..default()
                        },
                    ));
                });
        });
}
