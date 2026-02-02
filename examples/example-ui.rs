use bevy::{ecs::system::EntityCommands, prelude::*};

#[derive(Component)]
pub struct UiCamera;

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

/*
some issue should to be done
1. dynamic change the desc/effect
2. temp change the cost
3. change the cost after/before
*/

#[derive(Clone, Debug)]
pub struct CardProp {
    title: String,
    base_cost: u8,
    base_desc: Vec<String>,
    // color: Color,
    // todo: enum rare
    rare: u8,
    // path: Path,
}

#[derive(Component, Clone, Debug)]
pub struct CardTitle(String);

#[derive(Component, Clone, Debug)]
pub struct CardCost {
    base: u8,
    curr: u8,
}

#[derive(Component, Clone, Debug)]
pub struct CardDesc(String);

#[derive(Component, Clone, Debug)]
pub struct CardColor(String);

/// Extension trait for [`Commands`] to spawn `Card*`
pub trait CardSpawnExt {
    fn spawn_card(&'_ mut self, prop: &CardProp) -> EntityCommands<'_>;
}

impl CardSpawnExt for Commands<'_, '_> {
    fn spawn_card(&'_ mut self, prop: &CardProp) -> EntityCommands<'_> {
        // 1. Root Card Container
        // Acts as the parent frame with background and border.
        let mut card = self.spawn((
            Node {
                width: Val::Px(260.0),
                height: Val::Px(380.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                padding: UiRect::all(Val::Px(12.0)),
                border: UiRect::all(Val::Px(3.0)),
                ..default()
            },
            // Background color could be mapped from prop.rare in the future
            BackgroundColor(Color::srgb(0.35, 0.08, 0.08)),
            BorderColor::all(Color::srgb(0.7, 0.6, 0.3)),
        ));

        card.with_children(|parent| {
            // 2. Header Row (Cost Orb & Title)
            parent
                .spawn(Node {
                    width: Val::Percent(100.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|header| {
                    // Energy Orb: Contains the Cost Text and CardCost Component
                    header
                        .spawn((
                            Node {
                                width: Val::Px(42.0),
                                height: Val::Px(42.0),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                border_radius: BorderRadius::all(Val::Percent(50.0)),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.1, 0.3, 0.7)),
                        ))
                        .with_children(|orb| {
                            orb.spawn((
                                Text::new(prop.base_cost.to_string()),
                                TextFont {
                                    font_size: 24.0,
                                    ..default()
                                },
                                // Tagging the text node with CardCost for dynamic updates
                                CardCost {
                                    base: prop.base_cost,
                                    curr: prop.base_cost,
                                },
                            ));
                        });

                    // Card Title: Tagging with CardTitle Component
                    header.spawn((
                        Text::new(prop.title.clone()),
                        TextFont {
                            font_size: 20.0,
                            ..default()
                        },
                        CardTitle(prop.title.clone()),
                    ));
                });

            // 3. Illustration Area (Static placeholder for now)
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

            // 4. Description Area: Tagging with CardDesc Component
            parent
                .spawn(Node {
                    flex_grow: 1.0,
                    justify_content: JustifyContent::Center,
                    ..default()
                })
                .with_children(|desc_node| {
                    desc_node.spawn((
                        // Joining Vec<String> with newlines for the UI
                        Text::new(prop.base_desc.join("\n")),
                        TextFont {
                            font_size: 18.0,
                            ..default()
                        },
                        TextLayout::new_with_justify(Justify::Center),
                        CardDesc(prop.base_desc.join("\n")),
                    ));
                });
        });
        card
    }
}

fn setup_card_ui(mut commands: Commands) {
    // Spawn 2D Camera required for UI rendering

    let strike_config = CardProp {
        title: "Strike".into(),
        base_cost: 1,
        base_desc: vec!["Deal 6".into(), "damage.".into()],
        rare: 1,
    };

    commands.spawn_card(&strike_config);
}
