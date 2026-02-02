use bevy::{ecs::system::EntityCommands, prelude::*};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};

#[derive(Component)]
pub struct UiCamera;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, (setup, setup_card_ui))
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d::default());
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

const GOLEN_RATIO: f32 = 1.618;

/// Configuration derived from window size to ensure consistent card proportions.
pub struct CardUIConfig {
    pub width: f32,
    pub height: f32,
    pub scale: f32,
    pub padding: f32,
    pub font_title: f32,
    pub font_body: f32,
    pub orb_size: f32,
    pub spacing: f32,
}

impl CardUIConfig {
    pub fn new(target_height: f32) -> Self {
        // Use 300.0 as the base reference height from the original design
        let scale = target_height / 300.0;

        Self {
            width: target_height / GOLEN_RATIO,
            height: target_height,
            scale,
            padding: 12.0 * scale,
            font_title: 20.0 * scale,
            font_body: 16.0 * scale,
            orb_size: 42.0 * scale,
            spacing: 15.0 * scale,
        }
    }
}

/// Extension trait for [`Commands`] to spawn `Card*`
pub trait CardSpawnExt {
    // fn spawn_card(&'_ mut self, prop: &CardProp) -> EntityCommands<'_>;
    fn spawn_card(&'_ mut self, prop: &CardProp, config: &CardUIConfig) -> EntityCommands<'_>;
}

impl CardSpawnExt for Commands<'_, '_> {
    fn spawn_card(&'_ mut self, prop: &CardProp, config: &CardUIConfig) -> EntityCommands<'_> {
        // 1. Root Card Container
        // Uses fixed Px values derived from the dynamic config to maintain Aspect Ratio
        let mut card = self.spawn((
            Node {
                width: Val::Px(config.width),
                height: Val::Px(config.height),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                padding: UiRect::all(Val::Px(config.padding)),
                border: UiRect::all(Val::Px(3.0 * config.scale)),
                ..default()
            },
            BackgroundColor(Color::srgb(0.35, 0.08, 0.08)),
            BorderColor::all(Color::srgb(0.7, 0.6, 0.3)),
        ));

        card.with_children(|parent| {
            // 2. Header Row
            parent
                .spawn(Node {
                    width: Val::Percent(100.0),
                    height: Val::Percent(20.0),
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|header| {
                    // Energy Orb
                    header
                        .spawn((
                            Node {
                                width: Val::Percent(20.0),
                                aspect_ratio: Some(1.0),
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
                                    font_size: config.font_title * 1.2,
                                    ..default()
                                },
                                CardCost {
                                    base: prop.base_cost,
                                    curr: prop.base_cost,
                                },
                            ));
                        });

                    header.spawn((
                        Text::new(prop.title.clone()),
                        TextFont {
                            font_size: config.font_title,
                            ..default()
                        },
                        CardTitle(prop.title.clone()),
                    ));
                });

            // 3. Illustration Area
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_grow: 1.0,
                    margin: UiRect::vertical(Val::Percent(5.0)),
                    border: UiRect::all(Val::Px(2.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
            ));

            // 4. Description Area
            parent
                .spawn(Node {
                    width: Val::Percent(100.0),
                    height: Val::Percent(30.0),
                    justify_content: JustifyContent::Center,
                    ..default()
                })
                .with_children(|desc_node| {
                    desc_node.spawn((
                        Text::new(prop.base_desc.join("\n")),
                        TextFont {
                            font_size: config.font_body,
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

fn setup_card_ui(mut commands: Commands, window: Single<&Window>) {
    let card_height = window.height() * 0.30;
    let config = CardUIConfig::new(card_height);

    let strike_prop = CardProp {
        title: "Strike".into(),
        base_cost: 1,
        base_desc: vec!["Deal 6".into(), "damage.".into()],
        rare: 1,
    };

    commands.spawn_card(&strike_prop, &config);
}
