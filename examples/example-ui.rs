use bevy::{ecs::system::EntityCommands, prelude::*, window::WindowResized};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};

#[derive(Component)]
pub struct UiCamera;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(WorldInspectorPlugin::new())
        .add_systems(Startup, (setup, setup_card_ui))
        .add_systems(Update, window_resizing)
        .add_observer(on_drag_enter)
        .add_observer(on_drag_over)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d::default());
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

#[derive(Component)]
pub struct CardContainer;

#[derive(Component)]
pub enum CardUIText {
    CardUITitle(String),
    CardUICost { base: u8, curr: u8 },
    CardUIDesc(String),
}

#[derive(Component, Clone, Debug)]
pub struct CardColor(String);

const GOLEN_RATIO: f32 = 1.618;

#[derive(Component)]
pub struct DraggableCard;

#[derive(Component)]
pub struct GhostCard;

/// Configuration derived from window size to ensure consistent card proportions.
pub struct CardUIConfig {
    pub width: f32,
    pub height: f32,
    pub scale: f32,
    pub padding: f32,
    pub font_cost: f32,
    pub font_title: f32,
    pub font_desc: f32,
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
            font_cost: 24.0 * scale,
            font_title: 20.0 * scale,
            font_desc: 16.0 * scale,
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
            CardContainer,
            DraggableCard,
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
                                    font_size: config.font_cost,
                                    ..default()
                                },
                                CardUIText::CardUICost {
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
                        CardUIText::CardUITitle(prop.title.clone()),
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
                            font_size: config.font_desc,
                            ..default()
                        },
                        TextLayout::new_with_justify(Justify::Center),
                        CardUIText::CardUIDesc(prop.base_desc.join("\n")),
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

fn on_drag_enter(
    mut event: On<Pointer<DragEnter>>,
    cards: Query<Entity, With<DraggableCard>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    if let Ok(_entity) = cards.get(event.dragged) {
        let Some(position) = event.hit.position else {
            return;
        };

        info!("position is {:?}", position);

        commands.spawn((
            GhostCard,
            Mesh2d(meshes.add(Circle::new(25.0))),
            MeshMaterial2d(materials.add(Color::srgba(1.0, 1.0, 0.6, 0.5))),
            Transform::from_translation(position + 2. * Vec3::Z),
            Pickable::IGNORE,
        ));

        event.propagate(false);
    }
}

fn on_drag_over(
    mut event: On<Pointer<DragOver>>,
    cards: Query<Entity, With<DraggableCard>>,
    mut ghost_transform: Single<&mut Transform, With<GhostCard>>,
) {
    if let Ok(_) = cards.get(event.dragged) {
        let Some(position) = event.hit.position else {
            return;
        };

        ghost_transform.translation = position;
        event.propagate(false);
    }
}

pub fn window_resizing(
    mut resize_reader: MessageReader<WindowResized>,
    mut card_root: Query<&mut Node, With<CardContainer>>,
    mut card_texts: Query<(&mut TextFont, &CardUIText)>,
) {
    for e in resize_reader.read() {
        let target_card_height = e.height * 0.30;
        let new_config = CardUIConfig::new(target_card_height);

        for mut node in &mut card_root {
            node.width = Val::Px(new_config.width);
            node.height = Val::Px(new_config.height);
            node.padding = UiRect::all(Val::Px(new_config.padding));
            node.border = UiRect::all(Val::Px(3.0 * new_config.scale));
        }

        for (mut font, kind) in &mut card_texts {
            font.font_size = match *kind {
                CardUIText::CardUITitle(_) => new_config.font_title,
                CardUIText::CardUICost { .. } => new_config.font_cost,
                CardUIText::CardUIDesc(_) => new_config.font_desc,
            }
        }
    }
}
