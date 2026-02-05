use bevy::{ecs::system::EntityCommands, prelude::*, window::WindowResized};
use bevy_inspector_egui::{bevy_egui::EguiPlugin, quick::WorldInspectorPlugin};

#[derive(Component)]
pub struct UiCamera;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin::default())
        .add_plugins(WorldInspectorPlugin::new())
        .add_systems(
            Startup,
            (setup, setup_game_interface, setup_card_ui).chain(),
        )
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

#[derive(Component)]
struct HistoryCardsArea;
#[derive(Component)]
struct MiddleArea;

#[derive(Component)]
struct BottomArea;
#[derive(Component)]
pub struct BottomLeftArea;

#[derive(Component)]
pub struct BottomMiddleArea;

#[derive(Component)]
pub struct BottomRightArea;

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
                border_radius: BorderRadius::all(Val::Percent(5.0)),
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
                    // margin: UiRect::vertical(Val::Percent(5.0)),
                    position_type: PositionType::Relative,
                    justify_content: JustifyContent::Center,
                    // align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|header| {
                    // Energy Orb
                    header
                        .spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                left: Val::Px(-config.padding),
                                top: Val::Px(-config.padding),
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
                    aspect_ratio: Some(1.0),
                    // flex_grow: 1.0,
                    // margin: UiRect::vertical(Val::Percent(5.0)),
                    border: UiRect::all(Val::Px(2.0)),
                    border_radius: BorderRadius::all(Val::Percent(50.0)),
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

fn setup_card_ui(
    mut commands: Commands,
    window: Single<&Window>,
    middle_area_entity: Single<Entity, With<BottomMiddleArea>>,
) {
    let card_height = window.height() * 0.30;
    let config = CardUIConfig::new(card_height);

    let strike_prop = CardProp {
        title: "Strike".into(),
        base_cost: 1,
        base_desc: vec!["Deal 6".into(), "damage.".into()],
        rare: 1,
    };

    // Spawn the card as a child of the MiddleArea
    // commands.entity(middle_area_entity).with_children(|parent| {
    //     parent.spawn_card(&strike_prop, &config);
    // });

    let entity: Entity = middle_area_entity.into_inner();

    for _ in 0..10 {
        let card_entity = commands.spawn_card(&strike_prop, &config).id();
        commands.entity(card_entity).set_parent_in_place(entity);
    }

    // commands.spawn_card(&strike_prop, &config);
}

fn setup_game_interface(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn((Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            ..default()
        },))
        .with_children(|parent| {
            // (Top Section)
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(10.0),
                        flex_direction: FlexDirection::Row,
                        justify_content: JustifyContent::Start,
                        align_items: AlignItems::Start,
                        // padding: UiRect::all(Val::Px(10.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                ))
                .with_children(|top_section| {
                    // (HistoryCardsArea Section)
                    top_section.spawn((
                        Node {
                            width: Val::Percent(40.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.3, 0.1, 0.1)),
                        HistoryCardsArea,
                    ));
                });

            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_grow: 1.0,
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::srgb(0.1, 0.1, 0.2)),
                MiddleArea,
            ));
            //  (Bottom Section)
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Percent(40.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        position_type: PositionType::Relative,
                        flex_direction: FlexDirection::Row,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.1, 0.2, 0.1)),
                    BottomArea,
                ))
                .with_children(|bottom| {
                    // Left Sub-Area (e.g., Draw Pile)
                    bottom.spawn((
                        Node {
                            width: Val::Percent(20.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
                        BottomLeftArea,
                    ));

                    // Middle Sub-Area (Main Hand)
                    bottom.spawn((
                        Node {
                            width: Val::Percent(60.0), // Takes most of the space
                            height: Val::Percent(100.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        // Transparent or slightly different green
                        BackgroundColor(Color::srgb(0.12, 0.2, 0.12)),
                        BottomMiddleArea,
                    ));

                    // Right Sub-Area (e.g., Discard Pile)
                    bottom.spawn((
                        Node {
                            width: Val::Percent(20.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
                        BottomRightArea,
                    ));
                });
        });
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

fn point_over(
    on_over: On<Pointer<Over>>,
    mut query: Query<(&mut BackgroundColor, &mut BorderColor)>,
) {
    if let Ok((mut background_color, mut border_color)) = query.get_mut(on_over.event_target()) {
        let tile_color = background_color.0;
        let tile_border_color = border_color.top;
        background_color.0 = tile_color.lighter(0.1);
        border_color.set_all(tile_border_color.lighter(0.1));
    }
}

fn point_out(on_out: On<Pointer<Out>>, mut query: Query<(&mut BackgroundColor, &mut BorderColor)>) {
    if let Ok((mut background_color, mut border_color)) = query.get_mut(on_out.event_target()) {
        let tile_color = background_color.0;
        let tile_border_color = border_color.top;
        background_color.0 = tile_color;
        border_color.set_all(tile_border_color);
    }
}

fn point_drag_start(
    on_drag_start: On<Pointer<DragStart>>,
    mut query: Query<(&mut Outline, &mut GlobalZIndex)>,
) {
    if let Ok((mut outline, mut global_zindex)) = query.get_mut(on_drag_start.event_target()) {
        outline.color = Color::WHITE;
        global_zindex.0 = 1;
    }
}

fn point_drag(on_drag: On<Pointer<Drag>>, mut query: Query<&mut UiTransform>) {
    if let Ok(mut transform) = query.get_mut(on_drag.event_target()) {
        transform.translation = Val2::px(on_drag.distance.x, on_drag.distance.y);
    }
}

fn point_drag_end(
    on_drag_end: On<Pointer<DragEnd>>,
    mut query: Query<(&mut UiTransform, &mut Outline, &mut GlobalZIndex)>,
) {
    if let Ok((mut transform, mut outline, mut global_zindex)) =
        query.get_mut(on_drag_end.event_target())
    {
        transform.translation = Val2::ZERO;
        outline.color = Color::NONE;
        global_zindex.0 = 0;
    }
}

fn point_drag_drop(on_drag_drop: On<Pointer<DragDrop>>, mut query: Query<&mut Node>) {
    if let Ok([mut a, mut b]) =
        query.get_many_mut([on_drag_drop.event_target(), on_drag_drop.dropped])
    {
        core::mem::swap(&mut a.grid_row, &mut b.grid_row);
        core::mem::swap(&mut a.grid_column, &mut b.grid_column);
    }
}
