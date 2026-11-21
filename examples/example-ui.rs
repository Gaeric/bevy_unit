use bevy::{ecs::system::EntityCommands, prelude::*, window::WindowResized};

const GOLEN_RATIO: f32 = 1.618;

const CARD_TOTAL_BG: Color = Color::srgb(0.35, 0.08, 0.08);
const CARD_TOTAL_BORDER: Color = Color::srgb(0.7, 0.6, 0.3);
const CARD_ORB_BG: Color = Color::srgb(0.1, 0.3, 0.7);
const CARD_IMG_BG: Color = Color::srgb(0.15, 0.15, 0.15);

#[derive(Component)]
pub struct UiCamera;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Startup,
            (setup, setup_game_interface, setup_card_ui).chain(),
        )
        .add_systems(Update, window_resizing)
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

#[derive(Component, Clone, Default)]
pub struct CardContainer;

#[derive(Component, Clone)]
pub enum CardUIText {
    CardUITitle(String),
    CardUICost { base: u8, curr: u8 },
    CardUIDesc(String),
}

impl Default for CardUIText {
    fn default() -> Self {
        Self::CardUITitle(String::new())
    }
}

#[derive(Component, Clone, Default)]
pub enum CardColorArea {
    #[default]
    CardTotalBg,
    CardOrbBg,
    CardImgBg,
}

#[derive(Component, Clone, Default)]
struct HistoryCardsArea;
#[derive(Component, Clone, Default)]
struct MiddleArea;

#[derive(Component, Clone, Default)]
struct BottomArea;
#[derive(Component, Clone, Default)]
pub struct BottomLeftArea;

#[derive(Component, Clone, Default)]
pub struct BottomMiddleArea;

#[derive(Component, Clone, Default)]
pub struct BottomRightArea;

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
    fn spawn_card(&mut self, prop: &CardProp, config: &CardUIConfig) -> EntityCommands<'_> {
        let base_cost = prop.base_cost;
        let desc = prop.base_desc.join("\n");

        self.spawn_scene(bsn! {
            CardContainer
            CardColorArea::CardTotalBg
            Node {
                width: px(config.width),
                height: px(config.height),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                padding: px(config.padding),
                border: px(3.0 * config.scale),
                border_radius: BorderRadius::all(percent(5)),
            }
            BackgroundColor(CARD_TOTAL_BG)
            BorderColor::all(CARD_TOTAL_BORDER)
            on(point_over)
            on(point_out)
            on(point_drag_start)
            on(point_drag)
            on(point_drag_end)
            on(point_drag_drop)
            Children [
                // Header: energy orb, then title.
                Node {
                    width: percent(100),
                    height: percent(20),
                    position_type: PositionType::Relative,
                    justify_content: JustifyContent::Center,
                }
                Children [
                    CardColorArea::CardOrbBg
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(-config.padding),
                        top: px(-config.padding),
                        width: percent(20),
                        aspect_ratio: {Some(1.0)},
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border_radius: BorderRadius::all(percent(50)),
                    }
                    BackgroundColor(CARD_ORB_BG)
                    Children [
                        Text({base_cost.to_string()})
                        TextFont { font_size: px(config.font_cost) }
                        CardUIText::CardUICost { base: base_cost, curr: base_cost }
                    ]
                    --
                    Text({prop.title.clone()})
                    TextFont { font_size: px(config.font_title) }
                    CardUIText::CardUITitle({prop.title.clone()})
                ]
                --
                // Illustration.
                CardColorArea::CardImgBg
                Node {
                    width: percent(100),
                    aspect_ratio: {Some(1.0)},
                    border: px(2),
                    border_radius: BorderRadius::all(percent(50)),
                }
                BackgroundColor(CARD_IMG_BG)
                --
                // Description.
                Node {
                    width: percent(100),
                    height: percent(30),
                    justify_content: JustifyContent::Center,
                }
                Children [
                    Text({desc.clone()})
                    TextFont { font_size: px(config.font_desc) }
                    TextLayout::justify(Justify::Center)
                    CardUIText::CardUIDesc(desc)
                ]
            ]
        })
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

fn setup_game_interface(mut commands: Commands, _asset_server: Res<AssetServer>) {
    commands.spawn_scene(bsn! {
        Node {
            width: percent(100),
            height: percent(100),
            flex_direction: FlexDirection::Column,
        }
        Children [
            // Top bar.
            Node {
                width: percent(100),
                height: percent(10),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::Start,
                align_items: AlignItems::Start,
            }
            BackgroundColor(Color::srgb(0.2, 0.2, 0.2))
            Children [
                HistoryCardsArea
                Node { width: percent(40), height: percent(100) }
                BackgroundColor(Color::srgb(0.3, 0.1, 0.1))
            ]
            --
            MiddleArea
            Node {
                width: percent(100),
                flex_grow: 1.0,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
            }
            BackgroundColor(Color::srgb(0.1, 0.1, 0.2))
            --
            BottomArea
            Node {
                width: percent(100),
                height: percent(40),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                position_type: PositionType::Relative,
                flex_direction: FlexDirection::Row,
            }
            BackgroundColor(Color::srgb(0.1, 0.2, 0.1))
            Children [
                BottomLeftArea
                Node { width: percent(20), height: percent(100) }
                BackgroundColor(Color::srgb(0.15, 0.15, 0.15))
                --
                BottomMiddleArea
                Node {
                    width: percent(60),
                    height: percent(100),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                }
                BackgroundColor(Color::srgb(0.12, 0.2, 0.12))
                --
                BottomRightArea
                Node { width: percent(20), height: percent(100) }
                BackgroundColor(Color::srgb(0.15, 0.15, 0.15))
            ]
        ]
    });
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
            font.font_size = FontSize::Px(match *kind {
                CardUIText::CardUITitle(_) => new_config.font_title,
                CardUIText::CardUICost { .. } => new_config.font_cost,
                CardUIText::CardUIDesc(_) => new_config.font_desc,
            });
        }
    }
}

fn point_over(
    on_over: On<PointerOver>,
    mut card_areas: Query<(&mut BackgroundColor, &CardColorArea)>,
    mut border_areas: Query<&mut BorderColor>,
) {
    if let Ok(mut border_color) = border_areas.get_mut(on_over.event_target()) {
        border_color.set_all(CARD_TOTAL_BORDER.lighter(0.1));
    }

    if let Ok((mut bg_color, area)) = card_areas.get_mut(on_over.event_target()) {
        bg_color.0 = match area {
            CardColorArea::CardTotalBg => CARD_TOTAL_BG,
            CardColorArea::CardOrbBg => CARD_TOTAL_BG,
            CardColorArea::CardImgBg => CARD_TOTAL_BG,
        }
        .lighter(0.1)
    }
}

fn point_out(
    on_over: On<PointerOut>,
    mut card_areas: Query<(&mut BackgroundColor, &CardColorArea)>,
    mut border_areas: Query<&mut BorderColor>,
) {
    if let Ok(mut border_color) = border_areas.get_mut(on_over.event_target()) {
        border_color.set_all(CARD_TOTAL_BORDER);
    }

    if let Ok((mut bg_color, area)) = card_areas.get_mut(on_over.event_target()) {
        bg_color.0 = match area {
            CardColorArea::CardTotalBg => CARD_TOTAL_BG,
            CardColorArea::CardOrbBg => CARD_TOTAL_BG,
            CardColorArea::CardImgBg => CARD_TOTAL_BG,
        }
    }
}

fn point_drag_start(
    on_drag_start: On<PointerDragStart>,
    mut query: Query<(&mut Outline, &mut GlobalZIndex)>,
) {
    if let Ok((mut outline, mut global_zindex)) = query.get_mut(on_drag_start.event_target()) {
        outline.color = Color::WHITE;
        global_zindex.0 = 1;
    }
}

fn point_drag(on_drag: On<PointerDrag>, mut query: Query<&mut UiTransform>) {
    if let Ok(mut transform) = query.get_mut(on_drag.event_target()) {
        transform.translation = Val2::px(on_drag.distance.x, on_drag.distance.y);
    }
}

fn point_drag_end(
    on_drag_end: On<PointerDragEnd>,
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

fn point_drag_drop(on_drag_drop: On<PointerDragDrop>, mut query: Query<&mut Node>) {
    if let Ok([mut a, mut b]) =
        query.get_many_mut([on_drag_drop.event_target(), on_drag_drop.dropped])
    {
        core::mem::swap(&mut a.grid_row, &mut b.grid_row);
        core::mem::swap(&mut a.grid_column, &mut b.grid_column);
    }
}
