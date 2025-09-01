use bevy::{color::palettes::css::LIGHT_GRAY, prelude::*};

const MASK_GROUP_HEAD: u32 = 0;
const MASK_GROUP_LEFT_FRONT_LEG: u32 = 1;
const MASK_GROUP_RIGHT_FRONT_LEG: u32 = 2;
const MASK_GROUP_LEFT_HIND_LEG: u32 = 3;
const MASK_GROUP_RIGHT_HIND_LEG: u32 = 4;
const MASK_GROUP_TAIL: u32 = 5;

// This width in pixel of the samll buttons that allow the user to toggle a mask
// group on or off.
const MASK_GROUP_BUTTON_WIDTH: f32 = 250.0;

// The names of the bones that each mask group consists of. Each mask group is
// defined as a (prefix, suffix) tuple. The mask group consists of a single
// bone chain rooted at the prefix. For example, if the chain's prefix is
// "A/B/C" and the suffix is "D/E", then the bones that will be included in the
// mask group are "A/B/C", "A/B/C/D", and "A/B/C/D/E".
//
// The fact that our mask groups are single chains of bones isn't an engine
// requirement; it just so happens to be the case for the model we're using. A
// mask group can consist of any set of animation targets, regardless of whether
// they form a single chain.
const MASK_GROUP_PATHS: [(&str, &str); 6] = [
    // Head
    (
        "root/_rootJoint/b_Root_00/b_Hip_01/b_Spine01_02/b_Spine02_03",
        "b_Neck_04/b_Head_05",
    ),
    // Left front leg
    (
        "root/_rootJoint/b_Root_00/b_Hip_01/b_Spine01_02/b_Spine02_03/b_LeftUpperArm_09",
        "b_LeftForeArm_010/b_LeftHand_011",
    ),
    // Right front leg
    (
        "root/_rootJoint/b_Root_00/b_Hip_01/b_Spine01_02/b_Spine02_03/b_RightUpperArm_06",
        "b_RightForeArm_07/b_RightHand_08",
    ),
    // Left hind leg
    (
        "root/_rootJoint/b_Root_00/b_Hip_01/b_LeftLeg01_015",
        "b_LeftLeg02_016/b_LeftFoot01_017/b_LeftFoot02_018",
    ),
    // Right hind leg
    (
        "root/_rootJoint/b_Root_00/b_Hip_01/b_RightLeg01_019",
        "b_RightLeg02_020/b_RightFoot01_021/b_RightFoot02_022",
    ),
    // Tail
    (
        "root/_rootJoint/b_Root_00/b_Hip_01/b_Tail01_012",
        "b_Tail02_013/b_Tail03_014",
    ),
];

#[derive(Clone, Copy, Component)]
struct AnimationControl {
    // The ID of the mask group that this button controls.
    group_id: u32,
    label: AnimationLabel,
}

#[derive(Clone, Copy, Component, PartialEq, Debug)]
enum AnimationLabel {
    Idle = 0,
    Walk = 1,
    Run = 2,
    Off = 3,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy animation masks example".into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, (setup_scene, setup_ui))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-15.0, 10.0, 20.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 10_000_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(-4.0, 8.0, 13.0),
    ));

    // commands.spawn((
    //     SceneRoot(
    //         asset_server.load(GltfAssetLabel::Scene(0).from_asset("waltz/scenes/library/Fox.glb")),
    //     ),
    //     Transform::from_scale(Vec3::splat(0.07)),
    // ));

    commands.spawn((
        Mesh3d(meshes.add(Circle::new(7.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
}

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Click on a button to toggle animations for its associated bones"),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(12.0),
            top: Val::Px(12.0),
            ..default()
        },
    ));

    // Add the button that allow the user to toggle mask groups on and off.
    commands
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            position_type: PositionType::Absolute,
            row_gap: Val::Px(6.0),
            left: Val::Px(12.0),
            bottom: Val::Px(12.0),
            ..default()
        })
        .with_children(|parent| {
            let row_node = Node {
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(6.0),
                ..default()
            };

            add_mask_group_control(parent, "Head", Val::Auto, MASK_GROUP_HEAD);

            parent.spawn(row_node.clone()).with_children(|parent| {
                add_mask_group_control(
                    parent,
                    "Left Front Leg",
                    Val::Px(MASK_GROUP_BUTTON_WIDTH),
                    MASK_GROUP_LEFT_FRONT_LEG,
                );

                add_mask_group_control(
                    parent,
                    "Right Front Leg",
                    Val::Px(MASK_GROUP_BUTTON_WIDTH),
                    MASK_GROUP_RIGHT_FRONT_LEG,
                );
            });

            parent.spawn(row_node).with_children(|parent| {
                add_mask_group_control(
                    parent,
                    "Left Hind Leg",
                    Val::Px(MASK_GROUP_BUTTON_WIDTH),
                    MASK_GROUP_LEFT_HIND_LEG,
                );
                add_mask_group_control(
                    parent,
                    "Right Hind Leg",
                    Val::Px(MASK_GROUP_BUTTON_WIDTH),
                    MASK_GROUP_RIGHT_HIND_LEG,
                );
            });

            add_mask_group_control(parent, "Tail", Val::Auto, MASK_GROUP_TAIL);
        });
}

fn add_mask_group_control(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    width: Val,
    mask_group_id: u32,
) {
    let button_text_style = (
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor::WHITE,
    );

    let selected_button_text_style = (button_text_style.0.clone(), TextColor::BLACK);

    let label_text_style = (
        button_text_style.0.clone(),
        TextColor(Color::Srgba(LIGHT_GRAY)),
    );

    parent
        .spawn((
            Node {
                border: UiRect::all(Val::Px(1.0)),
                width,
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                padding: UiRect::ZERO,
                margin: UiRect::ZERO,
                ..default()
            },
            BorderColor(Color::WHITE),
            BorderRadius::all(Val::Px(3.0)),
            BackgroundColor(Color::BLACK),
        ))
        .with_children(|builder| {
            builder
                .spawn((
                    Node {
                        border: UiRect::ZERO,
                        width: Val::Percent(100.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        padding: UiRect::ZERO,
                        margin: UiRect::ZERO,
                        ..default()
                    },
                    BackgroundColor(Color::BLACK),
                ))
                .with_child((
                    Text::new(label),
                    label_text_style.clone(),
                    Node {
                        margin: UiRect::vertical(Val::Px(3.0)),
                        ..default()
                    },
                ));

            builder
                .spawn(
                    (Node {
                        width: Val::Percent(100.0),
                        flex_direction: FlexDirection::Row,
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        border: UiRect::top(Val::Px(1.0)),
                        ..default()
                    }),
                )
                .with_children(|builder| {
                    for (index, label) in [
                        AnimationLabel::Run,
                        AnimationLabel::Walk,
                        AnimationLabel::Idle,
                        AnimationLabel::Off,
                    ]
                    .iter()
                    .enumerate()
                    {
                        builder
                            .spawn((
                                Button,
                                BackgroundColor(if index > 0 {
                                    Color::BLACK
                                } else {
                                    Color::WHITE
                                }),
                                Node {
                                    flex_grow: 1.0,
                                    border: if index > 0 {
                                        UiRect::left(Val::Px(1.0))
                                    } else {
                                        UiRect::ZERO
                                    },
                                    ..default()
                                },
                                BorderColor(Color::WHITE),
                                AnimationControl {
                                    group_id: mask_group_id,
                                    label: *label,
                                },
                            ))
                            .with_child((
                                Text(format!("{:?}", label)),
                                if index > 0 {
                                    button_text_style.clone()
                                } else {
                                    selected_button_text_style.clone()
                                },
                                TextLayout::new_with_justify(JustifyText::Center),
                                Node {
                                    flex_grow: 1.0,
                                    margin: UiRect::vertical(Val::Px(3.0)),
                                    ..default()
                                },
                            ));
                    }
                });
        });
}
