use bevy::{
    feathers::{theme::ThemeBackgroundColor, tokens},
    input_focus::tab_navigation::TabGroup,
    prelude::*,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, 2.0, 3.0).looking_at(Vec3::new(0.0, -0.5, 0.0), Vec3::Y),
    ));

    let cube = meshes.add(Cuboid::new(0.5, 0.5, 0.5));
    commands.spawn((
        Mesh3d(cube.clone()),
        MeshMaterial3d(materials.add(Color::from(Hsla::hsl(300.0, 1.0, 0.5)))),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
    ));

    commands.spawn((
        Node {
            width: percent(100),
            height: percent(100),
            align_items: AlignItems::Start,
            justify_content: JustifyContent::Start,
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            ..default()
        },
        TabGroup::default(),
        ThemeBackgroundColor(tokens::WINDOW_BG),
        children![],
    ));
}

fn update_materials(
    material_handles: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for material_handle in material_handles.iter() {
        if let Some(material) = materials.get_mut(material_handle)
            && let Color::Hsla(ref mut hsla) = material.base_color
        {
            *hsla = hsla.rotate_hue(1.0 / 60.0 * 100.0);
        }
    }
}
