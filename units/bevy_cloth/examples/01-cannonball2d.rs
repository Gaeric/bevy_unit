use bevy::{color::palettes::css::PURPLE, prelude::*};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2d);

    commands.spawn((
        Mesh2d(meshes.add(Circle::new(0.2))),
        MeshMaterial2d(materials.add(Color::from(PURPLE))),
        Transform::default(),
    ));
}

#[derive(Component)]
struct Ball {
    radius: f32,
    vel: Vec2,
}

fn simulate(ball: Single<&mut Transform, With<Ball>>) {}
