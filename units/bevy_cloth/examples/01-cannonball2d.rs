use bevy::{color::palettes::css::WHITE, prelude::*, window::WindowResized};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<Gravity>()
        .add_systems(Startup, setup)
        .add_systems(Update, on_resize_system)
        .add_systems(FixedUpdate, simulate)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2d);

    commands.spawn((
        Mesh2d(meshes.add(Circle::new(2.0))),
        MeshMaterial2d(materials.add(Color::from(WHITE))),
        Transform::default(),
        Ball {
            vel: Vec2::new(10.0, 15.0),
        },
        Resolution::default(),
    ));
}

#[derive(Resource)]
struct Gravity(Vec2);

impl FromWorld for Gravity {
    fn from_world(_: &mut World) -> Self {
        Gravity(Vec2 { x: 0.0, y: -10.0 })
    }
}

#[derive(Component, Default)]
struct Resolution(Vec2);

#[derive(Component)]
struct Ball {
    vel: Vec2,
}

fn simulate(
    mut ball: Single<&mut Ball>,
    mut ball_transform: Single<&mut Transform, With<Ball>>,
    resolution: Single<&Resolution>,
    time: Res<Time>,
    gravity: Res<Gravity>,
) {
    info!(
        "resolution: {}, ball transform: {:?}",
        resolution.0, ball_transform.translation
    );

    let top = resolution.0.y / 2.0;
    let bottom = -resolution.0.y / 2.0;
    let left = -resolution.0.x / 2.0;
    let right = resolution.0.x / 2.0;

    let dl = time.delta_secs();

    ball.vel.x += gravity.0.x * dl;
    ball.vel.y += gravity.0.y * dl;

    ball_transform.translation.x += ball.vel.x * dl;
    ball_transform.translation.y += ball.vel.y * dl;

    if ball_transform.translation.x < left {
        ball_transform.translation.x = left;
        ball.vel.x = -ball.vel.x
    }

    if ball_transform.translation.x > right {
        ball_transform.translation.x = right;
        ball.vel.x = -ball.vel.x
    }

    if ball_transform.translation.y < bottom {
        ball_transform.translation.y = bottom;
        ball.vel.y = -ball.vel.y
    }

    if ball_transform.translation.y > top {
        ball_transform.translation.y = top;
        ball.vel.y = -ball.vel.y
    }
}

fn on_resize_system(
    mut resolution: Single<&mut Resolution>,
    mut resize_reader: MessageReader<WindowResized>,
) {
    for m in resize_reader.read() {
        resolution.0 = Vec2::new(m.width, m.height)
    }
}
