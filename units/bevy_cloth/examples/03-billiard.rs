use bevy::{color::palettes::css::WHITE, prelude::*, window::WindowResized};
use rand::Rng;

// const BOX_WIDTH: f32 = 300.0;
// const BOX_HEIGHT: f32 = 200.0;

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
    let mut rng = rand::rng();

    for _i in 0..5 {
        let rand_x: f32 = rng.random_range(0.0..100.0);
        let rand_y: f32 = rng.random_range(0.0..100.0);

        let vel_x: f32 = rng.random_range(0.0..15.0);
        let vel_y: f32 = rng.random_range(0.0..15.0);
        let radius: f32 = rng.random_range(2.0..5.0);

        commands.spawn((
            Mesh2d(meshes.add(Circle::new(2.0))),
            MeshMaterial2d(materials.add(Color::from(WHITE))),
            Transform::from_xyz(rand_x, rand_y, 0.0),
            Ball {
                mass: radius * radius * radius,
                radius,
                vel: Vec2::new(vel_x, vel_y),
            },
        ));
    }

    commands.spawn(Resolution::default());
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

#[derive(Component, Clone)]
struct Ball {
    mass: f32,
    radius: f32,
    vel: Vec2,
}

fn simulate(
    mut ball_query: Query<(&mut Transform, &mut Ball)>,
    resolution: Single<&Resolution>,
    time: Res<Time>,
    gravity: Res<Gravity>,
) {
    let mut all_ball = ball_query.iter_mut().collect::<Vec<_>>();

    for i in 0..all_ball.len() {
        for j in (i + 1)..all_ball.len() {
            let (left, right) = all_ball.split_at_mut(j);
            let (ball1_transform, ball1) = &mut left[i];
            let (ball2_transform, ball2) = &mut right[0];

            let dir = (ball1_transform.translation - ball2_transform.translation).xy();
            let d = dir.length();
            if d < 1e-7 || d > ball1.radius + ball2.radius {
                continue;
            }

            let dir = dir.normalize();

            let corr = (ball1.radius + ball2.radius - d) / 2.0;
            ball1_transform.translation += (corr * dir).extend(0.0);
            ball2_transform.translation += (-corr * dir).extend(0.0);

            let v1 = ball1.vel.dot(dir);
            let v2 = ball1.vel.dot(dir);

            let m1 = ball1.mass;
            let m2 = ball2.mass;

            let new_v1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2)) / (m1 + m2);
            let new_v2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1)) / (m1 + m2);

            ball1.vel += dir * (new_v1 - v1);
            ball2.vel += dir * (new_v2 - v2);
        }

        let (ball_transform, ball) = &mut all_ball[i];

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
}

fn on_resize_system(
    mut resolution: Single<&mut Resolution>,
    mut resize_reader: MessageReader<WindowResized>,
) {
    for m in resize_reader.read() {
        resolution.0 = Vec2::new(m.width, m.height)
    }
}
