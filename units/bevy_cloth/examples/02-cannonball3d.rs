use bevy::{color::palettes::css::WHITE, prelude::*};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<Gravity>()
        .init_resource::<PhyBox>()
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, simulate)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(2.0))),
        MeshMaterial3d(materials.add(Color::from(WHITE))),
        Ball {
            vel: Vec3::new(10.0, 15.0, 20.0),
        },
    ));
}

#[derive(Resource)]
struct Gravity(Vec3);

impl FromWorld for Gravity {
    fn from_world(_: &mut World) -> Self {
        Gravity(Vec3::NEG_Y * 10.0)
    }
}

#[derive(Resource)]
struct PhyBox {
    length: f32,
    height: f32,
    width: f32,
}

impl FromWorld for PhyBox {
    fn from_world(_: &mut World) -> Self {
        PhyBox {
            length: 10.0,
            height: 10.0,
            width: 10.0,
        }
    }
}

#[derive(Component)]
struct Ball {
    vel: Vec3,
}

fn simulate(
    mut ball: Single<&mut Ball>,
    mut ball_transform: Single<&mut Transform, With<Ball>>,
    time: Res<Time>,
    gravity: Res<Gravity>,
    phy_box: Res<PhyBox>,
) {
    let top = phy_box.height / 2.0;
    let bottom = -phy_box.height / 2.0;
    let left = -phy_box.length / 2.0;
    let right = phy_box.length / 2.0;
    let front = phy_box.width / 2.0;
    let back = -phy_box.width / 2.0;

    let dl = time.delta_secs();

    ball.vel.x += gravity.0.x * dl;
    ball.vel.y += gravity.0.y * dl;
    ball.vel.z += gravity.0.z * dl;

    ball_transform.translation.x += ball.vel.x * dl;
    ball_transform.translation.y += ball.vel.y * dl;
    ball_transform.translation.z += ball.vel.z * dl;

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

    if ball_transform.translation.z < back {
        ball_transform.translation.z = back;
        ball.vel.z = -ball.vel.z
    }

    if ball_transform.translation.z > front {
        ball_transform.translation.z = front;
        ball.vel.z = -ball.vel.z
    }
}
