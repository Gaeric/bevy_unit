/// Todo! this plugin not complete

use avian3d::prelude::{Collider, RigidBody};
use bevy::{prelude::*, render::mesh};
use bevy_silk::prelude::*;

pub(crate) fn plugin(app: &mut App) {
    app.insert_resource(ClothMovement { sign: -1.0, t: 0.0 })
        .add_systems(Startup, setup)
        // .add_systems(Update, move_cloth)
        // .add_plugins(ClothPlugin)
    ;
}

#[derive(Debug, Resource)]
struct ClothMovement {
    sign: f32,
    t: f32,
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    asset_server: Res<AssetServer>,
) {
    let flag_texture = asset_server.load("Bevy.png");
    let (size_x, size_y) = (15, 10);
    // let cloth = ClothBuilder::new()
    //     .with_pinned_vertex_ids(0..size_x)
    //     .with_stick_generation(bevy_silk::stick::StickGeneration::Triangles);
    // let mesh = rectangle_mesh((size_x, size_y), (-Vec3::X * 0.2, -Vec3::Y * 0.2), Vec3::Z);
    commands.spawn((
        PbrBundle {
            // mesh: meshes.add(mesh),
            material: materials.add(StandardMaterial {
                base_color_texture: Some(flag_texture),
                cull_mode: None,
                double_sided: true,
                ..default()
            }),
            transform: Transform::from_xyz(1.0, 4.0, 2.0),
            ..default()
        },
        // cloth,
        // ClothCollider {
        //     dampen_others: Some(0.02),
        //     ..default()
        // },
        Name::new("Cloth"),
    ));
}

fn move_cloth(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<ClothBuilder>>,
    mut movement: ResMut<ClothMovement>,
) {
    let delta_time = time.delta_seconds();
    for mut transform in query.iter_mut() {
        movement.t += delta_time * 2.0;
        transform.translation.z += movement.sign * delta_time * 2.0;
        if movement.t > 8.0 {
            movement.t = 0.0;
            movement.sign = -movement.sign;
        }
    }
}
