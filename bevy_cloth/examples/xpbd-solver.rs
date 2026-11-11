use bevy::prelude::*;
use bevy_cloth::xpbd::{Attachment, ClothStepSettings, StretchCloth, XpbdPlugin};

const SEGMENTS: u32 = 16;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "xpbd solver".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(XpbdPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let (mesh, attachments) = cloth_input();
    let marker = meshes.add(Sphere::new(0.035));
    let marker_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.1, 0.05),
        unlit: true,
        ..default()
    });
    for attachment in &attachments {
        commands.spawn((
            Mesh3d(marker.clone()),
            MeshMaterial3d(marker_material.clone()),
            Transform::from_translation(attachment.position),
        ));
        info!(
            particle = attachment.particle,
            target = ?attachment.position,
            "fixed attachment input (solver not connected)"
        );
    }

    commands.spawn((
        Name::new("Stretch cloth"),
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.55, 0.8),
            cull_mode: None,
            double_sided: true,
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::IDENTITY,
        StretchCloth {
            particle_mass: 1.0,
            stretch_compliance: 0.0001,
            attachments,
        },
        ClothStepSettings {
            gravity: Vec3::new(0.0, -9.8, 0.0),
            substeps: 6,
            iterations: 10,
        },
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.6, 2.0, 5.0).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(3.0, 5.0, 4.0).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
    ));
}

/// Example geometry only; it does not generate solver constraints or GPU buffers.
fn cloth_input() -> (Mesh, Vec<Attachment>) {
    let mesh = Plane3d::default()
        .mesh()
        .size(2.0, 2.0)
        .subdivisions(SEGMENTS - 1)
        .build()
        .transformed_by(Transform {
            translation: Vec3::new(0.0, 1.5, 0.0),
            rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            ..default()
        });
    let positions = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|attribute| attribute.as_float3())
        .expect("generated plane must have Float32x3 positions");
    let attachments = [-1.0, 1.0]
        .map(|x| {
            let position = Vec3::new(x, 2.5, 0.0);
            let particle = positions
                .iter()
                .position(|vertex| Vec3::from(*vertex).distance(position) < 1e-5)
                .expect("generated plane must contain the top corner")
                as u32;
            Attachment { particle, position }
        })
        .to_vec();
    (mesh, attachments)
}
