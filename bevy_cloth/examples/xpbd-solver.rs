use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
};
use bevy_cloth::{
    Cloth, ClothExternalFriction, ClothLongRangeAttachments, ClothPlugin, ClothSelfCollision,
    ClothSolver, ClothStepSettings, CollisionGroup, FixedAttachment, MeshCollider,
};

const SEGMENTS: u32 = 16;

#[derive(Component)]
struct MovingSphere;

#[derive(Resource, Default)]
struct DemoOptions {
    compare: bool,
    lra: bool,
    friction: bool,
    self_collision: bool,
}

fn main() {
    let mut options = DemoOptions::default();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--compare" => options.compare = true,
            "--lra" => options.lra = true,
            "--friction" => options.friction = true,
            "--self-collision" => options.self_collision = true,
            _ => {}
        }
    }
    App::new()
        .insert_resource(options)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "xpbd solver".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((ClothPlugin, FreeCameraPlugin))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, move_sphere)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    options: Res<DemoOptions>,
) {
    spawn_case(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut materials,
        ClothSolver::ColoredGs,
        if options.compare { -1.2 } else { 0.0 },
        &options,
    );
    if options.compare {
        spawn_case(
            &mut commands,
            &asset_server,
            &mut meshes,
            &mut materials,
            ClothSolver::Jacobi,
            1.2,
            &options,
        );
    }
    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Transform::from_xyz(0.6, 2.0, if options.compare { 7.0 } else { 5.0 })
            .looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(3.0, 5.0, 4.0).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
    ));
}

fn spawn_case(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    solver: ClothSolver,
    x: f32,
    options: &DemoOptions,
) {
    // In this demo each solver has its own collider.
    let group = match solver {
        ClothSolver::ColoredGs => CollisionGroup(0),
        ClothSolver::Jacobi => CollisionGroup(1),
    };
    let (mesh, mut attachments) = cloth_input();
    for attachment in &mut attachments {
        attachment.position.x += x;
    }
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
            "fixed attachment target"
        );
    }

    let mut cloth = commands.spawn((
        Name::new(format!("{solver:?} cloth")),
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.5, 1.0),
            base_color_texture: Some(asset_server.load("velvet/fabric2.jpg")),
            cull_mode: None,
            double_sided: true,
            perceptual_roughness: 1.0,
            reflectance: 0.01,
            ..default()
        })),
        Transform::from_xyz(x, 0.0, 0.0),
        solver,
        group,
        Cloth {
            particle_mass: 1.0,
            stretch_compliance: 0.0001,
            bend_compliance: 0.0001,
            attachments,
            collision_margin: 0.02,
            double_sided: false,
        },
        ClothStepSettings {
            gravity: Vec3::new(0.0, -9.8, 0.0),
            substeps: 6,
            iterations: 10,
        },
    ));
    if options.lra {
        cloth.insert(ClothLongRangeAttachments { compliance: 0.0 });
    }
    if options.friction {
        cloth.insert(ClothExternalFriction { coefficient: 0.4 });
    }
    if options.self_collision {
        cloth.insert(ClothSelfCollision {
            diameter: 2.0 / SEGMENTS as f32,
            friction: if options.friction { 0.4 } else { 0.0 },
            neighbor_capacity: 32,
        });
    }

    // Initial overlap exercises tick-start stabilization and substep mesh contact.
    commands.spawn((
        Name::new(format!("{solver:?} collider")),
        Mesh3d(meshes.add(Sphere::new(0.5).mesh().uv(32, 18))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_xyz(x, 1.4, -0.35),
        MeshCollider,
        group,
        MovingSphere,
    ));
}

fn move_sphere(time: Res<Time<Fixed>>, mut spheres: Query<&mut Transform, With<MovingSphere>>) {
    let z = -0.35 + 0.6 * (0.8 * time.elapsed_secs()).sin();
    for mut transform in &mut spheres {
        transform.translation.z = z;
    }
}

/// Example geometry only; it does not generate solver constraints or GPU buffers.
fn cloth_input() -> (Mesh, Vec<FixedAttachment>) {
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
            FixedAttachment { particle, position }
        })
        .to_vec();
    (mesh, attachments)
}
