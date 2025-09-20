use std::f32::consts::PI;

use bevy::camera::CameraMainTextureUsages;
use bevy::gltf::GltfMaterialName;
///! example follow solari
use bevy::prelude::*;
use bevy::render::render_resource::TextureUsages;
use bevy::scene::SceneInstanceReady;
use bevy::solari::pathtracer::{Pathtracer, PathtracingPlugin};
use bevy::solari::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SolariPlugins)
        .add_plugins(PathtracingPlugin)
        .add_systems(Startup, setup)
        // .add_systems(Update, get_camera_position)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn((
            SceneRoot(
                asset_server
                    .load(GltfAssetLabel::Scene(0).from_asset("pica_pica/mini_diorama_01.glb")),
            ),
            Transform::from_scale(Vec3::splat(10.0)),
        ))
        .observe(add_raytracing_meshes_on_scene_load);

    commands
        .spawn((
            SceneRoot(
                asset_server.load(GltfAssetLabel::Scene(0).from_asset("pica_pica/robot_01.glb")),
            ),
            Transform::from_scale(Vec3::splat(2.0))
                .with_translation(Vec3::new(-2.0, 0.05, -2.1))
                .with_rotation(Quat::from_rotation_y(PI / 2.0)),
        ))
        .observe(add_raytracing_meshes_on_scene_load);

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.219417, 2.5764852, 6.9718704)).with_rotation(
            Quat::from_xyzw(-0.1466768, 0.013738206, 0.002037309, 0.989087),
        ),
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        Pathtracer::default(),
    ));
}

fn get_camera_position(camera: Single<&Transform, With<Camera3d>>) {
    let transform = *camera;
    println!(
        "camera transform is {:?}, rotation is {:?}",
        transform.translation, transform.rotation
    );
}

#[derive(Resource)]
struct RobotLightMaterial(Handle<StandardMaterial>);

fn add_raytracing_meshes_on_scene_load(
    scene_ready: On<SceneInstanceReady>,
    children: Query<&Children>,
    mesh_query: Query<(
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        Option<&GltfMaterialName>,
    )>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
) {
    for descendant in children.iter_descendants(scene_ready.entity) {
        if let Ok((Mesh3d(mesh_handle), MeshMaterial3d(material_handle), material_name)) =
            mesh_query.get(descendant)
        {
            commands
                .entity(descendant)
                .insert(RaytracingMesh3d(mesh_handle.clone()));

            let mesh = meshes.get_mut(mesh_handle).unwrap();
            if !mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0) {
                let vertex_count = mesh.count_vertices();
                mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertex_count]);
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_TANGENT,
                    vec![[0.0, 0.0, 0.0, 0.0]; vertex_count],
                );
            }

            if !mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT) {
                mesh.generate_tangents().unwrap();
            }

            if mesh.contains_attribute(Mesh::ATTRIBUTE_UV_1) {
                mesh.remove_attribute(Mesh::ATTRIBUTE_UV_1);
            }

            commands.entity(descendant).remove::<Mesh3d>();

            if material_name.map(|s| s.0.as_str()) == Some("material") {
                let material = materials.get_mut(material_handle).unwrap();
                material.emissive = LinearRgba::BLACK;
            }
            if material_name.map(|s| s.0.as_str()) == Some("Lights") {
                let material = materials.get_mut(material_handle).unwrap();
                material.emissive =
                    LinearRgba::from(Color::srgb(0.941, 0.714, 0.043)) * 1_000_000.0;
                material.alpha_mode = AlphaMode::Opaque;
                material.specular_transmission = 0.0;

                commands.insert_resource(RobotLightMaterial(material_handle.clone()));
            }
            if material_name.map(|s| s.0.as_str()) == Some("Glass_Dark_01") {
                let material = materials.get_mut(material_handle).unwrap();
                material.alpha_mode = AlphaMode::Opaque;
                material.specular_transmission = 0.0;
            }
        }
    }
}
