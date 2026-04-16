use bevy::{
    app::{App, Plugin},
    camera::CameraMainTextureUsages,
    ecs::{hierarchy::Children, system::Query},
    gltf::GltfMaterialName,
    mesh::{Indices, Mesh3d},
    pbr::{MeshMaterial3d, StandardMaterial},
    prelude::*,
    render::render_resource::TextureUsages,
    scene::SceneInstanceReady,
    solari::{SolariPlugins, prelude::SolariLighting, scene::RaytracingMesh3d},
};

fn add_raytracing_meshes_on_scene_load(
    scene_ready: On<SceneInstanceReady>,
    children: Query<&Children>,
    mesh_query: Query<(
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        &GltfMaterialName,
    )>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut commands: Commands,
    _asset_server: Res<AssetServer>,
) {
    let sphere = meshes.add(Sphere::default());
    let material = materials.add(StandardMaterial::default());

    commands.spawn((
        Mesh3d(sphere.clone()),
        MeshMaterial3d(material),
        RaytracingMesh3d(sphere),
    ));

    for descendant in children.iter_descendants(scene_ready.entity) {
        if let Ok((Mesh3d(mesh_handle), MeshMaterial3d(material_handle), mat_name)) =
            mesh_query.get(descendant)
        {
            info!("entity {:?} material name {}", material_handle, mat_name.0);
            commands
                .entity(descendant)
                .insert(RaytracingMesh3d(mesh_handle.clone()));

            // Ensure meshes are Solari compatible
            let mesh = meshes.get_mut(mesh_handle).unwrap();
            if !mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0) {
                info!("mesh uv0 miss.");
            }

            if !mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT) {
                info!("mesh tangent miss.");
                mesh.generate_tangents().unwrap();
            }

            if mesh.contains_attribute(Mesh::ATTRIBUTE_UV_1) {
                warn!("mesh uv1 is removed");
                mesh.remove_attribute(Mesh::ATTRIBUTE_UV_1);
            }

            if mesh.contains_attribute(Mesh::ATTRIBUTE_COLOR) {
                mesh.remove_attribute(Mesh::ATTRIBUTE_COLOR);
            }

            if let Some(indices) = mesh.indices() {
                let new_indices: Vec<u32> = indices.iter().map(|i| i as u32).collect();
                mesh.insert_indices(Indices::U32(new_indices));
            }

            if let Some(indices) = mesh.indices() {
                let new_indices: Vec<u32> = indices.iter().map(|i| i as u32).collect();
                mesh.insert_indices(Indices::U32(new_indices));
            }

            let vertex_attributes = mesh.attributes().map(|(attribute, _)| attribute.id);
            let indexed_32 = matches!(mesh.indices(), Some(Indices::U32(..)));
            info!("mesh indexed 32? {}", indexed_32);
            for attribute in vertex_attributes {
                info!("mesh attributes {:?}", attribute);
            }
        }
    }
}

fn added_camera_rt_params(camera: On<Add, Camera3d>, mut commands: Commands) {
    commands.entity(camera.entity).insert((
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
        SolariLighting::default(),
    ));
}

pub struct DemoRTPlugin;

impl Plugin for DemoRTPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SolariPlugins);
        app.add_observer(add_raytracing_meshes_on_scene_load)
            .add_observer(added_camera_rt_params);
    }
}
