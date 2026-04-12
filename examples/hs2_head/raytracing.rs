use bevy::{
    app::{App, Plugin},
    ecs::{hierarchy::Children, system::Query},
    gltf::GltfMaterialName,
    mesh::Mesh3d,
    pbr::{MeshMaterial3d, StandardMaterial},
    prelude::*,
    scene::SceneInstanceReady,
    solari::{SolariPlugins, scene::RaytracingMesh3d},
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
    asset_server: Res<AssetServer>,
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
            let new_material = StandardMaterial::default();
            let handle = asset_server.add(new_material);
            commands.entity(descendant).insert(MeshMaterial3d(handle));
        }
    }
}

pub struct DemoRTPlugin;

impl Plugin for DemoRTPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SolariPlugins);
        app.add_observer(add_raytracing_meshes_on_scene_load);
    }
}
