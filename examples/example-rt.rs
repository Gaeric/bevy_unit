use bevy::{
    camera::CameraMainTextureUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    gltf::GltfMaterialName,
    mesh::{Indices, MeshVertexAttributeId},
    prelude::*,
    render::render_resource::TextureUsages,
    scene::SceneInstanceReady,
    solari::{SolariPlugins, prelude::SolariLighting, scene::RaytracingMesh3d},
};

fn main() {
    let mut app = App::new();

    app.add_plugins((DefaultPlugins, SolariPlugins, FreeCameraPlugin));

    app.add_systems(Startup, setup_pica_pica);

    app.run();
}

fn setup_pica_pica(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let sphere = meshes.add(Sphere::default());
    let material = materials.add(StandardMaterial::default());

    let mesh = meshes.get_mut(&sphere).unwrap();
    mesh.generate_tangents().unwrap();

    info!("sphere is {:?}", sphere);

    // commands.spawn((
    //     Mesh3d(sphere.clone()),
    //     RaytracingMesh3d(sphere),
    //     MeshMaterial3d(material),
    // ));

    commands
        .spawn((SceneRoot(asset_server.load(
            GltfAssetLabel::Scene(0).from_asset("materials/hs2_body_greybox_mini_2.glb"),
        )),))
        .observe(add_raytracing_meshes_on_scene_load);

    // commands
    //     .spawn((
    //         SceneRoot(
    //             asset_server.load(GltfAssetLabel::Scene(0).from_asset("mini_diorama_01.glb")),
    //         ),
    //         Transform::from_scale(Vec3::splat(10.0)),
    //     ))
    //     .observe(add_raytracing_meshes_on_scene_load);

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::FULL_DAYLIGHT,
            shadows_enabled: false, // Solari replaces shadow mapping
            ..default()
        },
        Transform::from_rotation(Quat::from_xyzw(
            -0.13334629,
            -0.86597735,
            -0.3586996,
            0.3219264,
        )),
    ));

    let mut camera = commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        FreeCamera {
            walk_speed: 3.0,
            run_speed: 10.0,
            ..Default::default()
        },
        Transform::from_translation(Vec3::new(0.219417, 2.5764852, 6.9718704)).with_rotation(
            Quat::from_xyzw(-0.1466768, 0.013738206, 0.002037309, 0.989087),
        ),
        // Msaa::Off and CameraMainTextureUsages with STORAGE_BINDING are required for Solari
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
    ));

    camera.insert(SolariLighting::default());
}

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
    let material = materials.add(StandardMaterial::default());

    for descendant in children.iter_descendants(scene_ready.entity) {
        if let Ok((Mesh3d(mesh_handle), MeshMaterial3d(material_handle), material_name)) =
            mesh_query.get(descendant)
        {
            // info!("mesh_handle is {:?}", mesh_handle);
            // Add raytracing mesh component
            commands
                .entity(descendant)
                .insert(MeshMaterial3d(material.clone()))
                .insert(RaytracingMesh3d(mesh_handle.clone()));

            // Ensure meshes are Solari compatible
            let mesh = meshes.get_mut(mesh_handle).unwrap();

            if !mesh.contains_attribute(Mesh::ATTRIBUTE_UV_0) {
                let vertex_count = mesh.count_vertices();
                mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; vertex_count]);
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_TANGENT,
                    vec![[0.0, 0.0, 0.0, 0.0]; vertex_count],
                );
            }
            // if !mesh.contains_attribute(Mesh::ATTRIBUTE_TANGENT) {
            info!("generate tangets for mesh {:?}", mesh_handle);
            mesh.generate_tangents().unwrap();
            // }
            if mesh.contains_attribute(Mesh::ATTRIBUTE_UV_1) {
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
