//! Actual model: skirt by default; --top also simulates the top; --skirt-only is explicit default.
//! Space pause, S paused step, R reset paused, B rebuild collider BVH; middle drag/wheel orbit.
//! --motion rotates torso and pin targets about the same prepared world pivot.
//! Top/skirt share particle intercollision (not triangle contact or CCD). L reloads; T toggles top.

use bevy::{
    asset::RenderAssetUsages,
    gltf::{GltfLoaderSettings, GltfMaterialName, convert_coordinates::GltfConvertCoordinates},
    prelude::*,
    world_serialization::WorldInstanceReady,
};
use bevy_cloth::{
    ClothAttachmentMotion, ClothControl, ClothMeshCollider, ClothPhysicalConfig, ClothSim,
    ClothSimulationGroup, ClothSolverConfig, ColliderMotion, CollisionGroup, GpuClothPlugin,
    MeshColliderPreparation, MeshRebuildRequests, OrbitCameraPlugin, demo::OrbitCamera,
};
use std::{collections::HashMap, path::Path};

#[derive(Component)]
struct ModelRoot;

#[derive(Component)]
struct OptionalTop;

#[derive(Component, Clone)]
struct GarmentConfig {
    cloth: ClothSim,
    physical: ClothPhysicalConfig,
    solver: ClothSolverConfig,
}

#[derive(Component)]
struct AwaitRigidPins;

fn torso_motion() -> ColliderMotion {
    ColliderMotion {
        angular_velocity: Vec3::Y * 0.4,
        ..default()
    }
}

fn main() {
    for arg in std::env::args().skip(1) {
        assert!(
            matches!(
                arg.as_str(),
                "--top" | "--skirt-only" | "--motion" | "--join-render-splits"
            ),
            "unknown option: {arg}"
        );
    }
    assert!(
        !(flag("--top") && flag("--skirt-only")),
        "choose --top or --skirt-only"
    );
    App::new()
        .add_plugins(
            DefaultPlugins.set(AssetPlugin {
                file_path: Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("workspace parent")
                    .join("assets")
                    .to_string_lossy()
                    .into_owned(),
                ..default()
            }),
        )
        .add_plugins((GpuClothPlugin, OrbitCameraPlugin))
        .add_observer(select_roles)
        .add_systems(Startup, setup)
        .add_systems(PreUpdate, controls.after(bevy::input::InputSystems))
        .add_systems(Update, bind_rigid_pins)
        .run();
}

fn flag(name: &str) -> bool {
    // Example unit tests exercise deterministic, non-moving controls without CLI flags.
    !cfg!(test) && std::env::args().any(|arg| arg == name)
}

fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    let scene = assets
        .load_builder()
        .with_settings(|settings: &mut GltfLoaderSettings| {
            settings.load_meshes = RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD;
            // This asset's node quaternions already establish Y-up; do not rotate it twice.
            settings.convert_coordinates = Some(GltfConvertCoordinates {
                rotate_scene_entity: false,
                rotate_meshes: false,
            });
        })
        .load(GltfAssetLabel::Scene(0).from_asset("materials/hs2_body_greybox_mini_2.glb"));
    commands.spawn((
        WorldAssetRoot(scene),
        ModelRoot,
        Transform::from_scale(Vec3::splat(10.0)),
    ));
    let mut orbit = OrbitCamera::source();
    orbit.target = Vec3::new(0.0, 10.0, 0.0);
    let offset = Vec3::new(0.0, 12.0, 28.0) - orbit.target;
    orbit.distance = offset.length();
    orbit.yaw = 0.0;
    orbit.pitch = (offset.y / orbit.distance).asin();
    commands.spawn((Camera3d::default(), orbit.transform(), orbit));
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(8.0, 20.0, 12.0).looking_at(orbit.target, Vec3::Y),
    ));
    info!(
        "Space pause / S step / R reset / B rebuild / L reload / T toggle top; shared particle intercollision"
    );
}

fn material_role(name: &str) -> Option<usize> {
    match name {
        "Torso_" => Some(0),
        "clothing_" => Some(1),
        "clothing_.004" => Some(2),
        _ => None,
    }
}

fn select_roles(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    roots: Query<(), With<ModelRoot>>,
    children: Query<&Children>,
    primitives: Query<(&Mesh3d, &GltfMaterialName)>,
    meshes: Res<Assets<Mesh>>,
) {
    if !roots.contains(ready.entity) {
        return;
    }
    let mut selected = [0; 3];
    for entity in children.iter_descendants(ready.entity) {
        let Ok((handle, material)) = primitives.get(entity) else {
            continue;
        };
        let Some(role) = material_role(&material.0) else {
            continue;
        };
        selected[role] += 1;
        commands.entity(entity).insert(CollisionGroup(0));
        if role == 0 {
            commands.entity(entity).insert(ClothMeshCollider);
            if flag("--motion") {
                commands.entity(entity).insert(torso_motion());
            }
            info!(?entity, material = %material.0, "model collider selected");
            continue;
        }
        let mesh = meshes.get(&handle.0).expect("loaded model mesh");
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|attribute| attribute.as_float3())
            .expect("model Float32x3 POSITION");
        let pins = upper_band_pins(positions);
        assert!(!pins.is_empty() && pins.len() < positions.len());
        // Application opt-in for THIS dataset, not a general weld policy: exact-position
        // render splits within this primitive only. No proximity search or cross-garment pairs.
        let seams = if flag("--join-render-splits") {
            exact_split_pairs(positions)
        } else {
            Vec::new()
        };
        let mut solver = ClothSolverConfig::default();
        solver.params.particle_diameter = 0.03;
        solver.params.collision_margin = 0.025;
        solver.params.max_speed = 20.0;
        solver.params.bend_compliance = 0.001;
        solver.params.num_substeps = 4;
        solver.params.num_iterations = 6;
        info!(?entity, material = %material.0, vertices = positions.len(),
            pin_count = pins.len(), pin_ids = ?pins, seam_pairs = seams.len(),
            config = ?solver, "model cloth selected (world-space solver units)");
        let config = GarmentConfig {
            cloth: ClothSim {
                pinned_vertices: pins,
            },
            physical: ClothPhysicalConfig { seams, ..default() },
            solver,
        };
        commands
            .entity(entity)
            .insert((ClothSimulationGroup(0), config.clone()));
        if role == 2 {
            commands.entity(entity).insert(OptionalTop);
        }
        if role == 1 || flag("--top") {
            if flag("--motion") {
                // Do not expose ClothSim until the shared provider pivot is ready.
                commands.entity(entity).insert(AwaitRigidPins);
            } else {
                commands
                    .entity(entity)
                    .insert((config.cloth, config.physical, config.solver));
            }
        }
    }
    assert_eq!(selected, [1, 1, 1]);
}

fn bind_rigid_pins(
    mut commands: Commands,
    colliders: Query<&MeshColliderPreparation, With<ClothMeshCollider>>,
    garments: Query<(Entity, &GarmentConfig), With<AwaitRigidPins>>,
) {
    // Read preparation, not WorldInstanceReady transforms (propagation may not have run).
    let Some(pivot) = colliders.iter().find_map(|state| match state {
        MeshColliderPreparation::Ready(prepared) => {
            Some(prepared.initial_transform.transform_point3(Vec3::ZERO))
        }
        _ => None,
    }) else {
        return;
    };
    for (entity, config) in &garments {
        commands
            .entity(entity)
            .insert((
                ClothAttachmentMotion {
                    motion: torso_motion(),
                    pivot,
                },
                config.cloth.clone(),
                config.physical.clone(),
                config.solver.clone(),
            ))
            .remove::<AwaitRigidPins>();
    }
}

fn upper_band_pins(positions: &[[f32; 3]]) -> Vec<u32> {
    // WorldInstanceReady precedes transform propagation. In this asset -local Z is
    // world up: select the highest 3% of each garment (waistband / upper shoulders).
    let low = positions.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min);
    let high = positions
        .iter()
        .map(|p| p[2])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(low.is_finite() && high.is_finite() && high > low);
    let threshold = low + 0.03 * (high - low);
    positions
        .iter()
        .enumerate()
        .filter_map(|(id, p)| (p[2] <= threshold).then_some(id as u32))
        .collect()
}

fn exact_split_pairs(positions: &[[f32; 3]]) -> Vec<[u32; 2]> {
    let mut first = HashMap::new();
    let mut pairs = Vec::new();
    for (id, position) in positions.iter().enumerate() {
        let key = position.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
        let representative = *first.entry(key).or_insert(id as u32);
        if representative != id as u32 {
            pairs.push([representative, id as u32]);
        }
    }
    pairs
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    mut control: ResMut<ClothControl>,
    mut rebuild: ResMut<MeshRebuildRequests>,
    mut commands: Commands,
    roots: Query<(Entity, &WorldAssetRoot), With<ModelRoot>>,
    tops: Query<(Entity, &GarmentConfig, Has<ClothSim>, Has<AwaitRigidPins>), With<OptionalTop>>,
) {
    if keys.just_pressed(KeyCode::KeyL) {
        // Despawn the complete scene instance; replacement role discovery starts from rest.
        for (entity, root) in &roots {
            commands.entity(entity).despawn();
            commands.spawn((
                WorldAssetRoot(root.0.clone()),
                ModelRoot,
                Transform::from_scale(Vec3::splat(10.0)),
            ));
        }
        control.paused = true;
        control.reset();
        return;
    }
    if keys.just_pressed(KeyCode::KeyT) {
        for (entity, config, active, waiting) in &tops {
            if active || waiting {
                commands
                    .entity(entity)
                    .remove::<ClothSim>()
                    .remove::<ClothAttachmentMotion>()
                    .remove::<AwaitRigidPins>();
            } else if flag("--motion") {
                commands.entity(entity).insert(AwaitRigidPins);
            } else {
                commands.entity(entity).insert((
                    config.cloth.clone(),
                    config.physical.clone(),
                    config.solver.clone(),
                ));
            }
        }
        control.paused = true;
        control.reset();
    }
    if keys.just_pressed(KeyCode::KeyB) {
        rebuild.request_all();
    }
    if keys.just_pressed(KeyCode::KeyR) {
        control.paused = true;
        control.reset();
    } else {
        if keys.just_pressed(KeyCode::Space) {
            control.paused = !control.paused;
        }
        if control.paused && keys.just_pressed(KeyCode::KeyS) {
            control.request_step();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::PrimitiveTopology;
    use bevy_cloth::cloth::prepare_mesh_collider;

    fn garment_config() -> GarmentConfig {
        let mut solver = ClothSolverConfig::default();
        solver.params.particle_diameter = 0.031;
        GarmentConfig {
            cloth: ClothSim {
                pinned_vertices: vec![0, 2],
            },
            physical: ClothPhysicalConfig {
                seams: vec![[0, 1]],
                inverse_masses: Some(vec![0.0, 1.5, 0.0]),
            },
            solver,
        }
    }

    fn controls_app() -> App {
        let mut app = App::new();
        app.init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ClothControl>()
            .init_resource::<MeshRebuildRequests>()
            .add_systems(Update, controls);
        app
    }

    fn press(app: &mut App, key: KeyCode) {
        // No InputPlugin: explicitly advance the input state between presses.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(key);
        app.update();
    }

    fn assert_reset_paused(app: &App) {
        let control = app.world().resource::<ClothControl>();
        assert!(control.paused);
        assert_eq!(control.single_steps, 0);
    }

    fn assert_config_installed(app: &App, entity: Entity, config: &GarmentConfig) {
        assert_eq!(
            app.world().get::<ClothSim>(entity).unwrap().pinned_vertices,
            config.cloth.pinned_vertices
        );
        let physical = app.world().get::<ClothPhysicalConfig>(entity).unwrap();
        assert_eq!(physical.seams, config.physical.seams);
        assert_eq!(physical.inverse_masses, config.physical.inverse_masses);
        assert_eq!(
            app.world()
                .get::<ClothSolverConfig>(entity)
                .unwrap()
                .params
                .particle_diameter,
            config.solver.params.particle_diameter
        );
    }

    #[test]
    fn actual_glb_garments_prepare_with_and_without_exact_render_split_seams() {
        use bevy::{gltf::gltf, mesh::Indices};
        use bevy_cloth::cloth::prepare_mesh_with_physical;

        let asset = gltf::Gltf::from_slice(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../assets/materials/hs2_body_greybox_mini_2.glb"
        )))
        .expect("valid embedded model GLB");
        let blob = asset.blob.as_deref().expect("embedded GLB buffer");
        let roots: Vec<_> = asset
            .scenes()
            .next()
            .unwrap()
            .nodes()
            .map(|n| n.index())
            .collect();
        let mut selected = [0; 3];
        for node in asset.nodes() {
            let Some(source_mesh) = node.mesh() else {
                continue;
            };
            for primitive in source_mesh.primitives() {
                let Some(role) = primitive.material().name().and_then(material_role) else {
                    continue;
                };
                selected[role] += 1;
                assert!(
                    roots.contains(&node.index()),
                    "selected node must be a scene root"
                );
                assert_eq!(primitive.mode(), gltf::mesh::Mode::Triangles);
                let reader = primitive.reader(|buffer| {
                    assert_eq!(buffer.index(), 0);
                    assert!(matches!(buffer.source(), gltf::buffer::Source::Bin));
                    Some(blob)
                });
                let positions: Vec<_> = reader.read_positions().unwrap().collect();
                let indices: Vec<_> = reader.read_indices().unwrap().into_u32().collect();
                // Match the root scale and disabled glTF coordinate conversion.
                let transform = Mat4::from_scale(Vec3::splat(10.0))
                    * Mat4::from_cols_array_2d(&node.transform().matrix());
                let world_positions: Vec<_> = positions
                    .iter()
                    .map(|&p| transform.transform_point3(Vec3::from(p)))
                    .collect();
                let mut mesh = Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::MAIN_WORLD,
                )
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone())
                .with_inserted_indices(Indices::U32(indices.clone()));
                if role == 0 {
                    assert_eq!((positions.len(), indices.len() / 3), (17230, 30304));
                    let prepared = prepare_mesh_collider(&mesh, transform)
                        .unwrap_or_else(|error| panic!("Torso_ collider: {error}"));
                    assert_eq!(prepared.positions, world_positions);
                    assert_eq!(prepared.indices, indices);
                    assert_eq!(prepared.initial_transform, transform);
                    let mut triangle_refs = prepared.bvh.triangle_refs.clone();
                    triangle_refs.sort_unstable();
                    assert_eq!(triangle_refs, (0..30304).collect::<Vec<u32>>());
                    continue;
                }
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_NORMAL,
                    reader.read_normals().unwrap().collect::<Vec<_>>(),
                );
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_UV_0,
                    reader
                        .read_tex_coords(0)
                        .unwrap()
                        .into_f32()
                        .collect::<Vec<_>>(),
                );
                mesh.insert_attribute(
                    Mesh::ATTRIBUTE_TANGENT,
                    reader.read_tangents().unwrap().collect::<Vec<_>>(),
                );
                if let Some(colors) = reader.read_colors(0) {
                    // Keep packed normalized colors with their matching vertex format.
                    mesh.insert_attribute(
                        bevy::mesh::MeshVertexAttribute {
                            format: bevy::mesh::VertexFormat::Unorm8x4,
                            ..Mesh::ATTRIBUTE_COLOR
                        },
                        bevy::mesh::VertexAttributeValues::Unorm8x4(
                            colors.into_rgba_u8().collect(),
                        ),
                    );
                }
                let render_attributes: Vec<_> = mesh
                    .attributes()
                    .map(|(attribute, values)| (attribute.id, values.clone()))
                    .collect();
                let pins = upper_band_pins(&positions);
                let seams = exact_split_pairs(&positions);
                let expected = if role == 1 {
                    (1135, 2122, 24, 22, [(3254, 3112, 142), (3257, 3131, 104)])
                } else {
                    (3103, 4622, 22, 749, [(7621, 6245, 1376), (7726, 6889, 88)])
                };
                assert_eq!(
                    (positions.len(), indices.len() / 3, pins.len(), seams.len()),
                    (expected.0, expected.1, expected.2, expected.3)
                );
                for join in [false, true] {
                    let physical = ClothPhysicalConfig {
                        seams: if join { seams.clone() } else { Vec::new() },
                        ..default()
                    };
                    let prepared = prepare_mesh_with_physical(&mesh, transform, &pins, &physical)
                        .unwrap_or_else(|error| panic!("role {role}, join={join}: {error}"));
                    assert_eq!(prepared.positions, world_positions);
                    assert_eq!(prepared.initial_transform, transform);
                    assert_eq!(prepared.indices, indices);
                    assert_eq!(prepared.pinned_vertices, pins);
                    assert_eq!(prepared.inv_masses.len(), positions.len());
                    assert_eq!(
                        prepared.inv_masses.iter().filter(|&&m| m == 0.0).count(),
                        pins.len()
                    );
                    for (id, &mass) in prepared.inv_masses.iter().enumerate() {
                        assert_eq!(
                            mass,
                            if pins.contains(&(id as u32)) {
                                0.0
                            } else {
                                1.0
                            }
                        );
                    }
                    let topology = &prepared.topology;
                    assert_eq!(
                        (
                            topology.stretch.len(),
                            topology.bend.len(),
                            topology.boundary_edges
                        ),
                        expected.4[usize::from(join)]
                    );
                    assert!(topology.stretch.iter().all(|c| c.rest_length.is_finite()));
                    assert!(topology.bend.iter().all(|c| c.rest_angle.is_finite()));
                    assert_eq!(
                        topology
                            .stretch
                            .iter()
                            .filter(|c| c.rest_length == 0.0)
                            .count(),
                        if join { seams.len() } else { 0 }
                    );
                    assert_eq!(topology.stretch_csr.offsets.len(), positions.len() + 1);
                    assert_eq!(
                        topology.stretch_csr.entries.len(),
                        topology.stretch.len() * 2
                    );
                    assert_eq!(topology.bend_csr.offsets.len(), positions.len() + 1);
                    assert_eq!(topology.bend_csr.entries.len(), topology.bend.len() * 4);
                    // Preparation must not replace render indices or any source attributes.
                    assert_eq!(mesh.indices(), Some(&Indices::U32(indices.clone())));
                    for (id, values) in &render_attributes {
                        assert_eq!(mesh.attribute(*id), Some(values));
                    }
                }
            }
        }
        assert_eq!(selected, [1, 1, 1]);
    }

    #[test]
    fn upper_band_pins_uses_negative_local_z_and_includes_band_boundary() {
        let positions = [
            [100.0, 100.0, -100.0],
            [-100.0, -100.0, -99.0],
            [0.0, 0.0, -97.0], // Exactly low + 3% of the 100-unit range.
            [0.0, 1000.0, -96.999],
            [0.0, 2000.0, 0.0],
        ];
        assert_eq!(upper_band_pins(&positions), vec![0, 1, 2]);
    }

    #[test]
    fn exact_split_pairs_normalizes_signed_zero_without_near_welding() {
        let near = f32::from_bits(1.0f32.to_bits() + 1);
        let positions = [
            [0.0, -0.0, 1.0],
            [-0.0, 0.0, 1.0],
            [0.0, 0.0, near], // A single ULP is not an exact split.
            [0.0, 0.0, 1.0],
            [2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0],
        ];
        assert_eq!(exact_split_pairs(&positions), vec![[0, 1], [0, 3], [4, 5]]);
        assert!(exact_split_pairs(&[]).is_empty());
    }

    #[test]
    fn material_roles_require_exact_names() {
        assert_eq!(material_role("Torso_"), Some(0));
        assert_eq!(material_role("clothing_"), Some(1));
        assert_eq!(material_role("clothing_.004"), Some(2));
        for name in [
            "",
            "torso_",
            "Torso",
            "Torso_.001",
            "clothing",
            "clothing_.003",
            "clothing_.004 ",
            "prefix_clothing_",
            "clothing_.004.001",
        ] {
            assert_eq!(material_role(name), None, "unexpected role for {name:?}");
        }
    }

    #[test]
    fn toggle_top_removes_then_readds_and_pauses_resets_each_time() {
        assert!(!flag("--motion"));
        let mut app = controls_app();
        let config = garment_config();
        let top = app
            .world_mut()
            .spawn((
                OptionalTop,
                config.clone(),
                config.cloth.clone(),
                config.physical.clone(),
                config.solver.clone(),
                ClothAttachmentMotion {
                    motion: torso_motion(),
                    pivot: Vec3::ONE,
                },
            ))
            .id();
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        press(&mut app, KeyCode::KeyT);
        assert_reset_paused(&app);
        assert!(app.world().get::<ClothSim>(top).is_none());
        assert!(app.world().get::<ClothAttachmentMotion>(top).is_none());
        assert!(app.world().get::<AwaitRigidPins>(top).is_none());
        assert!(app.world().get::<GarmentConfig>(top).is_some());
        {
            let mut control = app.world_mut().resource_mut::<ClothControl>();
            control.paused = false;
            control.request_step();
        }
        press(&mut app, KeyCode::KeyT);
        assert_reset_paused(&app);
        assert_config_installed(&app, top, &config);
        assert!(app.world().get::<ClothAttachmentMotion>(top).is_none());
        assert!(app.world().get::<AwaitRigidPins>(top).is_none());
    }

    #[test]
    fn toggle_top_cancels_pending_binding() {
        let mut app = controls_app();
        let top = app
            .world_mut()
            .spawn((OptionalTop, garment_config(), AwaitRigidPins))
            .id();
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        press(&mut app, KeyCode::KeyT);
        assert_reset_paused(&app);
        assert!(app.world().get::<AwaitRigidPins>(top).is_none());
        assert!(app.world().get::<ClothSim>(top).is_none());
    }

    #[test]
    fn reload_replaces_scene_root_and_descendants_without_loading() {
        let mut app = controls_app();
        // No asset or scene plugins: exercise replacement, not asynchronous loading.
        let scene = Handle::default();
        let old_root = app
            .world_mut()
            .spawn((
                ModelRoot,
                WorldAssetRoot(scene.clone()),
                Transform::from_xyz(1.0, 2.0, 3.0),
            ))
            .id();
        let child = app
            .world_mut()
            .spawn((
                ChildOf(old_root),
                garment_config(),
                ClothSim::default(),
                OptionalTop,
            ))
            .id();
        let grandchild = app.world_mut().spawn(ChildOf(child)).id();
        let unrelated = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<ClothControl>()
            .request_step();
        // Reload takes priority over simultaneous toggles/step requests.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyT);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyS);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyL);
        app.update();
        assert_reset_paused(&app);
        for old in [old_root, child, grandchild] {
            assert!(app.world().get_entity(old).is_err());
        }
        assert!(app.world().get_entity(unrelated).is_ok());
        let mut roots = app
            .world_mut()
            .query_filtered::<(Entity, &WorldAssetRoot, &Transform), With<ModelRoot>>();
        let replacements: Vec<_> = roots.iter(app.world()).collect();
        assert_eq!(replacements.len(), 1);
        let (entity, root, transform) = replacements[0];
        assert_ne!(entity, old_root);
        assert_eq!(root.0, scene);
        assert_eq!(*transform, Transform::from_scale(Vec3::splat(10.0)));
        assert!(app.world().get::<Children>(entity).is_none());
    }

    #[test]
    fn binding_waits_for_ready_then_installs_matching_pivot_and_motion() {
        let mut app = App::new();
        app.add_systems(Update, bind_rigid_pins);
        let config = garment_config();
        let garments: Vec<_> = (0..2)
            .map(|_| app.world_mut().spawn((config.clone(), AwaitRigidPins)).id())
            .collect();
        let inactive = app.world_mut().spawn(config.clone()).id();
        app.update();
        let collider = app
            .world_mut()
            .spawn((ClothMeshCollider, MeshColliderPreparation::Waiting))
            .id();
        for state in [
            MeshColliderPreparation::Waiting,
            MeshColliderPreparation::Failed("test fixture".into()),
        ] {
            app.world_mut().entity_mut(collider).insert(state);
            app.update();
            for &garment in &garments {
                assert!(app.world().get::<AwaitRigidPins>(garment).is_some());
                assert!(app.world().get::<ClothSim>(garment).is_none());
                assert!(app.world().get::<ClothAttachmentMotion>(garment).is_none());
            }
        }
        let mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD,
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        );
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::splat(3.0),
            Quat::from_rotation_y(0.7),
            Vec3::new(4.0, 5.0, 6.0),
        );
        let prepared = prepare_mesh_collider(&mesh, transform).unwrap();
        app.world_mut().entity_mut(collider).insert((
            MeshColliderPreparation::Ready(Box::new(prepared)),
            torso_motion(),
            // Binding must use the prepared provider pivot, not a render Transform.
            Transform::from_xyz(90.0, 80.0, 70.0),
        ));
        app.update();
        for &garment in &garments {
            assert!(app.world().get::<AwaitRigidPins>(garment).is_none());
            assert_config_installed(&app, garment, &config);
            let attachment = app.world().get::<ClothAttachmentMotion>(garment).unwrap();
            assert_eq!(attachment.pivot, Vec3::new(4.0, 5.0, 6.0));
            assert_eq!(
                attachment.motion,
                *app.world().get::<ColliderMotion>(collider).unwrap()
            );
        }
        assert!(app.world().get::<ClothSim>(inactive).is_none());
        assert!(app.world().get::<ClothAttachmentMotion>(inactive).is_none());
        // A later preparation change must not rebind an already-installed garment.
        app.world_mut()
            .entity_mut(collider)
            .insert(MeshColliderPreparation::Waiting);
        app.update();
        for &garment in &garments {
            assert!(app.world().get::<ClothSim>(garment).is_some());
            assert_eq!(
                app.world()
                    .get::<ClothAttachmentMotion>(garment)
                    .unwrap()
                    .pivot,
                Vec3::new(4.0, 5.0, 6.0)
            );
        }
    }
}
