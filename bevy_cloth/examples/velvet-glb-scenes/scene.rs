//! Load the baked primitives, then assemble the original Velvet scene fixtures.

use bevy::{
    asset::{LoadState, RenderAssetUsages},
    gltf::{GltfAssetLabel, GltfLoaderSettings, convert_coordinates::GltfConvertCoordinates},
    mesh::MeshCompressionArgs,
    prelude::*,
};
use bevy_cloth::{
    Cloth, ClothDynamicAttachments, ClothExternalFriction, ClothSelfCollision, ClothSolver,
    ClothStepSettings, ClothVbdContact, ClothVbdMaterial, DynamicAttachment, FixedAttachment,
    MeshCollider, VbdStaticCollider,
    legacy::{
        demo::scenes::{LegacyClothSpec, LegacyColliderSpec, legacy_comparison_scene},
        demo::{LegacyColliderMotion, LegacySceneId, legacy_scene_params},
        sim::collide::LegacyColliderKind,
    },
};

const RESOLUTIONS: [u32; 6] = [16, 36, 40, 60, 64, 200];
const SURFACE_DENSITY: f32 = 0.2; // kg/m² for this demo, independent of grid resolution.

#[derive(Component)]
pub struct SceneEntity;

#[derive(Component)]
pub struct SceneMotion(pub LegacyColliderSpec);

#[derive(Resource)]
pub struct SceneAssets {
    sheets: [Handle<Mesh>; RESOLUTIONS.len()],
    ground: Handle<Mesh>,
    sphere: Handle<Mesh>,
    cube: Handle<Mesh>,
    fabrics: [Handle<StandardMaterial>; 3],
    layered_fabric: Handle<StandardMaterial>,
    ground_material: Handle<StandardMaterial>,
    collider_material: Handle<StandardMaterial>,
}

impl SceneAssets {
    pub fn load(server: &AssetServer, materials: &mut Assets<StandardMaterial>) -> Self {
        let load_mesh = |name: &str| {
            server
                .load_builder()
                .with_settings(|settings: &mut GltfLoaderSettings| {
                    settings.load_meshes = RenderAssetUsages::default();
                    // Pin IDs and output shader layouts rely on the authored mesh data.
                    settings.convert_coordinates = Some(GltfConvertCoordinates {
                        rotate_scene_entity: false,
                        rotate_meshes: false,
                    });
                    settings.mesh_compression = Some(MeshCompressionArgs::none());
                })
                .load(
                    GltfAssetLabel::Primitive {
                        mesh: 0,
                        primitive: 0,
                    }
                    .from_asset(format!("velvet/scenes/{name}.glb")),
                )
        };
        let fabric = |number| StandardMaterial {
            base_color_texture: Some(server.load(format!("velvet/fabric{number}.jpg"))),
            base_color: Color::srgb(0.15, 0.64, 0.95),
            cull_mode: None,
            double_sided: true,
            perceptual_roughness: 1.0,
            ..default()
        };
        Self {
            sheets: RESOLUTIONS.map(|r| load_mesh(&format!("cloth-{r}"))),
            ground: load_mesh("ground"),
            sphere: load_mesh("sphere"),
            cube: load_mesh("cube"),
            fabrics: [1, 2, 3].map(|number| materials.add(fabric(number))),
            layered_fabric: materials.add(StandardMaterial {
                base_color: Color::WHITE,
                ..fabric(1)
            }),
            ground_material: materials.add(StandardMaterial {
                base_color: Color::srgb(0.32, 0.28, 0.24),
                perceptual_roughness: 1.0,
                ..default()
            }),
            collider_material: materials.add(StandardMaterial {
                perceptual_roughness: 0.9,
                ..default()
            }),
        }
    }

    fn sheet(&self, resolution: u32) -> &Handle<Mesh> {
        let index = RESOLUTIONS
            .iter()
            .position(|&r| r == resolution)
            .expect("a GLB must exist for every Velvet resolution");
        &self.sheets[index]
    }

    fn collider(&self, kind: LegacyColliderKind) -> (&Handle<Mesh>, &Handle<StandardMaterial>) {
        match kind {
            LegacyColliderKind::Plane => (&self.ground, &self.ground_material),
            LegacyColliderKind::Sphere => (&self.sphere, &self.collider_material),
            LegacyColliderKind::Cube => (&self.cube, &self.collider_material),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct VbdOptions {
    pub friction: bool,
    pub self_contact: bool,
}

/// Returns the cloth entity and its mesh ID, or `Ok(None)` while GLBs are loading.
/// Only the new cloth copy is written by the GPU; collider and source meshes are shared.
pub(super) fn spawn_scene(
    commands: &mut Commands,
    assets: &SceneAssets,
    server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    id: LegacySceneId,
    solver: ClothSolver,
    vbd_options: VbdOptions,
) -> Result<Option<(Entity, AssetId<Mesh>)>, String> {
    let scene = legacy_comparison_scene(id).expect("all seven scenes have fixtures");
    let spec = &scene.cloths[0];
    let source_handle = assets.sheet(spec.resolution);
    let all_meshes = std::iter::once(source_handle)
        .chain(scene.colliders.iter().map(|c| assets.collider(c.kind).0));
    for handle in all_meshes {
        if let Some(LoadState::Failed(error)) = server.get_load_state(handle.id()) {
            return Err(format!("GLB {:?}: {error}", handle.path()));
        }
        if meshes.get(handle).is_none() {
            return Ok(None);
        }
    }
    let Some(source) = meshes.get(source_handle) else {
        return Ok(None);
    };
    let positions = source
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|attribute| attribute.as_float3())
        .ok_or("cloth GLB needs Float32x3 POSITION")?;
    let side = spec.resolution as usize + 1;
    if positions.len() != side * side
        || source.indices().map(|indices| indices.len()) != Some(6 * (side - 1).pow(2))
    {
        return Err(format!(
            "cloth-{}.glb does not match its Velvet grid",
            spec.resolution
        ));
    }
    let vbd = solver == ClothSolver::Vbd;
    let dynamic_pin = id == LegacySceneId::Swirl;
    let layered = scene.cloths.len() > 1;
    let pose = Transform::from_matrix(spec.pose());
    let attachments = if dynamic_pin {
        Vec::new()
    } else {
        spec.attached
            .iter()
            .map(|&particle| FixedAttachment {
                particle,
                position: pose.transform_point(Vec3::from(positions[particle as usize])),
            })
            .collect()
    };
    let (mesh, transform, material) = if layered {
        (
            merge_layers(source, &scene.cloths)?,
            Transform::IDENTITY,
            assets.layered_fabric.clone(),
        )
    } else {
        (
            source.clone(),
            pose,
            assets.fabrics[spec.texture as usize - 1].clone(),
        )
    };
    let particle_mass = 4.0 * SURFACE_DENSITY / positions.len() as f32;
    let mesh = meshes.add(mesh);
    let mesh_id = mesh.id();
    let mut target = None;
    for collider in scene.colliders {
        let (geometry, material) = assets.collider(collider.kind);
        let transform = collider_pose(&collider, id, vbd);
        let mut entity = commands.spawn((
            SceneEntity,
            Mesh3d(geometry.clone()),
            MeshMaterial3d(material.clone()),
            transform,
        ));
        if collider.enabled {
            if vbd {
                entity.insert(VbdStaticCollider);
            } else {
                entity.insert(MeshCollider);
            }
        }
        if collider.motion == LegacyColliderMotion::SwirlSlot {
            target = Some(entity.id()); // decorative sphere is the target, not a collider
        }
        if collider.motion != LegacyColliderMotion::Static && (!vbd || !collider.enabled) {
            entity.insert(SceneMotion(collider));
        }
    }

    let params = legacy_scene_params(id);
    let mut cloth = commands.spawn((
        Name::new(id.label()),
        SceneEntity,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        transform,
        solver,
        Cloth {
            particle_mass,
            stretch_compliance: 0.0,
            bend_compliance: params.bend_compliance,
            attachments,
            collision_margin: params.collision_margin,
            double_sided: false,
        },
        ClothStepSettings {
            gravity: params.gravity,
            substeps: params.num_substeps,
            iterations: params.num_iterations,
        },
    ));
    if vbd {
        cloth.insert((
            ClothVbdMaterial {
                shear: 500.0,
                area: 500.0,
                bend: 0.005,
                ..default()
            },
            ClothVbdContact { stiffness: 200.0 },
        ));
    }
    if !vbd || vbd_options.friction {
        cloth.insert(ClothExternalFriction {
            coefficient: params.friction,
        });
    }
    if !vbd || vbd_options.self_contact {
        cloth.insert(ClothSelfCollision {
            diameter: 2.0 * params.particle_diameter_scalar / spec.resolution as f32,
            friction: if vbd { 0.0 } else { params.friction },
            neighbor_capacity: params.max_num_neighbors,
        });
    }
    if let Some(target) = target {
        cloth.insert(ClothDynamicAttachments(vec![DynamicAttachment {
            particle: spec.attached[0],
            target,
            local_offset: Vec3::ZERO,
        }]));
    }
    Ok(Some((cloth.id(), mesh_id)))
}

fn collider_pose(collider: &LegacyColliderSpec, id: LegacySceneId, vbd: bool) -> Transform {
    if collider.motion == LegacyColliderMotion::SwirlSlot {
        return Transform::from_matrix(collider.transform_at(0.0));
    }
    let mut pose = Transform::from_matrix(collider.initial_transform());
    if vbd && id == LegacySceneId::Collision && collider.motion == LegacyColliderMotion::SweepZ {
        // Freeze the sphere where it can contact the hanging sheet. Its original
        // z = -1 starting pose leaves a gap until it moves (unsupported by VBD).
        pose.translation.z = -0.5;
    }
    pose
}

fn merge_layers(source: &Mesh, layers: &[LegacyClothSpec]) -> Result<Mesh, String> {
    let colors = [
        [0.25, 0.65, 1.0, 1.0],
        [1.0, 0.55, 0.25, 1.0],
        [0.35, 1.0, 0.55, 1.0],
    ];
    let transformed = |spec: &LegacyClothSpec, color| {
        source
            .clone()
            .transformed_by(Transform::from_matrix(spec.pose()))
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![color; source.count_vertices()])
    };
    // One writable primitive, three disconnected components: no constraints
    // connect the sheets, but XPBD self-contact can see all three layers.
    let mut combined = transformed(&layers[0], colors[0]);
    for (layer, color) in layers[1..].iter().zip(colors[1..].iter().copied()) {
        combined
            .merge(&transformed(layer, color))
            .map_err(|error| error.to_string())?;
    }
    Ok(combined)
}
