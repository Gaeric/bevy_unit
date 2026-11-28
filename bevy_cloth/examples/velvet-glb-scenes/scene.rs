//! Load the baked primitives and assemble the Velvet Attach fixture.

use bevy::{
    asset::{LoadState, RenderAssetUsages},
    gltf::{GltfAssetLabel, GltfLoaderSettings, convert_coordinates::GltfConvertCoordinates},
    mesh::MeshCompressionArgs,
    prelude::*,
};
use bevy_cloth::{
    Cloth, ClothExternalFriction, ClothSelfCollision, ClothSolver, ClothStepSettings,
    ClothVbdContact, ClothVbdMaterial, FixedAttachment, MeshCollider, VbdStaticCollider,
    legacy::{
        demo::{LegacySceneId, legacy_scene_params, scenes::legacy_comparison_scene},
        sim::collide::LegacyColliderKind,
    },
};

const SURFACE_DENSITY: f32 = 0.2; // kg/m² for this demo, independent of grid resolution.

#[derive(Component)]
pub struct SceneEntity;

#[derive(Resource)]
pub struct SceneAssets {
    sheet: Handle<Mesh>,
    ground: Handle<Mesh>,
    sphere: Handle<Mesh>,
    fabric: Handle<StandardMaterial>,
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
        let fabric = materials.add(StandardMaterial {
            base_color_texture: Some(server.load("velvet/fabric1.jpg")),
            base_color: Color::srgb(0.15, 0.64, 0.95),
            cull_mode: None,
            double_sided: true,
            perceptual_roughness: 1.0,
            ..default()
        });
        Self {
            sheet: load_mesh("cloth-40"),
            ground: load_mesh("ground"),
            sphere: load_mesh("sphere"),
            fabric,
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

    fn collider(&self, kind: LegacyColliderKind) -> (&Handle<Mesh>, &Handle<StandardMaterial>) {
        match kind {
            LegacyColliderKind::Plane => (&self.ground, &self.ground_material),
            LegacyColliderKind::Sphere => (&self.sphere, &self.collider_material),
            LegacyColliderKind::Cube => unreachable!("the Attach fixture has no cube"),
        }
    }
}

/// Returns `Ok(None)` while the GLBs for this scene are still loading.
/// Only the new cloth copy is written by the GPU; collider and source meshes are shared.
pub(super) fn spawn_scene(
    commands: &mut Commands,
    assets: &SceneAssets,
    server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    solver: ClothSolver,
) -> Result<Option<AssetId<Mesh>>, String> {
    let id = LegacySceneId::Attach;
    let scene = legacy_comparison_scene(id).expect("the Attach fixture exists");
    let spec = &scene.cloths[0];
    let source_handle = &assets.sheet;
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
    let pose = Transform::from_matrix(spec.pose());
    let attachments = spec
        .attached
        .iter()
        .map(|&particle| FixedAttachment {
            particle,
            position: pose.transform_point(Vec3::from(positions[particle as usize])),
        })
        .collect();
    let particle_mass = 4.0 * SURFACE_DENSITY / positions.len() as f32;
    let mesh = meshes.add(source.clone());
    let mesh_id = mesh.id();
    for collider in scene.colliders {
        let (geometry, material) = assets.collider(collider.kind);
        let transform = Transform::from_matrix(collider.initial_transform());
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
    }

    let params = legacy_scene_params(id);
    let mut cloth = commands.spawn((
        Name::new(id.label()),
        SceneEntity,
        Mesh3d(mesh),
        MeshMaterial3d(assets.fabric.clone()),
        pose,
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
    } else {
        cloth.insert(ClothExternalFriction {
            coefficient: params.friction,
        });
        cloth.insert(ClothSelfCollision {
            diameter: 2.0 * params.particle_diameter_scalar / spec.resolution as f32,
            friction: params.friction,
            neighbor_capacity: params.max_num_neighbors,
        });
    }
    Ok(Some(mesh_id))
}
