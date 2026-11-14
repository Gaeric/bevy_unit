//! the seven scenes of `main.cpp`, 1:1 (plan.org §1.7).
//!
//! `LegacySimParams` overrides follow the source's `Scene::ModifyParameter` semantics: applied when a
//! scene is entered, reverted when it is left, so nothing leaks into the next scene. every switch
//! rebuilds the whole scene (camera, light, colliders, cloth), like the source rebuilding its
//! `GameInstance`.

use bevy::{ecs::system::SystemParam, prelude::*};
use glam::{Mat3, Mat4, Quat, Vec3};

use crate::legacy::{
    demo::{
        camera::LegacyOrbitCamera,
        plugin::{
            LEGACY_FIXED_DELTA, LegacyClothCollider, LegacyClothVisual, LegacyColliderHistory,
            LegacyColliderMotion, LegacySimParamsRes, LegacySolverRes,
        },
        render::legacy_build_cloth_mesh,
    },
    sim::{
        collide::LegacyColliderKind, mesh_gen::legacy_generate_cloth_mesh, params::LegacySimParams,
    },
};

/// the source's `main.cpp` scene list, in `1`-`7` order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LegacySceneId {
    #[default]
    Attach,
    Collision,
    SelfCollision,
    Friction,
    Multiple,
    HighResolution,
    Swirl,
}

impl LegacySceneId {
    /// every scene, in the source's `1`-`7` order (`GameInstance::ProcessKeyboard`).
    pub const ALL: [LegacySceneId; 7] = [
        LegacySceneId::Attach,
        LegacySceneId::Collision,
        LegacySceneId::SelfCollision,
        LegacySceneId::Friction,
        LegacySceneId::Multiple,
        LegacySceneId::HighResolution,
        LegacySceneId::Swirl,
    ];

    /// `Scene::name`.
    pub fn label(self) -> &'static str {
        match self {
            LegacySceneId::Attach => "Cloth / Attach",
            LegacySceneId::Collision => "Cloth / SDF Collision",
            LegacySceneId::SelfCollision => "Cloth / Self Collision",
            LegacySceneId::Friction => "Cloth / Friction",
            LegacySceneId::Multiple => "Cloth / Multiple Object",
            LegacySceneId::HighResolution => "Cloth / High Resolution",
            LegacySceneId::Swirl => "Cloth / Swirl",
        }
    }
}

/// the scene currently built into the world.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LegacyCurrentScene(pub LegacySceneId);

/// `VtEngine::SwitchScene` / `Reset`: the scene to build on the next `Update`.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LegacyPendingScene(pub Option<LegacySceneId>);

/// everything a scene build spawns; torn down on switch (the source rebuilds `GameInstance`).
#[derive(Component)]
pub struct LegacySceneEntity;

/// `Scene::ModifyParameter`: this scene's overrides on top of the parameter defaults.
///
/// the source applies them in `onEnter` and reverts them in `onExit`, so a switch never leaks
/// (plan.org §1.7, §6).
pub fn legacy_scene_params(id: LegacySceneId) -> LegacySimParams {
    let mut params = LegacySimParams::default();

    match id {
        LegacySceneId::SelfCollision => {
            // the source stacks `numSubsteps` 3 then 8, so 8 wins (plan.org §1.7)
            params.num_substeps = 8;
            params.friction = 0.3;
        }
        LegacySceneId::Friction | LegacySceneId::Multiple => {
            params.friction = 0.6;
            params.num_substeps = 5;
            params.num_iterations = 5;
        }
        LegacySceneId::HighResolution => {
            params.num_substeps = 10;
            params.num_iterations = 10;
        }
        LegacySceneId::Attach | LegacySceneId::Collision | LegacySceneId::Swirl => {}
    }

    params
}

/// Shared fixtures for all seven CPU/GPU scenes. Parameters remain in `legacy_scene_params`.
pub struct LegacyComparisonScene {
    pub cloths: Vec<LegacyClothSpec>,
    /// Source spawn order: ground first, then any sphere or cube (including disabled visuals).
    pub colliders: Vec<LegacyColliderSpec>,
}

/// Analytic collider description; model matrices include the shape's scale exactly once.
#[derive(Clone)]
pub struct LegacyColliderSpec {
    pub kind: LegacyColliderKind,
    pub radius: f32,
    pub half_extents: Vec3,
    pub position: Vec3,
    pub motion: LegacyColliderMotion,
    pub enabled: bool,
}

impl LegacyColliderSpec {
    /// Initial world model, independent of the animation clock.
    pub fn initial_transform(&self) -> Mat4 {
        let scale = match self.kind {
            LegacyColliderKind::Sphere => Vec3::splat(self.radius),
            LegacyColliderKind::Cube => self.half_extents * 2.0,
            LegacyColliderKind::Plane => Vec3::ONE,
        };
        Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, self.position)
    }

    /// World model at simulation time, including shape scale exactly once.
    /// Disabled colliders still animate visually (notably the Swirl sphere).
    pub fn transform_at(&self, seconds: f32) -> Mat4 {
        let mut model = self.initial_transform();
        match self.motion {
            LegacyColliderMotion::Static => {}
            LegacyColliderMotion::SweepZ => {
                model.w_axis = legacy_sweep_z_position(self.radius, seconds).extend(1.0);
            }
            LegacyColliderMotion::FrictionSweep => {
                let (position, rotation) = legacy_friction_sweep_pose(self.radius, seconds);
                model = Mat4::from_scale_rotation_translation(
                    Vec3::splat(self.radius),
                    rotation,
                    position.unwrap_or(self.position),
                );
            }
            LegacyColliderMotion::SwirlSlot => {
                model.w_axis = legacy_swirl_slot_position(seconds).extend(1.0);
            }
        }
        model
    }
}

/// `SceneClothCollision` sphere center, shared by CPU animation and comparison fixtures.
pub fn legacy_sweep_z_position(radius: f32, seconds: f32) -> Vec3 {
    Vec3::new(0.0, radius, -(seconds * 2.0).cos())
}

/// `SceneClothFriction`: translation is assigned only after the half-second delay;
/// spin uses truncation to signed integer and signed remainder, not Euclidean modulo.
pub fn legacy_friction_sweep_pose(radius: f32, seconds: f32) -> (Option<Vec3>, Quat) {
    let time = seconds - 0.5;
    let position = (time > 0.0).then(|| Vec3::new(time.sin(), radius, 0.0));
    let spin = if (time as i32) % 4 > 1 {
        -time * 180.0
    } else {
        time * 180.0
    };
    (position, Quat::from_rotation_y(spin.to_radians()))
}

/// `SceneClothSwirl`: shared attachment slot and disabled decorative sphere center.
pub fn legacy_swirl_slot_position(seconds: f32) -> Vec3 {
    Vec3::new((seconds * 3.0).sin(), (seconds * 3.0).cos() + 2.0, 0.0)
}

/// Every `LegacySceneId` has a shared fixture; the `Option` signature is retained for callers.
pub fn legacy_comparison_scene(id: LegacySceneId) -> Option<LegacyComparisonScene> {
    let (cloth, collider) = match id {
        LegacySceneId::Attach => {
            let resolution = 40;
            (
                LegacyClothSpec {
                    resolution,
                    texture: 1,
                    position: Vec3::new(0.0, 1.5, 1.0),
                    rotation: Vec3::new(90.0, 0.0, 0.0),
                    attached: vec![
                        0,
                        resolution,
                        (resolution + 1) * (resolution + 1) - 1,
                        (resolution + 1) * resolution,
                    ],
                },
                LegacyColliderSpec {
                    kind: LegacyColliderKind::Sphere,
                    radius: 0.5,
                    half_extents: Vec3::ONE,
                    position: Vec3::new(0.0, 0.5, 0.0),
                    motion: LegacyColliderMotion::Static,
                    enabled: true,
                },
            )
        }
        LegacySceneId::Collision => (
            LegacyClothSpec {
                resolution: 16,
                texture: 2,
                position: Vec3::new(0.0, 2.5, 0.0),
                rotation: Vec3::ZERO,
                attached: vec![0, 16],
            },
            LegacyColliderSpec {
                kind: LegacyColliderKind::Sphere,
                radius: 0.6,
                half_extents: Vec3::ONE,
                position: Vec3::new(0.0, 0.6, -1.0),
                motion: LegacyColliderMotion::SweepZ,
                enabled: true,
            },
        ),
        LegacySceneId::SelfCollision => (
            LegacyClothSpec {
                resolution: 60,
                texture: 1,
                position: Vec3::new(0.0, 1.5, 1.0),
                rotation: Vec3::new(-15.0, 10.0, 10.0),
                attached: Vec::new(),
            },
            // No actor collider beyond the ground in this scene.
            LegacyColliderSpec {
                kind: LegacyColliderKind::Plane,
                radius: 0.0,
                half_extents: Vec3::ONE,
                position: Vec3::ZERO,
                motion: LegacyColliderMotion::Static,
                enabled: true,
            },
        ),
        LegacySceneId::Friction
        | LegacySceneId::Multiple
        | LegacySceneId::HighResolution
        | LegacySceneId::Swirl => {
            let (resolution, texture, radius, kind, motion, enabled, attached) = match id {
                LegacySceneId::Friction => (
                    64,
                    2,
                    0.5,
                    LegacyColliderKind::Sphere,
                    LegacyColliderMotion::FrictionSweep,
                    true,
                    vec![],
                ),
                LegacySceneId::Multiple => (
                    64,
                    1,
                    0.0,
                    LegacyColliderKind::Cube,
                    LegacyColliderMotion::Static,
                    true,
                    vec![],
                ),
                LegacySceneId::HighResolution => (
                    200,
                    2,
                    0.6,
                    LegacyColliderKind::Sphere,
                    LegacyColliderMotion::Static,
                    true,
                    vec![],
                ),
                LegacySceneId::Swirl => (
                    36,
                    1,
                    0.1,
                    LegacyColliderKind::Sphere,
                    LegacyColliderMotion::SwirlSlot,
                    false,
                    vec![0],
                ),
                _ => unreachable!(),
            };
            (
                LegacyClothSpec {
                    resolution,
                    texture,
                    position: Vec3::new(0.0, 1.5, 1.0),
                    rotation: Vec3::new(90.0, 0.0, 0.0),
                    attached,
                },
                LegacyColliderSpec {
                    kind,
                    radius,
                    half_extents: if kind == LegacyColliderKind::Cube {
                        Vec3::splat(0.5)
                    } else {
                        Vec3::ONE
                    },
                    position: Vec3::new(
                        0.0,
                        if kind == LegacyColliderKind::Cube {
                            0.5
                        } else {
                            radius
                        },
                        0.0,
                    ),
                    motion,
                    enabled,
                },
            )
        }
    };
    let mut cloths = vec![cloth];
    if id == LegacySceneId::Multiple {
        for (texture, height) in [(2, 1.8), (3, 2.1)] {
            cloths.push(LegacyClothSpec {
                texture,
                position: Vec3::new(0.0, height, 1.0),
                ..cloths[0].clone()
            });
        }
    }
    let mut scene = LegacyComparisonScene {
        cloths,
        colliders: vec![LegacyColliderSpec {
            kind: LegacyColliderKind::Plane,
            radius: 0.0,
            half_extents: Vec3::ONE,
            position: Vec3::ZERO,
            motion: LegacyColliderMotion::Static,
            enabled: true,
        }],
    };
    if id != LegacySceneId::SelfCollision {
        scene.colliders.push(collider);
    }
    Some(scene)
}

/// the assets and solver state a scene build needs, grouped so the helper signatures stay short.
#[derive(SystemParam)]
pub struct LegacySceneContext<'w, 's> {
    commands: Commands<'w, 's>,
    asset_server: Res<'w, AssetServer>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    solver: ResMut<'w, LegacySolverRes>,
    params: ResMut<'w, LegacySimParamsRes>,
}

/// `Scene::PopulateActors` + `onEnter` for the initial scene.
pub fn legacy_setup_initial_scene(mut context: LegacySceneContext) {
    build_scene(&mut context, LegacySceneId::default());
}

/// `VtEngine::SwitchScene` / `Reset`: run the current scene's `onExit`, then the next `onEnter`.
pub fn legacy_apply_scene_switch(
    mut context: LegacySceneContext,
    mut current: ResMut<LegacyCurrentScene>,
    mut pending: ResMut<LegacyPendingScene>,
    entities: Query<Entity, With<LegacySceneEntity>>,
    meshes: Query<&Mesh3d, With<LegacySceneEntity>>,
    materials: Query<&MeshMaterial3d<StandardMaterial>, With<LegacySceneEntity>>,
) {
    let Some(next) = pending.0.take() else {
        return;
    };

    // the source drops its resource cache between scenes; do the same so `res = 200` does not pile
    // meshes up on every switch
    for mesh in &meshes {
        context.meshes.remove(&mesh.0);
    }
    for material in &materials {
        context.materials.remove(&material.0);
    }
    for entity in &entities {
        context.commands.entity(entity).despawn();
    }
    // the source throws the whole `GameInstance` away, so the solver starts empty every switch
    *context.solver = LegacySolverRes::default();

    build_scene(&mut context, next);
    current.0 = next;
}

/// a scene's actor list, shared by `Scene::PopulateActors` and the switch entry point.
fn build_scene(context: &mut LegacySceneContext, id: LegacySceneId) {
    // `onEnter`: the overrides must be in place *before* any cloth is added, so `AddCloth`'s
    // `maxSpeed = 2 * pd / dt * numSubsteps` sees the final `numSubsteps` (plan.org §4-6)
    *context.params = LegacySimParamsRes(legacy_scene_params(id));

    spawn_camera_and_light(context);

    spawn_comparison_scene(context, id);
}

/// Keep the CPU visuals and history initialization in the existing spawn helpers.
fn spawn_comparison_scene(context: &mut LegacySceneContext, id: LegacySceneId) {
    let scene = legacy_comparison_scene(id).expect("all seven scenes have shared descriptors");
    for collider in scene.colliders {
        match collider.kind {
            LegacyColliderKind::Plane => spawn_ground(context),
            LegacyColliderKind::Sphere => spawn_sphere(
                context,
                collider.radius,
                collider.position,
                collider.motion,
                collider.enabled,
            ),
            LegacyColliderKind::Cube => {
                spawn_cube(context, collider.half_extents.x, collider.position)
            }
        }
    }
    for cloth in scene.cloths {
        spawn_cloth(context, cloth);
    }
}

/// one `SpawnCloth` call.
#[derive(Clone)]
pub struct LegacyClothSpec {
    pub resolution: u32,
    pub texture: u32,
    pub position: Vec3,
    /// the source's `Transform::rotation`, in degrees
    pub rotation: Vec3,
    /// pinned particle ids, in the source's order
    pub attached: Vec<u32>,
}

impl LegacyClothSpec {
    /// Pose baked into particle positions by both CPU and GPU scene construction.
    pub fn pose(&self) -> Mat4 {
        source_pose(self.position, self.rotation)
    }
}

/// `Scene.hpp::SpawnCloth` + `Actor::Initialize`: the pose is baked into the mesh by `add_cloth`,
/// so the entity itself stays at `Transform::IDENTITY` (plan.org §4-1).
fn spawn_cloth(context: &mut LegacySceneContext, spec: LegacyClothSpec) {
    let cloth = legacy_generate_cloth_mesh(spec.resolution);
    let pose = spec.pose();

    let offset = context
        .solver
        .0
        .add_cloth(
            &mut context.params.0,
            &cloth,
            pose,
            &spec.attached,
            LEGACY_FIXED_DELTA,
        )
        .expect("generated meshes are source-layout grids");

    let mesh = context.meshes.add(legacy_build_cloth_mesh(&cloth));
    let material = context
        .materials
        .add(cloth_material(&context.asset_server, spec.texture));

    context.commands.spawn((
        LegacySceneEntity,
        Mesh3d(mesh.clone()),
        MeshMaterial3d(material),
        Transform::IDENTITY,
        LegacyClothVisual {
            mesh,
            offset: offset as usize,
            count: cloth.positions.len(),
        },
    ));
}

/// `Scene.hpp::SpawnCloth`'s material property.
pub(crate) fn cloth_material(asset_server: &AssetServer, texture: u32) -> StandardMaterial {
    StandardMaterial {
        // `material.tint`
        base_color: Color::srgb(0.0, 0.5, 1.0),
        // the asset root is `CARGO_MANIFEST_DIR`, so this resolves under `bevy_cloth/assets/`
        base_color_texture: Some(
            asset_server.load(format!("velvet/fabric{}.jpg", texture.clamp(1, 3))),
        ),
        double_sided: true,
        cull_mode: None,
        perceptual_roughness: 1.0,
        reflectance: 0.01,
        ..default()
    }
}

/// `Scene.hpp::SpawnCameraAndLight`, rebuilt with every scene.
fn spawn_camera_and_light(context: &mut LegacySceneContext) {
    // camera position (0.35, 3.3, 7.2) / euler (-21, 2.25, 0) in degrees. `LegacyOrbitCameraPlugin`
    // takes over the transform when it is added; without it the camera stays at this pose.
    let orbit = LegacyOrbitCamera::source();
    context.commands.spawn((
        LegacySceneEntity,
        Camera3d::default(),
        orbit.transform(),
        orbit,
    ));

    // spot light at (2.5, 5, 2.5). the source uses its own euler convention for the direction, so
    // we aim at the cloth instead.
    context
        .commands
        .spawn((LegacySceneEntity, comparison_spotlight()));
}

/// Shared CPU/GPU spotlight, including the CPU scene's aim and shadow settings.
pub(crate) fn comparison_spotlight() -> (SpotLight, Transform) {
    (
        SpotLight {
            color: Color::WHITE,
            intensity: 3_000_000.0,
            range: 50.0,
            radius: 0.2,
            shadow_maps_enabled: true,
            inner_angle: 40.0_f32.to_radians(),
            outer_angle: 50.0_f32.to_radians(),
            ..default()
        },
        Transform::from_xyz(2.5, 5.0, 2.5).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    )
}

/// Shared CPU/GPU ambient environment; this resource does not create a camera.
pub(crate) fn comparison_ambient_light() -> GlobalAmbientLight {
    GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
        affects_lightmapped_meshes: true,
    }
}

/// Shared CPU/GPU ground appearance.
pub(crate) fn comparison_ground_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(0.32, 0.28, 0.24),
        perceptual_roughness: 1.0,
        ..default()
    }
}

/// `Scene.hpp::SpawnInfinitePlane`: a large quad for display, plus the `Plane` SDF at `y = 0`.
fn spawn_ground(context: &mut LegacySceneContext) {
    let mesh = context
        .meshes
        .add(Plane3d::default().mesh().size(50.0, 50.0));
    let material = context.materials.add(comparison_ground_material());

    context.commands.spawn((
        LegacySceneEntity,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::IDENTITY,
        LegacyClothCollider {
            kind: LegacyColliderKind::Plane,
            radius: 0.0,
            half_extents: Vec3::ONE,
            motion: LegacyColliderMotion::Static,
            enabled: true,
        },
        LegacyColliderHistory(Mat4::IDENTITY),
    ));
}

/// Shared CPU/GPU collider appearance, also used by the CPU cube.
pub(crate) fn comparison_collider_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        ..default()
    }
}

/// `Scene.hpp::SpawnSphere` with the scene's radius, motion and `Collider::enabled`.
fn spawn_sphere(
    context: &mut LegacySceneContext,
    radius: f32,
    position: Vec3,
    motion: LegacyColliderMotion,
    enabled: bool,
) {
    let mesh = context.meshes.add(Sphere::new(radius));
    let material = context.materials.add(comparison_collider_material());

    context.commands.spawn((
        LegacySceneEntity,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_translation(position),
        LegacyClothCollider {
            kind: LegacyColliderKind::Sphere,
            radius,
            half_extents: Vec3::ONE,
            motion,
            enabled,
        },
        // a fresh collider starts with zero relative velocity (`Collider::Start`)
        LegacyColliderHistory(Mat4::from_translation(position)),
    ));
}

/// `Scene.hpp::SpawnColoredCube`. `half_extent` is the world half size; `size` doubles it back into
/// the source's unit-cube `transform.scale`.
fn spawn_cube(context: &mut LegacySceneContext, half_extent: f32, position: Vec3) {
    let mesh = context.meshes.add(Cuboid::new(
        half_extent * 2.0,
        half_extent * 2.0,
        half_extent * 2.0,
    ));
    let material = context.materials.add(comparison_collider_material());

    context.commands.spawn((
        LegacySceneEntity,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_translation(position),
        LegacyClothCollider {
            kind: LegacyColliderKind::Cube,
            radius: 0.0,
            half_extents: Vec3::splat(half_extent),
            motion: LegacyColliderMotion::Static,
            enabled: true,
        },
        LegacyColliderHistory(Mat4::from_translation(position)),
    ));
}

#[cfg(test)]
mod comparison_appearance_tests {
    use super::*;

    #[test]
    fn spotlight_preserves_cpu_lighting_and_pose() {
        let (light, transform) = comparison_spotlight();
        assert_eq!(light.color, Color::WHITE);
        assert_eq!(light.intensity, 3_000_000.0);
        assert_eq!(light.range, 50.0);
        assert_eq!(light.radius, 0.2);
        assert!(light.shadow_maps_enabled);
        assert_eq!(light.inner_angle, 40.0_f32.to_radians());
        assert_eq!(light.outer_angle, 50.0_f32.to_radians());
        assert_eq!(
            transform,
            Transform::from_xyz(2.5, 5.0, 2.5).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y)
        );
    }

    #[test]
    fn ambient_preserves_cpu_environment() {
        let ambient = comparison_ambient_light();
        assert_eq!(ambient.color, Color::WHITE);
        assert_eq!(ambient.brightness, 200.0);
        assert!(ambient.affects_lightmapped_meshes);
    }

    #[test]
    fn ground_and_collider_preserve_cpu_materials() {
        let ground = comparison_ground_material();
        let collider = comparison_collider_material();
        assert_eq!(ground.base_color, Color::srgb(0.32, 0.28, 0.24));
        assert_eq!(collider.base_color, Color::WHITE);
        for material in [ground, collider] {
            assert_eq!(material.perceptual_roughness, 1.0);
            assert_eq!(
                material.reflectance,
                StandardMaterial::default().reflectance
            );
            assert!(!material.unlit);
            assert!(!material.double_sided);
            assert_eq!(material.cull_mode, StandardMaterial::default().cull_mode);
            assert!(material.base_color_texture.is_none());
            assert!(material.emissive_texture.is_none());
            assert!(material.metallic_roughness_texture.is_none());
            assert!(material.normal_map_texture.is_none());
            assert!(material.occlusion_texture.is_none());
        }
    }

    #[test]
    fn cloth_preserves_cpu_lit_textured_double_sided_material() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Image>();
        let asset_server = app.world().resource::<AssetServer>();
        for (requested, expected) in [(0, 1), (1, 1), (2, 2), (3, 3), (4, 3)] {
            let material = cloth_material(asset_server, requested);
            assert_eq!(material.base_color, Color::srgb(0.0, 0.5, 1.0));
            assert_eq!(material.perceptual_roughness, 1.0);
            assert_eq!(material.reflectance, 0.01);
            assert!(!material.unlit);
            assert!(material.double_sided);
            assert_eq!(material.cull_mode, None);
            let texture = material.base_color_texture.expect("cloth has a texture");
            assert_eq!(
                asset_server.get_path(texture.id()).unwrap().path(),
                std::path::Path::new(&format!("velvet/fabric{expected}.jpg"))
            );
            assert!(material.normal_map_texture.is_none());
        }
    }
}

/// `Transform::matrix`: `translate * Ry * Rz * Rx * scale`, with the source's degree angles.
fn source_pose(position: Vec3, rotation: Vec3) -> Mat4 {
    let rotation = Mat3::from_rotation_y(rotation.y.to_radians())
        * Mat3::from_rotation_z(rotation.z.to_radians())
        * Mat3::from_rotation_x(rotation.x.to_radians());

    Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::from_mat3(&rotation), position)
}
