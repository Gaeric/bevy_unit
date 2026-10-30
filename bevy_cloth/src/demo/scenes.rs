//! the seven scenes of `main.cpp`, 1:1 (plan.org §1.7).
//!
//! `SimParams` overrides follow the source's `Scene::ModifyParameter` semantics: applied when a
//! scene is entered, reverted when it is left, so nothing leaks into the next scene. every switch
//! rebuilds the whole scene (camera, light, colliders, cloth), like the source rebuilding its
//! `GameInstance`.

use bevy::{ecs::system::SystemParam, prelude::*};
use glam::{Mat3, Mat4, Quat, Vec3};

use crate::{
    demo::{
        camera::OrbitCamera,
        plugin::{
            ClothCollider, ClothVisual, ColliderHistory, ColliderMotion, FIXED_DELTA, SimParamsRes,
            SolverRes,
        },
        render::build_cloth_mesh,
    },
    sim::{collide::ColliderKind, mesh_gen::generate_cloth_mesh, params::SimParams},
};

/// the source's `main.cpp` scene list, in `1`-`7` order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SceneId {
    #[default]
    Attach,
    Collision,
    SelfCollision,
    Friction,
    Multiple,
    HighResolution,
    Swirl,
}

impl SceneId {
    /// every scene, in the source's `1`-`7` order (`GameInstance::ProcessKeyboard`).
    pub const ALL: [SceneId; 7] = [
        SceneId::Attach,
        SceneId::Collision,
        SceneId::SelfCollision,
        SceneId::Friction,
        SceneId::Multiple,
        SceneId::HighResolution,
        SceneId::Swirl,
    ];

    /// `Scene::name`.
    pub fn label(self) -> &'static str {
        match self {
            SceneId::Attach => "Cloth / Attach",
            SceneId::Collision => "Cloth / SDF Collision",
            SceneId::SelfCollision => "Cloth / Self Collision",
            SceneId::Friction => "Cloth / Friction",
            SceneId::Multiple => "Cloth / Multiple Object",
            SceneId::HighResolution => "Cloth / High Resolution",
            SceneId::Swirl => "Cloth / Swirl",
        }
    }
}

/// the scene currently built into the world.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CurrentScene(pub SceneId);

/// `VtEngine::SwitchScene` / `Reset`: the scene to build on the next `Update`.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PendingScene(pub Option<SceneId>);

/// everything a scene build spawns; torn down on switch (the source rebuilds `GameInstance`).
#[derive(Component)]
pub struct SceneEntity;

/// `Scene::ModifyParameter`: this scene's overrides on top of the parameter defaults.
///
/// the source applies them in `onEnter` and reverts them in `onExit`, so a switch never leaks
/// (plan.org §1.7, §6).
pub fn scene_params(id: SceneId) -> SimParams {
    let mut params = SimParams::default();

    match id {
        SceneId::SelfCollision => {
            // the source stacks `numSubsteps` 3 then 8, so 8 wins (plan.org §1.7)
            params.num_substeps = 8;
            params.friction = 0.3;
        }
        SceneId::Friction | SceneId::Multiple => {
            params.friction = 0.6;
            params.num_substeps = 5;
            params.num_iterations = 5;
        }
        SceneId::HighResolution => {
            params.num_substeps = 10;
            params.num_iterations = 10;
        }
        SceneId::Attach | SceneId::Collision | SceneId::Swirl => {}
    }

    params
}

/// Shared fixtures for the first CPU/GPU comparison batch. Parameters remain in `scene_params`.
pub struct ComparisonScene {
    pub cloths: Vec<ClothSpec>,
    /// Source spawn order: ground first, then sphere.
    pub colliders: Vec<ColliderSpec>,
}

/// Analytic collider description; model matrices include the shape's scale exactly once.
#[derive(Clone)]
pub struct ColliderSpec {
    pub kind: ColliderKind,
    pub radius: f32,
    pub half_extents: Vec3,
    pub position: Vec3,
    pub motion: ColliderMotion,
    pub enabled: bool,
}

impl ColliderSpec {
    /// Initial world model, independent of the animation clock.
    pub fn initial_transform(&self) -> Mat4 {
        let scale = match self.kind {
            ColliderKind::Sphere => Vec3::splat(self.radius),
            ColliderKind::Cube => self.half_extents * 2.0,
            ColliderKind::Plane => Vec3::ONE,
        };
        Mat4::from_scale_rotation_translation(scale, Quat::IDENTITY, self.position)
    }

    /// World model at simulation time, including shape scale.
    ///
    /// Comparison fixtures use only Static and SweepZ. Other demo-only motions retain their
    /// initial pose here; their existing CPU animation remains in `animate_colliders`.
    pub fn transform_at(&self, seconds: f32) -> Mat4 {
        let mut model = self.initial_transform();
        if self.motion == ColliderMotion::SweepZ {
            model.w_axis = sweep_z_position(self.radius, seconds).extend(1.0);
        }
        model
    }
}

/// `SceneClothCollision` sphere center, shared by CPU animation and comparison fixtures.
pub fn sweep_z_position(radius: f32, seconds: f32) -> Vec3 {
    Vec3::new(0.0, radius, -(seconds * 2.0).cos())
}

/// Only Attach and Collision are supported in this comparison batch.
pub fn comparison_scene(id: SceneId) -> Option<ComparisonScene> {
    let (cloth, sphere) = match id {
        SceneId::Attach => {
            let resolution = 40;
            (
                ClothSpec {
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
                ColliderSpec {
                    kind: ColliderKind::Sphere,
                    radius: 0.5,
                    half_extents: Vec3::ONE,
                    position: Vec3::new(0.0, 0.5, 0.0),
                    motion: ColliderMotion::Static,
                    enabled: true,
                },
            )
        }
        SceneId::Collision => (
            ClothSpec {
                resolution: 16,
                texture: 2,
                position: Vec3::new(0.0, 2.5, 0.0),
                rotation: Vec3::ZERO,
                attached: vec![0, 16],
            },
            ColliderSpec {
                kind: ColliderKind::Sphere,
                radius: 0.6,
                half_extents: Vec3::ONE,
                position: Vec3::new(0.0, 0.6, -1.0),
                motion: ColliderMotion::SweepZ,
                enabled: true,
            },
        ),
        _ => return None,
    };
    Some(ComparisonScene {
        cloths: vec![cloth],
        colliders: vec![
            ColliderSpec {
                kind: ColliderKind::Plane,
                radius: 0.0,
                half_extents: Vec3::ONE,
                position: Vec3::ZERO,
                motion: ColliderMotion::Static,
                enabled: true,
            },
            sphere,
        ],
    })
}

/// the assets and solver state a scene build needs, grouped so the helper signatures stay short.
#[derive(SystemParam)]
pub struct SceneContext<'w, 's> {
    commands: Commands<'w, 's>,
    asset_server: Res<'w, AssetServer>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    solver: ResMut<'w, SolverRes>,
    params: ResMut<'w, SimParamsRes>,
}

/// `Scene::PopulateActors` + `onEnter` for the initial scene.
pub fn setup_initial_scene(mut context: SceneContext) {
    build_scene(&mut context, SceneId::default());
}

/// `VtEngine::SwitchScene` / `Reset`: run the current scene's `onExit`, then the next `onEnter`.
pub fn apply_scene_switch(
    mut context: SceneContext,
    mut current: ResMut<CurrentScene>,
    mut pending: ResMut<PendingScene>,
    entities: Query<Entity, With<SceneEntity>>,
    meshes: Query<&Mesh3d, With<SceneEntity>>,
    materials: Query<&MeshMaterial3d<StandardMaterial>, With<SceneEntity>>,
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
    *context.solver = SolverRes::default();

    build_scene(&mut context, next);
    current.0 = next;
}

/// a scene's actor list, shared by `Scene::PopulateActors` and the switch entry point.
fn build_scene(context: &mut SceneContext, id: SceneId) {
    // `onEnter`: the overrides must be in place *before* any cloth is added, so `AddCloth`'s
    // `maxSpeed = 2 * pd / dt * numSubsteps` sees the final `numSubsteps` (plan.org §4-6)
    *context.params = SimParamsRes(scene_params(id));

    spawn_camera_and_light(context);

    match id {
        SceneId::Attach => scene_attach(context),
        SceneId::Collision => scene_collision(context),
        SceneId::SelfCollision => scene_self_collision(context),
        SceneId::Friction => scene_friction(context),
        SceneId::Multiple => scene_multiple(context),
        SceneId::HighResolution => scene_high_resolution(context),
        SceneId::Swirl => scene_swirl(context),
    }
}

/// `SceneClothAttach`: res 40 pinned at all four corners, over a static sphere.
fn scene_attach(context: &mut SceneContext) {
    spawn_comparison_scene(context, SceneId::Attach);
}

/// `SceneClothCollision`: res 16 hanging from its top edge, a sphere sweeping `z = -cos(2t)`.
fn scene_collision(context: &mut SceneContext) {
    spawn_comparison_scene(context, SceneId::Collision);
}

/// Keep the CPU visuals and history initialization in the existing spawn helpers.
fn spawn_comparison_scene(context: &mut SceneContext, id: SceneId) {
    let scene = comparison_scene(id).expect("CPU comparison builders use supported scenes");
    for collider in scene.colliders {
        match collider.kind {
            ColliderKind::Plane => spawn_ground(context),
            ColliderKind::Sphere => spawn_sphere(
                context,
                collider.radius,
                collider.position,
                collider.motion,
                collider.enabled,
            ),
            ColliderKind::Cube => unreachable!("comparison batch contains only ground and sphere"),
        }
    }
    for cloth in scene.cloths {
        spawn_cloth(context, cloth);
    }
}

/// `SceneClothSelfCollision`: res 60, no attachment, `substeps 8` / `friction 0.3`.
fn scene_self_collision(context: &mut SceneContext) {
    spawn_ground(context);

    spawn_cloth(
        context,
        ClothSpec {
            resolution: 60,
            texture: 1,
            position: Vec3::new(0.0, 1.5, 1.0),
            rotation: Vec3::new(-15.0, 10.0, 10.0),
            attached: Vec::new(),
        },
    );
}

/// `SceneClothFriction`: res 64, a sphere sliding along `x` with a spin, `friction 0.6`.
fn scene_friction(context: &mut SceneContext) {
    spawn_ground(context);

    spawn_sphere(
        context,
        0.5,
        Vec3::new(0.0, 0.5, 0.0),
        ColliderMotion::FrictionSweep,
        true,
    );
    spawn_cloth(
        context,
        ClothSpec {
            resolution: 64,
            texture: 2,
            position: Vec3::new(0.0, 1.5, 1.0),
            rotation: Vec3::new(90.0, 0.0, 0.0),
            attached: Vec::new(),
        },
    );
}

/// `SceneClothMultiple`: three res-64 sheets sharing one solver, over a unit cube.
fn scene_multiple(context: &mut SceneContext) {
    spawn_ground(context);

    // `SpawnColoredCube` with `scale = 1.0`; the mesh is a unit cube, so the half extent is 0.5
    spawn_cube(context, 0.5, Vec3::new(0.0, 0.5, 0.0));

    for (texture, height) in [(1, 1.5), (2, 1.8), (3, 2.1)] {
        spawn_cloth(
            context,
            ClothSpec {
                resolution: 64,
                texture,
                position: Vec3::new(0.0, height, 1.0),
                rotation: Vec3::new(90.0, 0.0, 0.0),
                attached: Vec::new(),
            },
        );
    }
}

/// `SceneClothHD`: res 200, `substeps 10` / `iterations 10` (plan.org §1.7 marks this a stress case).
fn scene_high_resolution(context: &mut SceneContext) {
    spawn_ground(context);

    spawn_sphere(
        context,
        0.6,
        Vec3::new(0.0, 0.6, 0.0),
        ColliderMotion::Static,
        true,
    );
    spawn_cloth(
        context,
        ClothSpec {
            resolution: 200,
            texture: 2,
            position: Vec3::new(0.0, 1.5, 1.0),
            rotation: Vec3::new(90.0, 0.0, 0.0),
            attached: Vec::new(),
        },
    );
}

/// `SceneClothSwirl`: res 36 with a single pinned corner; the sphere rides the slot but is disabled.
fn scene_swirl(context: &mut SceneContext) {
    spawn_ground(context);

    spawn_sphere(
        context,
        0.1,
        Vec3::new(0.0, 0.1, 0.0),
        ColliderMotion::SwirlSlot,
        false,
    );
    spawn_cloth(
        context,
        ClothSpec {
            resolution: 36,
            texture: 1,
            position: Vec3::new(0.0, 1.5, 1.0),
            rotation: Vec3::new(90.0, 0.0, 0.0),
            attached: vec![0],
        },
    );
}

/// one `SpawnCloth` call.
#[derive(Clone)]
pub struct ClothSpec {
    pub resolution: u32,
    pub texture: u32,
    pub position: Vec3,
    /// the source's `Transform::rotation`, in degrees
    pub rotation: Vec3,
    /// pinned particle ids, in the source's order
    pub attached: Vec<u32>,
}

impl ClothSpec {
    /// Pose baked into particle positions by both CPU and GPU scene construction.
    pub fn pose(&self) -> Mat4 {
        source_pose(self.position, self.rotation)
    }
}

/// `Scene.hpp::SpawnCloth` + `Actor::Initialize`: the pose is baked into the mesh by `add_cloth`,
/// so the entity itself stays at `Transform::IDENTITY` (plan.org §4-1).
fn spawn_cloth(context: &mut SceneContext, spec: ClothSpec) {
    let cloth = generate_cloth_mesh(spec.resolution);
    let pose = spec.pose();

    let offset = context
        .solver
        .0
        .add_cloth(
            &mut context.params.0,
            &cloth,
            pose,
            &spec.attached,
            FIXED_DELTA,
        )
        .expect("generated meshes are source-layout grids");

    let mesh = context.meshes.add(build_cloth_mesh(&cloth));
    let material = context
        .materials
        .add(cloth_material(&context.asset_server, spec.texture));

    context.commands.spawn((
        SceneEntity,
        Mesh3d(mesh.clone()),
        MeshMaterial3d(material),
        Transform::IDENTITY,
        ClothVisual {
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
fn spawn_camera_and_light(context: &mut SceneContext) {
    // camera position (0.35, 3.3, 7.2) / euler (-21, 2.25, 0) in degrees. `OrbitCameraPlugin`
    // takes over the transform when it is added; without it the camera stays at this pose.
    let orbit = OrbitCamera::source();
    context
        .commands
        .spawn((SceneEntity, Camera3d::default(), orbit.transform(), orbit));

    // spot light at (2.5, 5, 2.5). the source uses its own euler convention for the direction, so
    // we aim at the cloth instead.
    context
        .commands
        .spawn((SceneEntity, comparison_spotlight()));
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
fn spawn_ground(context: &mut SceneContext) {
    let mesh = context
        .meshes
        .add(Plane3d::default().mesh().size(50.0, 50.0));
    let material = context.materials.add(comparison_ground_material());

    context.commands.spawn((
        SceneEntity,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::IDENTITY,
        ClothCollider {
            kind: ColliderKind::Plane,
            radius: 0.0,
            half_extents: Vec3::ONE,
            motion: ColliderMotion::Static,
            enabled: true,
        },
        ColliderHistory(Mat4::IDENTITY),
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
    context: &mut SceneContext,
    radius: f32,
    position: Vec3,
    motion: ColliderMotion,
    enabled: bool,
) {
    let mesh = context.meshes.add(Sphere::new(radius));
    let material = context.materials.add(comparison_collider_material());

    context.commands.spawn((
        SceneEntity,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_translation(position),
        ClothCollider {
            kind: ColliderKind::Sphere,
            radius,
            half_extents: Vec3::ONE,
            motion,
            enabled,
        },
        // a fresh collider starts with zero relative velocity (`Collider::Start`)
        ColliderHistory(Mat4::from_translation(position)),
    ));
}

/// `Scene.hpp::SpawnColoredCube`. `half_extent` is the world half size; `size` doubles it back into
/// the source's unit-cube `transform.scale`.
fn spawn_cube(context: &mut SceneContext, half_extent: f32, position: Vec3) {
    let mesh = context.meshes.add(Cuboid::new(
        half_extent * 2.0,
        half_extent * 2.0,
        half_extent * 2.0,
    ));
    let material = context.materials.add(comparison_collider_material());

    context.commands.spawn((
        SceneEntity,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_translation(position),
        ClothCollider {
            kind: ColliderKind::Cube,
            radius: 0.0,
            half_extents: Vec3::splat(half_extent),
            motion: ColliderMotion::Static,
            enabled: true,
        },
        ColliderHistory(Mat4::from_translation(position)),
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
