//! The seven Velvet fixtures with GLB meshes and the production ClothPlugin.

mod panel;
mod scene;

use bevy::{
    asset::AssetPlugin,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::allocator::MeshAllocator,
        pipelined_rendering::PipelinedRenderingPlugin,
        render_resource::{CachedPipelineState, PipelineCache, PipelineDescriptor},
    },
    shader::ShaderCacheError,
    time::{TimeSystems, TimeUpdateStrategy},
};

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use bevy_cloth::{
    ClothPlugin, ClothSolver,
    debug::{ClothDebugPlugin, timing::ClothGpuTimings},
    legacy::demo::{
        LEGACY_FIXED_HZ, LegacyOrbitCamera, LegacyOrbitCameraPlugin, LegacySceneId,
        legacy_scene_params,
    },
};
use scene::{SceneAssets, SceneEntity, SceneMotion, VbdOptions};

#[derive(Clone, Copy, PartialEq, Eq)]
struct StepBudget {
    substeps: u32,
    sweeps: u32,
}

impl StepBudget {
    fn for_scene(scene: LegacySceneId) -> Self {
        let params = legacy_scene_params(scene);
        Self {
            substeps: params.num_substeps,
            sweeps: params.num_iterations,
        }
    }
}

#[derive(Resource)]
struct Demo {
    scene: LegacySceneId,
    solver: ClothSolver,
    vbd_options: VbdOptions,
    budget: StepBudget,
    pending_budget: StepBudget,
    rebuild: bool,
    playing: bool,
    step: bool,
    tick: u64,
    timing_enabled: bool,
}

impl Default for Demo {
    fn default() -> Self {
        let scene = LegacySceneId::Attach;
        let budget = StepBudget::for_scene(scene);
        Self {
            scene,
            solver: ClothSolver::ColoredGs,
            vbd_options: VbdOptions::default(),
            budget,
            pending_budget: budget,
            rebuild: true,
            playing: false,
            step: false,
            tick: 0,
            timing_enabled: false,
        }
    }
}

impl Demo {
    fn reset(&mut self, timings: &mut ClothGpuTimings) {
        timings.set_target(None);
        self.pending_budget = self.budget;
        self.rebuild = true;
        self.playing = false;
        self.step = false;
        self.tick = 0;
    }
}

#[derive(Clone, Resource, ExtractResource)]
#[extract_app(RenderApp)]
struct SceneReady {
    cloth: Entity,
    mesh: AssetId<Mesh>,
    solver: ClothSolver,
    has_dynamic_pin: bool,
    has_self_contact: bool,
    ready: Arc<AtomicBool>,
}

fn main() -> AppExit {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(AssetPlugin {
                file_path: Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("assets")
                    .display()
                    .to_string(),
                ..default()
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Velvet GLB scenes — ClothPlugin".into(),
                    resolution: (1280, 850).into(),
                    ..default()
                }),
                ..default()
            })
            .disable::<PipelinedRenderingPlugin>(),
    )
    .add_plugins((
        ClothPlugin,
        ClothDebugPlugin,
        LegacyOrbitCameraPlugin,
        ExtractResourcePlugin::<SceneReady>::default(),
    ))
    .init_resource::<Demo>()
    .insert_resource(TimeUpdateStrategy::FixedTimesteps(0))
    .insert_resource(ClearColor(Color::srgb(0.04, 0.05, 0.08)))
    .insert_resource(GlobalAmbientLight {
        brightness: 300.0,
        ..default()
    })
    .add_systems(Startup, (setup, panel::spawn, panel::spawn_timing_panel))
    .add_systems(First, set_tick_budget.before(TimeSystems))
    .add_systems(
        Update,
        (
            handle_keys,
            rebuild_scene,
            update_hud,
            panel::highlight_selection,
            panel::update_budget_controls,
            panel::update_timings,
        )
            .chain(),
    )
    .add_systems(FixedUpdate, animate_scene);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .set_timestep_hz(f64::from(LEGACY_FIXED_HZ));
    app.sub_app_mut(RenderApp)
        .add_systems(Render, check_scene_ready.after(RenderSystems::Render));
    app.run()
}

fn setup(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(SceneAssets::load(&server, &mut materials));
    let orbit = LegacyOrbitCamera::source();
    commands.spawn((Camera3d::default(), orbit.transform(), orbit));
    commands.spawn((
        DirectionalLight {
            illuminance: 9_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(3.0, 5.0, 4.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}

fn handle_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut demo: ResMut<Demo>,
    mut timings: ResMut<ClothGpuTimings>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        demo.reset(&mut timings);
    } else if keys.just_pressed(KeyCode::Space) {
        demo.playing = !demo.playing;
    } else if keys.just_pressed(KeyCode::KeyS) {
        demo.playing = false;
        demo.step = true;
    }
}

fn rebuild_scene(
    mut commands: Commands,
    mut demo: ResMut<Demo>,
    scene_assets: Res<SceneAssets>,
    server: Res<AssetServer>,
    old_entities: Query<Entity, With<SceneEntity>>,
    previous: Option<Res<SceneReady>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut timings: ResMut<ClothGpuTimings>,
) -> Result {
    if !demo.rebuild {
        return Ok(());
    }
    if let Some(previous) = previous {
        timings.set_target(None);
        for entity in &old_entities {
            commands.entity(entity).despawn();
        }
        meshes.remove(previous.mesh);
        commands.remove_resource::<SceneReady>();
    }
    if let Some((cloth, mesh)) = scene::spawn_scene(
        &mut commands,
        &scene_assets,
        &server,
        &mut meshes,
        demo.scene,
        demo.solver,
        demo.vbd_options,
        demo.budget,
    )? {
        demo.rebuild = false;
        if demo.timing_enabled {
            timings.set_target(Some(cloth));
        }
        commands.insert_resource(SceneReady {
            cloth,
            mesh,
            solver: demo.solver,
            has_dynamic_pin: demo.scene == LegacySceneId::Swirl,
            has_self_contact: demo.solver != ClothSolver::Vbd || demo.vbd_options.self_contact,
            ready: Arc::new(AtomicBool::new(false)),
        });
    }
    Ok(())
}

fn set_tick_budget(
    mut demo: ResMut<Demo>,
    ready: Option<Res<SceneReady>>,
    fixed: Res<Time<Fixed>>,
    mut clock: ResMut<TimeUpdateStrategy>,
    mut next_tick: Local<Option<Instant>>,
) {
    let ready = !demo.rebuild && ready.is_some_and(|scene| scene.ready.load(Ordering::Acquire));
    let due = if ready && demo.playing {
        let now = Instant::now();
        let deadline = next_tick.get_or_insert(now);
        if now >= *deadline {
            // Never batch catch-up ticks: ClothPlugin samples one collider/pin pose per frame.
            *deadline = (*deadline + fixed.timestep()).max(now);
            true
        } else {
            false
        }
    } else {
        *next_tick = None;
        false
    };
    let advance = ready && (due || demo.step);
    if advance {
        demo.step = false;
    }
    *clock = TimeUpdateStrategy::FixedTimesteps(u32::from(advance));
}

fn animate_scene(mut demo: ResMut<Demo>, mut movers: Query<(&SceneMotion, &mut Transform)>) {
    // A UI click can request a rebuild after First has already budgeted a tick.
    if demo.rebuild {
        return;
    }
    demo.tick += 1;
    let time = (demo.tick as f64 / LEGACY_FIXED_HZ) as f32;
    for (motion, mut transform) in &mut movers {
        *transform = Transform::from_matrix(motion.0.transform_at(time));
    }
}

fn check_scene_ready(
    scene: Option<Res<SceneReady>>,
    cache: Res<PipelineCache>,
    allocator: Res<MeshAllocator>,
    mut waiting: Local<Option<(AssetId<Mesh>, Instant)>>,
) {
    let Some(scene) = scene else { return };
    if scene.ready.load(Ordering::Acquire) {
        return;
    }
    if waiting.is_none_or(|(mesh, _)| mesh != scene.mesh) {
        *waiting = Some((scene.mesh, Instant::now()));
    }
    let common = [
        "xpbd update_velocity",
        "xpbd output_mesh",
        "xpbd collider transform_vertices",
        "xpbd collider refit_leaves",
        "xpbd collider refit_internal",
    ];
    let xpbd = ["xpbd predict", "stabilize_contact", "solve_contact"];
    let self_contact = [
        "cloth self-contact clear_hash",
        "cloth self-contact count_hash",
        "cloth self-contact scan_blocks",
        "cloth self-contact add_block_offsets",
        "cloth self-contact scatter_hash",
        "cloth self-contact find_candidates",
        "cloth self-contact gather_self_contact",
        "cloth self-contact apply_self_contact",
    ];
    let solver: &[&str] = match scene.solver {
        ClothSolver::ColoredGs => &["xpbd solve_stretch", "xpbd solve_bending"],
        ClothSolver::Jacobi => &["solve_stretch_jacobi", "solve_bend_jacobi", "apply_jacobi"],
        ClothSolver::Vbd => &[
            "predict_vbd",
            "solve_vbd_color",
            "reset_vbd_contacts",
            "gather_vbd_contacts",
        ],
    };
    let required = common
        .into_iter()
        .chain(solver.iter().copied())
        .chain(
            xpbd.into_iter()
                .filter(|_| scene.solver != ClothSolver::Vbd),
        )
        .chain(self_contact.into_iter().filter(|_| scene.has_self_contact))
        .chain(
            scene
                .has_dynamic_pin
                .then_some("xpbd apply attachment targets"),
        );
    let mut pending = Vec::new();
    for label in required {
        let state = cache.pipelines().find_map(|p| {
            let PipelineDescriptor::ComputePipelineDescriptor(descriptor) = &p.descriptor else {
                return None;
            };
            (descriptor.label.as_deref() == Some(label)).then_some(&p.state)
        });
        match state {
            Some(CachedPipelineState::Ok(_)) => (),
            Some(CachedPipelineState::Err(
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable,
            )) => pending.push(label),
            Some(CachedPipelineState::Err(error)) => panic!("{label}: {error}"),
            _ => pending.push(label),
        }
    }
    let mesh_ready = allocator.mesh_vertex_slice(&scene.mesh).is_some();
    if pending.is_empty() && mesh_ready {
        // Checked after the render graph, while fixed ticks are still paused.
        scene.ready.store(true, Ordering::Release);
    } else if waiting.is_some_and(|(_, since)| since.elapsed().as_secs() >= 30) {
        panic!(
            "Timed out preparing scene; pending GPU pipelines: {pending:?}; mesh allocated: {mesh_ready}"
        );
    }
}

fn update_hud(
    demo: Res<Demo>,
    ready: Option<Res<SceneReady>>,
    mut hud: Single<&mut Text, With<panel::Hud>>,
) {
    let state = if demo.rebuild {
        "Loading GLBs"
    } else if !ready.is_some_and(|scene| scene.ready.load(Ordering::Acquire)) {
        "Preparing GPU"
    } else if demo.playing {
        "Playing"
    } else {
        "Paused"
    };
    let features = if demo.solver == ClothSolver::Vbd {
        let friction = if demo.vbd_options.friction {
            legacy_scene_params(demo.scene).friction
        } else {
            0.0
        };
        let frozen = if matches!(
            demo.scene,
            LegacySceneId::Collision | LegacySceneId::Friction
        ) {
            "; sphere frozen"
        } else {
            ""
        };
        let self_contact = if demo.vbd_options.self_contact {
            "on"
        } else {
            "off"
        };
        format!(
            "VBD: static contact, friction μ={friction:.2}, self-contact {self_contact}{frozen}"
        )
    } else {
        "XPBD: mesh contact, friction, particle self-contact".to_owned()
    };
    hud.0 = format!(
        "{}  |  {:?}  |  {state}, tick {}\nActive: {} substeps/tick × {} sweeps/substep\n{features}\nSpace play/pause  S step  R reset",
        demo.scene.label(),
        demo.solver,
        demo.tick,
        demo.budget.substeps,
        demo.budget.sweeps,
    );
}
