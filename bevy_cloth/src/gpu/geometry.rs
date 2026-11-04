//! Analytic world-space collider positions, evaluated from immutable initial vertices.
//! The parent plugin registers `geometry.wesl` and initializes `PositionPipeline`.

use bevy::{
    asset::load_embedded_asset,
    material::descriptor::{
        BindGroupLayoutDescriptor, CachedComputePipelineId, ComputePipelineDescriptor,
    },
    prelude::*,
    render::{
        render_resource::{
            BindGroupEntries, BindGroupLayoutEntries, Buffer, BufferInitDescriptor, BufferUsages,
            CommandEncoderDescriptor, ComputePassDescriptor, PipelineCache, ShaderStages,
            binding_types::{
                storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer_sized,
            },
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};

/// Analytic motion relative to the initial world geometry, not the render Transform.
/// Angular velocity is a world-space axis times radians/second. The wave adds world-Z
/// displacement after rigid motion; frequency is radians/second and spatial phase
/// is `rest.x - pivot.x`. This is a test deformation provider, not skinning.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct ColliderMotion {
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub deform_amplitude: f32,
    pub deform_frequency: f32,
}

impl ColliderMotion {
    pub fn validate(self) -> Result<(), String> {
        if !self.linear_velocity.is_finite()
            || !self.angular_velocity.is_finite()
            || !self.angular_velocity.length().is_finite()
            || !self.deform_amplitude.is_finite()
            || !self.deform_frequency.is_finite()
        {
            return Err("collider motion must be finite with representable angular speed".into());
        }
        Ok(())
    }

    fn model(self, pivot: Vec3, time: f32) -> Option<Mat4> {
        self.validate().ok()?;
        if !pivot.is_finite() || !time.is_finite() || time < 0.0 {
            return None;
        }
        let speed = self.angular_velocity.length();
        let angle = speed * time;
        if !angle.is_finite() || !(self.deform_frequency * time).is_finite() {
            return None;
        }
        let rotation = if speed > 0.0 {
            Quat::from_axis_angle(self.angular_velocity / speed, angle)
        } else {
            Quat::IDENTITY
        };
        let model = Mat4::from_translation(self.linear_velocity * time + pivot)
            * Mat4::from_quat(rotation)
            * Mat4::from_translation(-pivot);
        model.is_finite().then_some(model)
    }
}

/// Shared analytic sampler for CPU diagnostics and imported rigid attachment targets;
/// production collider positions and their history stay on GPU. Sampling a rest point
/// with deformation is not a barycentric surface binding (attachments reject deformation).
/// Invalid inputs or unrepresentable outputs return None. At t=0 this returns rest.
/// Contact surface displacement is the barycentric interpolation of current minus
/// previous positions; dividing by fixed dt gives the discrete surface velocity.
pub fn sample_position(rest: Vec3, pivot: Vec3, motion: ColliderMotion, time: f32) -> Option<Vec3> {
    if !rest.is_finite() {
        return None;
    }
    let model = motion.model(pivot, time)?;
    let phase = rest.x - pivot.x;
    let wave =
        motion.deform_amplitude * ((motion.deform_frequency * time + phase).sin() - phase.sin());
    let position = model.transform_point3(rest) + Vec3::Z * wave;
    position.is_finite().then_some(position)
}

/// Index is the completed fixed tick within an epoch; zero denotes initial state.
/// Times never come from display-frame delta or the last rendered position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GeometryTick {
    pub epoch: u64,
    pub index: u64,
    pub dt: f32,
}

impl GeometryTick {
    pub fn times(self) -> Option<(f32, f32)> {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return None;
        }
        let current = (self.index as f64 * f64::from(self.dt)) as f32;
        let previous = (self.index.saturating_sub(1) as f64 * f64::from(self.dt)) as f32;
        (current.is_finite() && previous.is_finite()).then_some((current, previous))
    }
}

/// Optional submission gate: duplicate/older ticks and dt changes within an epoch
/// are rejected. Commit only after update succeeds (call consume on a copied gate).
/// A new epoch resets the gate; analytic history still means index-1, not last frame.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TickHistory {
    last: Option<GeometryTick>,
}

impl TickHistory {
    pub fn consume(&mut self, stamp: GeometryTick) -> Option<(f32, f32)> {
        let times = stamp.times()?;
        if self.last.is_some_and(|last| {
            last.epoch == stamp.epoch && (stamp.index <= last.index || stamp.dt != last.dt)
        }) {
            return None;
        }
        self.last = Some(stamp);
        Some(times)
    }
}

const WORKGROUP_SIZE: u32 = 256;

/// Exact uniform ABI: matrices 0/64, wave 128, pivot 144, count/padding 160..176.
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PositionParams {
    current_model: [[f32; 4]; 4],
    previous_model: [[f32; 4]; 4],
    wave: [f32; 4],
    pivot: [f32; 4],
    count: u32,
    pad: [u32; 3],
}

fn supports_positions(limits: &wgpu::Limits) -> bool {
    limits.max_storage_buffers_per_shader_stage >= 3
        && limits.max_uniform_buffers_per_shader_stage >= 1
        && limits.max_bind_groups >= 1
        && limits.max_bindings_per_bind_group >= 4
        && limits.max_compute_invocations_per_workgroup >= WORKGROUP_SIZE
        && limits.max_compute_workgroup_size_x >= WORKGROUP_SIZE
        && limits.max_uniform_buffer_binding_size >= size_of::<PositionParams>() as _
        && limits.max_buffer_size >= size_of::<PositionParams>() as u64
}

#[derive(Resource)]
pub(super) struct PositionPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: Option<CachedComputePipelineId>,
}

impl FromWorld for PositionPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "geometry.wesl");
        let layout = BindGroupLayoutDescriptor::new(
            "collider analytic positions",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(false, None),
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let pipeline = supports_positions(&world.resource::<RenderDevice>().limits()).then(|| {
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("collider analytic positions".into()),
                    layout: vec![layout.clone()],
                    shader,
                    entry_point: Some("update_positions".into()),
                    ..default()
                })
        });
        Self { layout, pipeline }
    }
}

impl PositionPipeline {
    pub(super) fn ready(&self, cache: &PipelineCache) -> bool {
        self.pipeline
            .and_then(|id| cache.get_compute_pipeline(id))
            .is_some()
    }
}

fn position_capacity(count: usize, limits: &wgpu::Limits) -> Result<u32, String> {
    let count = u32::try_from(count).map_err(|_| "collider vertex count exceeds u32")?;
    let bytes = u64::from(count) * 16;
    if count == 0
        || !supports_positions(limits)
        || bytes > limits.max_buffer_size
        || bytes > u64::from(limits.max_storage_buffer_binding_size)
        || count.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension
    {
        return Err("collider positions exceed device storage/dispatch limits or are empty".into());
    }
    Ok(count)
}

/// Fixed-topology world positions. All three buffers start at rest; updates evaluate
/// both times from rest, never from the last display frame or accumulated motion.
/// The caller owns tick gating and must not consume a tick when update returns false.
/// Updating these buffers does not refit a BVH or update a render mesh.
pub(super) struct GpuPositions {
    rest: Buffer,
    current: Buffer,
    previous: Buffer,
    count: u32,
}

impl GpuPositions {
    pub(super) fn new(device: &RenderDevice, rest: &[[f32; 4]]) -> Result<Self, String> {
        let count = position_capacity(rest.len(), &device.limits())?;
        if rest.iter().flatten().any(|value| !value.is_finite()) {
            return Err("collider rest positions must be finite".into());
        }
        let upload = |label| {
            device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(rest),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            })
        };
        Ok(Self {
            rest: upload("collider rest positions"),
            current: upload("collider current positions"),
            previous: upload("collider previous positions"),
            count,
        })
    }

    pub(super) fn rest(&self) -> &Buffer {
        &self.rest
    }

    pub(super) fn current(&self) -> &Buffer {
        &self.current
    }

    pub(super) fn previous(&self) -> &Buffer {
        &self.previous
    }

    pub(super) fn count(&self) -> u32 {
        self.count
    }

    /// Submits before subsequent contact/refit work on the same queue. False means
    /// invalid motion/time or a pipeline that is not ready; no buffers are changed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn update(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
        pipeline: &PositionPipeline,
        motion: ColliderMotion,
        pivot: Vec3,
        time: f32,
        previous_time: f32,
    ) -> bool {
        let Some(compute) = pipeline
            .pipeline
            .and_then(|id| cache.get_compute_pipeline(id))
        else {
            return false;
        };
        let (Some(current), Some(previous)) = (
            motion.model(pivot, time),
            motion.model(pivot, previous_time),
        ) else {
            return false;
        };
        if previous_time > time {
            return false;
        }
        let params = PositionParams {
            current_model: current.to_cols_array_2d(),
            previous_model: previous.to_cols_array_2d(),
            wave: [
                motion.deform_amplitude,
                motion.deform_frequency * time,
                motion.deform_frequency * previous_time,
                0.0,
            ],
            pivot: pivot.extend(0.0).to_array(),
            count: self.count,
            pad: [0; 3],
        };
        let uniform = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("collider position parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(
            "collider analytic positions",
            &cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((
                uniform.as_entire_binding(),
                self.rest.as_entire_binding(),
                self.current.as_entire_binding(),
                self.previous.as_entire_binding(),
            )),
        );
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("collider analytic positions"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("collider analytic positions"),
                ..default()
            });
            pass.set_pipeline(compute);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit([encoder.finish()]);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of};

    fn close(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).length() < 2e-5,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn translation_and_discrete_surface_velocity() {
        let rest = Vec3::new(2.0, -3.0, 4.0);
        let motion = ColliderMotion {
            linear_velocity: Vec3::new(3.0, -2.0, 1.0),
            ..default()
        };
        let current = sample_position(rest, Vec3::ONE, motion, 0.5).unwrap();
        let previous = sample_position(rest, Vec3::ONE, motion, 0.25).unwrap();
        close(current, Vec3::new(3.5, -4.0, 4.5));
        close((current - previous) / 0.25, motion.linear_velocity);
    }

    #[test]
    fn rotation_uses_world_axis_and_pivot() {
        let pivot = Vec3::new(3.0, 4.0, 5.0);
        let motion = ColliderMotion {
            angular_velocity: Vec3::Z * std::f32::consts::FRAC_PI_2,
            ..default()
        };
        close(
            sample_position(pivot + Vec3::X * 2.0, pivot, motion, 1.0).unwrap(),
            pivot + Vec3::Y * 2.0,
        );
        close(sample_position(pivot, pivot, motion, 7.0).unwrap(), pivot);
    }

    #[test]
    fn deformation_is_rest_relative_and_added_after_rigid_motion() {
        let pivot = Vec3::new(1.0, 2.0, 3.0);
        let rest = pivot + Vec3::X * std::f32::consts::FRAC_PI_2;
        let motion = ColliderMotion {
            linear_velocity: Vec3::X * 2.0,
            angular_velocity: Vec3::Z * std::f32::consts::FRAC_PI_2,
            deform_amplitude: 0.5,
            deform_frequency: std::f32::consts::FRAC_PI_2,
        };
        close(sample_position(rest, pivot, motion, 0.0).unwrap(), rest);
        close(
            sample_position(rest, pivot, motion, 1.0).unwrap(),
            pivot + Vec3::X * 2.0 + Vec3::Y * std::f32::consts::FRAC_PI_2 - Vec3::Z * 0.5,
        );
        let stationary = ColliderMotion {
            deform_frequency: 0.0,
            ..motion
        };
        let rigid = ColliderMotion {
            deform_amplitude: 0.0,
            ..motion
        };
        close(
            sample_position(rest, pivot, stationary, 1.0).unwrap(),
            sample_position(rest, pivot, rigid, 1.0).unwrap(),
        );
    }

    #[test]
    fn invalid_motion_and_times_are_rejected() {
        let valid = ColliderMotion::default();
        for time in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(sample_position(Vec3::ZERO, Vec3::ZERO, valid, time).is_none());
        }
        for motion in [
            ColliderMotion {
                linear_velocity: Vec3::NAN,
                ..valid
            },
            ColliderMotion {
                angular_velocity: Vec3::splat(f32::MAX),
                ..valid
            },
            ColliderMotion {
                deform_amplitude: f32::NAN,
                ..valid
            },
            ColliderMotion {
                deform_frequency: f32::INFINITY,
                ..valid
            },
        ] {
            assert!(motion.validate().is_err());
        }
        assert!(sample_position(Vec3::NAN, Vec3::ZERO, valid, 0.0).is_none());
        assert!(sample_position(Vec3::ZERO, Vec3::NAN, valid, 0.0).is_none());
        let overflow = ColliderMotion {
            linear_velocity: Vec3::splat(f32::MAX),
            ..valid
        };
        assert!(sample_position(Vec3::ZERO, Vec3::ZERO, overflow, 2.0).is_none());
    }

    #[test]
    fn zero_single_multi_tick_frames_and_reset_keep_tick_history() {
        let stamp = |epoch, index| GeometryTick {
            epoch,
            index,
            dt: 0.25,
        };
        let mut history = TickHistory::default();
        let motion = ColliderMotion {
            linear_velocity: Vec3::X * 2.0,
            ..default()
        };
        let pair = |(now, old)| {
            (
                sample_position(Vec3::ZERO, Vec3::ZERO, motion, now).unwrap(),
                sample_position(Vec3::ZERO, Vec3::ZERO, motion, old).unwrap(),
            )
        };
        assert_eq!(
            pair(history.consume(stamp(0, 0)).unwrap()),
            (Vec3::ZERO, Vec3::ZERO)
        );
        // A frame with no tick performs no update; duplicate submission is rejected.
        assert!(history.consume(stamp(0, 0)).is_none());
        assert_eq!(
            pair(history.consume(stamp(0, 1)).unwrap()),
            (Vec3::X * 0.5, Vec3::ZERO)
        );
        assert_eq!(
            pair(history.consume(stamp(0, 2)).unwrap()),
            (Vec3::X, Vec3::X * 0.5)
        );
        assert_eq!(
            pair(history.consume(stamp(0, 3)).unwrap()),
            (Vec3::X * 1.5, Vec3::X)
        );
        // Even a skipped submission evaluates index-1, not the last submitted index.
        assert_eq!(
            pair(history.consume(stamp(0, 7)).unwrap()),
            (Vec3::X * 3.5, Vec3::X * 3.0)
        );
        assert!(history.consume(stamp(0, 6)).is_none());
        assert!(
            history
                .consume(GeometryTick {
                    dt: 0.5,
                    ..stamp(0, 8)
                })
                .is_none()
        );
        assert_eq!(
            pair(history.consume(stamp(1, 0)).unwrap()),
            (Vec3::ZERO, Vec3::ZERO)
        );
        assert_eq!(history.consume(stamp(1, 1)), Some((0.25, 0.0)));
    }

    #[test]
    fn invalid_tick_does_not_advance_history() {
        let mut history = TickHistory::default();
        for dt in [0.0, -0.25, f32::NAN, f32::INFINITY, f32::MAX] {
            assert!(
                history
                    .consume(GeometryTick {
                        epoch: 0,
                        index: 2,
                        dt
                    })
                    .is_none()
            );
        }
        assert_eq!(
            history.consume(GeometryTick {
                epoch: 0,
                index: 2,
                dt: 0.25
            }),
            Some((0.5, 0.25))
        );
    }

    #[test]
    fn uniform_abi_matches_shader() {
        assert_eq!(size_of::<PositionParams>(), 176);
        assert_eq!(align_of::<PositionParams>(), 16);
        assert_eq!(offset_of!(PositionParams, current_model), 0);
        assert_eq!(offset_of!(PositionParams, previous_model), 64);
        assert_eq!(offset_of!(PositionParams, wave), 128);
        assert_eq!(offset_of!(PositionParams, pivot), 144);
        assert_eq!(offset_of!(PositionParams, count), 160);
        assert_eq!(offset_of!(PositionParams, pad), 164);
    }

    #[test]
    fn capacity_checks_storage_allocation_dispatch_and_features() {
        let limits = wgpu::Limits::default();
        assert_eq!(position_capacity(257, &limits), Ok(257));
        assert!(position_capacity(0, &limits).is_err());
        if usize::BITS > 32 {
            assert!(position_capacity(u32::MAX as usize + 1, &limits).is_err());
        }
        let mut small = limits.clone();
        small.max_storage_buffer_binding_size = 16;
        assert_eq!(position_capacity(1, &small), Ok(1));
        assert!(position_capacity(2, &small).is_err());
        small = limits.clone();
        small.max_buffer_size = 176;
        assert!(position_capacity(12, &small).is_err());
        small = limits.clone();
        small.max_compute_workgroups_per_dimension = 1;
        assert_eq!(position_capacity(256, &small), Ok(256));
        assert!(position_capacity(257, &small).is_err());
        for restricted in 0..5 {
            small = limits.clone();
            match restricted {
                0 => small.max_storage_buffers_per_shader_stage = 2,
                1 => small.max_uniform_buffer_binding_size = 175,
                2 => small.max_compute_workgroup_size_x = 255,
                3 => small.max_compute_invocations_per_workgroup = 255,
                _ => small.max_bindings_per_bind_group = 3,
            }
            assert!(position_capacity(1, &small).is_err());
        }
    }
}
