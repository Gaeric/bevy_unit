//! the example fork from avian kinematic_character_3d

use avian3d::{math::*, prelude::*};
use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            PhysicsPlugins::default(),
            PhysicsDebugPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_event::<MovementAction>()
        .add_systems(
            Update,
            (
                keyboard_input,
                update_grounded,
                apply_gravity,
                movement,
                apply_movement_damping,
            )
                .chain(),
        )
        .add_systems(
            PhysicsSchedule,
            kinematic_controller_collisions.in_set(NarrowPhaseSet::Last),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Capsule3d::new(0.4, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(0.0, 5.0, 0.0),
        CharacterControllerBundle::new(Collider::capsule(0.4, 1.0), Vector::NEG_Y * 9.81 * 0.5)
            .with_movement(30.0, 0.92, 7.0, (30.0 as Scalar).to_radians()),
    ));

    // commands.spawn((
    //     SceneRoot(asset_server.load("waltz/ani_model_1.0_20250608.glb#Scene0")),
    //     Transform::from_xyz(1.0, 5.0, 0.0),
    //     CharacterControllerBundle::new(
    //         Collider::capsule_endpoints(0.5, 0.5 * Vector::Y, 1.2 * Vector::Y),
    //         Vector::NEG_Y * 9.81 * 0.5,
    //     )
    //     .with_movement(30.0, 0.92, 7.0, (30.0 as Scalar).to_radians()),
    // ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.0, 0.0, 0.0))),
        Collider::cuboid(50.0, 0.1, 50.0),
        RigidBody::Static,
    ));

    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            range: 50.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 15.0, 0.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-7.0, 9.5, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// An event sent for a movement input action.
#[derive(Event)]
pub enum MovementAction {
    Move(Vector2),
    Jump,
}

/// A marker component indicating that an entity is using a character controller.
#[derive(Component, Debug)]
pub struct CharacterController;

/// The maximum angle a slope can have for a character controller
/// to be able to climb and jump. If the slope is steeper than this angle,
/// the character will slide down.
#[derive(Component)]
pub struct MaxSlopeAngle(Scalar);

/// The acceleration used for character movement.
#[derive(Component)]
pub struct MovementAcceleration(Scalar);

/// The strength of a jump.
#[derive(Component)]
pub struct JumpImpulse(Scalar);

/// A marker component indicating that an entity is on the ground.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct Grounded;

/// The gravitational acceleration used for a character controller.
#[derive(Component)]
pub struct ControllerGravity(Vector);

/// The damping factor used for slowing down movement
#[derive(Component)]
pub struct MovementDampingFactor(Scalar);

/// A bundle that contains the components needed for a basic
/// kinematic character controller.
#[derive(Bundle)]
pub struct CharacterControllerBundle {
    character_controller: CharacterController,
    body: RigidBody,
    collider: Collider,
    ground_caster: ShapeCaster,
    gravity: ControllerGravity,
    movement: MovementBundle,
}

/// A boundle that contains components for character movement.
#[derive(Bundle)]
pub struct MovementBundle {
    acceleration: MovementAcceleration,
    damping: MovementDampingFactor,
    jump_impulse: JumpImpulse,
    max_slope_angle: MaxSlopeAngle,
}

impl MovementBundle {
    pub const fn new(
        acceleration: Scalar,
        damping: Scalar,
        jump_impulse: Scalar,
        max_slope_angle: Scalar,
    ) -> Self {
        Self {
            acceleration: MovementAcceleration(acceleration),
            damping: MovementDampingFactor(damping),
            jump_impulse: JumpImpulse(jump_impulse),
            max_slope_angle: MaxSlopeAngle(max_slope_angle),
        }
    }
}

impl Default for MovementBundle {
    fn default() -> Self {
        Self::new(30.0, 0.9, 7.0, PI * 0.45)
    }
}

impl CharacterControllerBundle {
    pub fn new(collider: Collider, gravity: Vector) -> Self {
        // Create shape caster as a lightly smaller version of collider
        let mut caster_shape = collider.clone();
        caster_shape.set_scale(Vector::ONE * 0.99, 10);

        Self {
            character_controller: CharacterController,
            body: RigidBody::Kinematic,
            collider,
            ground_caster: ShapeCaster::new(
                caster_shape,
                Vector::ZERO,
                Quaternion::default(),
                Dir3::NEG_Y,
            )
            .with_max_distance(0.2),
            gravity: ControllerGravity(gravity),
            movement: MovementBundle::default(),
        }
    }

    pub fn with_movement(
        mut self,
        acceleration: Scalar,
        damping: Scalar,
        jump_impulse: Scalar,
        max_slope_angle: Scalar,
    ) -> Self {
        self.movement = MovementBundle::new(acceleration, damping, jump_impulse, max_slope_angle);
        self
    }
}

/// Kinematic bodies do not get pushed by collisions by default,
/// so it needs to be done manually.
///
/// This system handles collision response for kinematic character controllers
/// by pushing them along their contact normals by the current penetration depth,
/// and applying velocity corrections in order to snap to slopes, slide along walls,
/// and predict collisions using speculative contacts.
fn kinematic_controller_collisions(
    collisions: Collisions,
    bodies: Query<&RigidBody>,
    collider_rbs: Query<&ColliderOf, Without<Sensor>>,
    mut character_controllers: Query<
        (&mut Position, &mut LinearVelocity, Option<&MaxSlopeAngle>),
        (With<RigidBody>, With<CharacterController>),
    >,
    time: Res<Time>,
) {
    // Iterate through collisions and move the kinematic body to resolve penetration
    for contacts in collisions.iter() {
        // Get the rigid body entities of the colliders (colliders could be children)
        let Ok([&ColliderOf { body: rb1 }, &ColliderOf { body: rb2 }]) =
            collider_rbs.get_many([contacts.collider1, contacts.collider2])
        else {
            continue;
        };

        // Get the body of the character controller and whether it is the first
        // or second entity in the collision.
        let is_first: bool;

        let character_rb: RigidBody;
        let is_other_dynamic: bool;

        let (mut position, mut linear_velocity, max_slope_angle) =
            if let Ok(character) = character_controllers.get_mut(rb1) {
                is_first = true;
                character_rb = *bodies.get(rb1).unwrap();
                is_other_dynamic = bodies.get(rb2).is_ok_and(|rb| rb.is_dynamic());
                character
            } else if let Ok(character) = character_controllers.get_mut(rb2) {
                is_first = false;
                character_rb = *bodies.get(rb2).unwrap();
                is_other_dynamic = bodies.get(rb1).is_ok_and(|rb| rb.is_dynamic());
                character
            } else {
                continue;
            };

        // This system only handless collision response for kinematic character controllers.
        if !character_rb.is_kinematic() {
            continue;
        }

        // Iterate through contact manifolds and their contacts.
        // Each contact in a single manifold shares the same contact normal.
        for manifold in contacts.manifolds.iter() {
            let normal = if is_first {
                -manifold.normal
            } else {
                manifold.normal
            };

            let mut deepest_penetration: Scalar = Scalar::MIN;

            // Slove each penetrating contact in the manifold.
            for contact in manifold.points.iter() {
                if contact.penetration > 0.0 {
                    position.0 += normal * contact.penetration;
                }
                deepest_penetration = deepest_penetration.max(contact.penetration);
            }

            // For now, this system only handles velocity corrections for collisions against static geometry.
            if is_other_dynamic {
                continue;
            }

            // Determine if the slope is climbable or if it's too steep to walk on.
            let slope_angle = normal.angle_between(Vector::Y);
            let climbable = max_slope_angle.is_some_and(|angle| slope_angle.abs() <= angle.0);

            if deepest_penetration > 0.0 {
                // If the slope is climbable, snap the velocity so that the character
                // up and down the surface smoothly.
                if climbable {
                    // Points in the normal's direction in the XZ plane.
                    let normal_direction_xz =
                        normal.reject_from_normalized(Vector::Y).normalize_or_zero();

                    // The movement speed along the direction above.
                    let linear_velocity_xz = linear_velocity.dot(normal_direction_xz);

                    let max_y_speed = -linear_velocity_xz * slope_angle.tan();
                    linear_velocity.y = linear_velocity.y.max(max_y_speed);
                } else {
                    // The character is intersecting an unclimbable object, like a wall.
                    // We want the character to slide along the surface, similarly to
                    // a collide-and-slide algorithm.

                    // Don't apply an impulse if the character is moving away from the surface.
                    if linear_velocity.dot(normal) > 0.0 {
                        continue;
                    }

                    // Slide along the surface, rejecting the velocity along the contact normal.
                    let impulse = linear_velocity.reject_from_normalized(normal);
                    linear_velocity.0 = impulse;
                }
            } else {
                // The character is not yet intersecting the other object,
                // but the narrow phase detected a speculative collision.
                //
                // We need to push back the part of the velocity
                // that would cause penetration within the next frame.

                let normal_speed = linear_velocity.dot(normal);

                // Don't apply an impulse if the character is moving away from the surface.
                if normal_speed > 0.0 {
                    continue;
                }

                // Compute the impulse to apply.
                let impulse_magnitude =
                    normal_speed - (deepest_penetration / time.delta_secs_f64().adjust_precision());
                let mut impulse = impulse_magnitude * normal;

                // Apply the impulse differently depending on the slope angle.
                if climbable {
                    // Avoid sliding down slopes.
                    linear_velocity.y -= impulse.y.min(0.0);
                } else {
                    // Avoid climbing up walls.
                    impulse.y = impulse.y.max(0.0);
                    linear_velocity.0 -= impulse;
                }
            }
        }
    }
}

/// Sends [`MovementAction`] events based on keyboard input.
fn keyboard_input(
    mut movement_event_writer: EventWriter<MovementAction>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    let up = keyboard_input.any_pressed([KeyCode::KeyW, KeyCode::ArrowUp]);
    let down = keyboard_input.any_pressed([KeyCode::KeyS, KeyCode::ArrowDown]);
    let left = keyboard_input.any_pressed([KeyCode::KeyA, KeyCode::ArrowLeft]);
    let right = keyboard_input.any_pressed([KeyCode::KeyD, KeyCode::ArrowRight]);

    let horizontal = right as i8 - left as i8;
    let vertical = up as i8 - down as i8;

    let direction = Vector2::new(horizontal as Scalar, vertical as Scalar).clamp_length_max(1.0);

    if direction != Vector2::ZERO {
        movement_event_writer.write(MovementAction::Move(direction));
    }

    if keyboard_input.just_pressed(KeyCode::Space) {
        movement_event_writer.write(MovementAction::Jump);
    }
}

/// Updates the [`Grounded`] status for character controllers.
fn update_grounded(
    mut commands: Commands,
    mut query: Query<
        (Entity, &ShapeHits, &Rotation, Option<&MaxSlopeAngle>),
        With<CharacterController>,
    >,
) {
    for (entity, hits, rotation, max_slope_angle) in &mut query {
        // The character is grounded if the shape caster has a hit with a normal
        // that isn't too steep

        let is_grounded = hits.iter().any(|hit| {
            if let Some(angle) = max_slope_angle {
                (rotation * -hit.normal2).angle_between(Vector::Y).abs() <= angle.0
            } else {
                true
            }
        });

        if is_grounded {
            commands.entity(entity).insert(Grounded);
        } else {
            commands.entity(entity).remove::<Grounded>();
        }
    }
}

/// Responds to [`MovementAction`] events and moves character controllers accordingly.
fn movement(
    time: Res<Time>,
    mut movement_event_reader: EventReader<MovementAction>,
    mut controllers: Query<(
        &MovementAcceleration,
        &JumpImpulse,
        &mut LinearVelocity,
        Has<Grounded>,
    )>,
) {
    // Precision is adjusted so that the example works with
    // both the `f32` and `f64` features. Otherwise you don't need this.
    let delta_time = time.delta_secs_f64().adjust_precision();

    for event in movement_event_reader.read() {
        for (movement_acceleration, jump_impulse, mut linear_velocity, is_grounded) in
            &mut controllers
        {
            match event {
                MovementAction::Move(direction) => {
                    linear_velocity.x += direction.x * movement_acceleration.0 * delta_time;
                    linear_velocity.z -= direction.y * movement_acceleration.0 * delta_time;
                }
                MovementAction::Jump => {
                    if is_grounded {
                        linear_velocity.y = jump_impulse.0;
                    }
                }
            }
        }
    }
}

/// Applies [`controllerGravity`] to character controllers.
fn apply_gravity(
    time: Res<Time>,
    mut controllers: Query<(&ControllerGravity, &mut LinearVelocity)>,
) {
    // Precision is adjusted so that the example works with
    // both the `f32` and `f64` features. Otherwise you don't need this.
    let delta_time = time.delta_secs_f64().adjust_precision();

    for (gravity, mut linear_velocity) in &mut controllers {
        linear_velocity.0 += gravity.0 * delta_time;
    }
}

/// SLow down movement in the XZ plane.
fn apply_movement_damping(mut query: Query<(&MovementDampingFactor, &mut LinearVelocity)>) {
    for (damping_factor, mut linear_velocity) in &mut query {
        // We could use `LinearDamping`, but we don't want to dampen movement along the Y axis
        linear_velocity.x *= damping_factor.0;
        linear_velocity.z *= damping_factor.0;
    }
}
