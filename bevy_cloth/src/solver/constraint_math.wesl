// Constraint geometry and XPBD update, shared by both sweep schedules.
import super::layout::{Particle, TetherConstraint};

struct StretchEvaluation {
    direction: vec3<f32>,
    error: f32,
    valid: bool,
};

fn evaluate_stretch(a: vec3<f32>, b: vec3<f32>, rest_length: f32) -> StretchEvaluation {
    let separation = a - b;
    let distance = length(separation);
    // Coincident endpoints have no defined constraint direction.
    if distance == 0.0 {
        return StretchEvaluation(vec3<f32>(0.0), 0.0, false);
    }
    return StretchEvaluation(separation / distance, distance - rest_length, true);
}

struct BendEvaluation {
    g0: vec3<f32>,
    g1: vec3<f32>,
    g2: vec3<f32>,
    g3: vec3<f32>,
    error: f32,
    valid: bool,
};

fn evaluate_bend(
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, rest_angle: f32,
) -> BendEvaluation {
    var invalid: BendEvaluation;
    // Same oriented faces as bending.rs: (a,c,d) and (b,d,c).
    let edge = d - c;
    let edge_squared = dot(edge, edge);
    if edge_squared == 0.0 {
        return invalid;
    }
    let edge_length = sqrt(edge_squared);
    let ca = c - a;
    let da = d - a;
    let db = d - b;
    let cb = c - b;
    let area0 = cross(ca, da);
    let area1 = cross(db, cb);
    let area0_squared = dot(area0, area0);
    let area1_squared = dot(area1, area1);

    // Same dimensionless skinny-triangle threshold as CPU rest preparation.
    let scale0_squared = max(edge_squared, max(dot(ca, ca), dot(da, da)));
    let scale1_squared = max(edge_squared, max(dot(db, db), dot(cb, cb)));
    if area0_squared <= 1e-12 * scale0_squared * scale0_squared
        || area1_squared <= 1e-12 * scale1_squared * scale1_squared
    {
        return invalid;
    }

    let normal0 = area0 * inverseSqrt(area0_squared);
    let normal1 = area1 * inverseSqrt(area1_squared);
    let sin_angle = dot(edge / edge_length, cross(normal0, normal1));
    let cos_angle = clamp(dot(normal0, normal1), -1.0, 1.0);
    let angle = atan2(sin_angle, cos_angle);

    // Rest angles are signed; choose the shortest difference across +/-pi.
    let pi = 3.141592653589793;
    let two_pi = 6.283185307179586;
    var error = angle - rest_angle;
    if error > pi {
        error -= two_pi;
    } else if error < -pi {
        error += two_pi;
    }

    let scaled0 = area0 / area0_squared;
    let scaled1 = area1 / area1_squared;
    let g0 = -edge_length * scaled0;
    let g1 = -edge_length * scaled1;
    let g2 = -(dot(a - d, edge) * scaled0 + dot(b - d, edge) * scaled1) / edge_length;
    let g3 = -(dot(c - a, edge) * scaled0 + dot(c - b, edge) * scaled1) / edge_length;
    return BendEvaluation(g0, g1, g2, g3, error, true);
}

fn xpbd_delta_lambda(error: f32, lambda: f32, alpha: f32, denominator: f32) -> f32 {
    return (-error - alpha * lambda) / denominator;
}

struct TetherUpdate {
    lambda: f32,
    correction: vec3<f32>,
};

fn solve_tether(
    particle: Particle, anchor_position: vec3<f32>, constraint: TetherConstraint,
    lambda: f32, dt: f32, degree: f32,
) -> TetherUpdate {
    if particle.inv_mass == 0.0 {
        return TetherUpdate(0.0, vec3<f32>(0.0));
    }
    let evaluation = evaluate_stretch(particle.position, anchor_position, constraint.max_distance);
    if !evaluation.valid {
        // Coincident endpoints satisfy the limit and have no correction direction.
        return TetherUpdate(0.0, vec3<f32>(0.0));
    }
    let alpha = constraint.compliance / (dt * dt);
    let denominator = alpha + degree * particle.inv_mass;
    let trial_delta = xpbd_delta_lambda(evaluation.error, lambda, alpha, denominator);
    // For C = distance - limit <= 0, tension is a nonpositive multiplier.
    // A slack tether may release previous tension, never accumulate compression.
    let next_lambda = min(lambda + trial_delta, 0.0);
    let delta_lambda = next_lambda - lambda;
    let correction = particle.inv_mass * delta_lambda * evaluation.direction;
    return TetherUpdate(next_lambda, correction);
}
