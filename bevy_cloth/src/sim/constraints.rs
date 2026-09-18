//! constraint tables for the solver, built from a baked rest-pose mesh.

use glam::Vec3;

use crate::sim::{
    ClothError,
    mesh_gen::{validate_cloth_grid, vertex_index_at},
};

#[derive(Clone, Debug, Default)]
pub struct StretchConstraints {
    /// particle pairs, in the source's traversal order.
    pub indices: Vec<[u32; 2]>,
    /// rest length of `indices[i]`, in world units.
    pub lengths: Vec<f32>,
}

/// bending constraints; the source never computes the rest angle
#[derive(Clone, Debug, Default)]
pub struct BendConstraints {
    pub indices: Vec<[u32; 4]>,
    pub angles: Vec<f32>,
}

/// attachment and long range attachment: every particle is constrained to every slot.
#[derive(Clone, Debug, Default)]
pub struct AttachConstraints {
    pub particle_ids: Vec<u32>,
    pub slot_ids: Vec<u32>,
    pub distances: Vec<f32>,
    /// world position of each slot, in `attached` order.
    pub slot_positions: Vec<Vec3>,
}

/// everything the solver needs besides the particle arrays.
#[derive(Clone, Debug, Default)]
pub struct ClothConstraints {
    pub stretch: StretchConstraints,
    pub bend: BendConstraints,
    pub attach: AttachConstraints,
    /// inverse mass: 1.0, except particles pinned by a slot (0.0).
    pub inv_masses: Vec<f32>,
}

/// builds the constraint tables from a *baked* (world space) cloth mesh
/// `attached` lists the particles that carry an attachment slot. the grid layout contract is
/// validated first, so this only accepts meshes `mesh_gen::generate_cloth_mesh` could produce.
pub fn build_constraints(
    vertices: &[Vec3],
    indices: &[u32],
    attached: &[u32],
) -> Result<ClothConstraints, ClothError> {
    let res = validate_cloth_grid(vertices, indices)?;

    let mut inv_masses = vec![1.0; vertices.len()];

    let stretch = build_stretch(vertices, res);
    let attach = build_attach(vertices, attached, &mut inv_masses)?;
    let bend = build_bend(indices);

    Ok(ClothConstraints {
        stretch,
        bend,
        attach,
        inv_masses,
    })
}

/// source `GenerateStretch`: for every particle one constraint to its `+y` neighbour, one to its
/// `+x` neighbour, and both quad diagonals, `4 * res^2 + 2 * res` constraints in total.
fn build_stretch(vertices: &[Vec3], res: u32) -> StretchConstraints {
    let side = res + 1;
    let mut stretch = StretchConstraints::default();

    for row in 0..side {
        for column in 0..side {
            let idx1 = vertex_index_at(res, row, column);
            if column != res {
                push_stretch(
                    vertices,
                    &mut stretch,
                    idx1,
                    vertex_index_at(res, row, column + 1),
                );
            }
            if row != res {
                push_stretch(
                    vertices,
                    &mut stretch,
                    idx1,
                    vertex_index_at(res, row + 1, column),
                );
            }

            if row != res && column != res {
                push_stretch(
                    vertices,
                    &mut stretch,
                    idx1,
                    vertex_index_at(res, row + 1, column + 1),
                );

                push_stretch(
                    vertices,
                    &mut stretch,
                    vertex_index_at(res, row, column + 1),
                    vertex_index_at(res, row + 1, column),
                );
            }
        }
    }

    stretch
}

fn push_stretch(vertices: &[Vec3], sink: &mut StretchConstraints, i1: u32, i2: u32) {
    sink.indices.push([i1, i2]);
    sink.lengths
        .push((vertices[i1 as usize] - vertices[i2 as usize]).length());
}

/// source `GenerateAttach`: one slot per attached particle, held at that particle's *baked*
/// position, plus a long range constraint from every particle to every slot. a zero distance pins
/// the particle, which is how `AddAttach` zeroes its inverse mass.
fn build_attach(
    vertices: &[Vec3],
    attached: &[u32],
    inv_masses: &mut [f32],
) -> Result<AttachConstraints, ClothError> {
    let mut attach = AttachConstraints::default();

    for (idx, &particle) in attached.iter().enumerate() {
        let Some(&slot_position) = vertices.get(particle as usize) else {
            return Err(ClothError::AttachmentOutOfRange { particle });
        };
        attach.slot_positions.push(slot_position);

        for (i, &position) in vertices.iter().enumerate() {
            let distance = (slot_position - position).length();
            if distance == 0.0 {
                inv_masses[i] = 0.0;
            }

            attach.particle_ids.push(i as u32);
            attach.slot_ids.push(idx as u32);
            attach.distances.push(distance);
        }
    }

    Ok(attach)
}

fn build_bend(indices: &[u32]) -> BendConstraints {
    let mut bend = BendConstraints::default();

    // `validate_cloth_grid` guarantees the index count is a multiple of 6
    for quad in indices.as_chunks::<6>().0 {
        bend.indices.push([quad[0], quad[5], quad[2], quad[1]]);
        bend.angles.push(0.0);
    }

    bend
}
