//! rest-pose cloth mesh, 1:1 with the source

use std::collections::HashMap;

use glam::{Mat4, Vec2, Vec3};

use crate::sim::ClothError;

pub const CLOTH_SIZE: f32 = 2.0;

/// rest-pose cloth mesh; `positions` is also the solver's particle array.
#[derive(Clone, Debug, Default)]
pub struct ClothMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
}

/// source `VertexIndexAt(x, y) = x * (resolution + 1) + y`.
pub fn vertex_index_at(res: u32, row: u32, column: u32) -> u32 {
    row * (res + 1) + column
}

/// generate the rest-pose cloth mesh
pub fn generate_cloth_mesh(res: u32) -> ClothMesh {
    assert!(res >= 1, "cloth resolution must be >= 1");
    let verts_per_side = res + 1;
    let inv = 1.0 / res as f32;
    let vertex_count = (verts_per_side * verts_per_side) as usize;

    let mut mesh = ClothMesh {
        positions: Vec::with_capacity(vertex_count),
        normals: Vec::with_capacity(vertex_count),
        uvs: Vec::with_capacity(vertex_count),
        indices: Vec::with_capacity(6 * (res * res) as usize),
    };

    for row in 0..verts_per_side {
        for column in 0..verts_per_side {
            let u = column as f32 * inv;
            let v = row as f32 * inv;
            // u is centered on the origin: [0, 1] -> [-0.5, 0.5], scaled by CLOTH_SIZE
            // v is negated, not centered: `-v` runs down the world (-Y) as it runs down the
            // texture, so the uv is not flipped vertically, and leaves the v = 0 row on y = 0,
            // so the sheet hangs from the origin.
            mesh.positions
                .push(CLOTH_SIZE * Vec3::new(u - 0.5, -v, 0.0));
            mesh.normals.push(Vec3::Z);
            mesh.uvs.push(Vec2::new(u, v));
        }
    }

    for row in 0..res {
        for column in 0..res {
            let a = vertex_index_at(res, row, column);
            let b = vertex_index_at(res, row + 1, column);
            let c = vertex_index_at(res, row, column + 1);
            let d = vertex_index_at(res, row + 1, column + 1);

            // one grid cell per iteration, split into two triangles.
            //
            //     a --- c
            //     |   / |
            //     | /   |
            //     b --- d
            //
            // the split is along b-c, and both triangles are wound so that (b - a) cross (c - a)
            // points along +Z, matching `normals`. `constraints::build_bend` reads this buffer as
            // [i], [i + 5], [i + 2], [i + 1].
            mesh.indices.extend_from_slice(&[a, b, c, c, b, d]);
        }
    }

    mesh
}

/// bakes a transform into the vertices
pub fn bake_transform(positions: &mut [Vec3], transform: Mat4) {
    for position in positions.iter_mut() {
        *position = transform.transform_point3(*position)
    }
}

pub fn particle_diameter_from_first_edge(positions: &[Vec3], scalar: f32) -> f32 {
    (positions[0] - positions[1]).length() * scalar
}

/// the spatial hash keeps it forever to filter out the particle pairs that
/// start out touching (`length2(orig - orig_j) > diameter2`).
pub fn collect_original_positions(positions: &[Vec3]) -> Vec<Vec3> {
    positions.to_vec()
}

pub fn grid_res(vertex_count: usize) -> Option<u32> {
    if vertex_count < 4 {
        return None;
    }

    let side = (vertex_count as f64).sqrt().round() as usize;
    (side * side == vertex_count).then(|| (side - 1) as u32)
}

/// checks that a mesh can be used as a source-layout cloth grid, and returns its resolution
pub fn validate_cloth_grid(vertices: &[Vec3], indices: &[u32]) -> Result<u32, ClothError> {
    let Some(res) = grid_res(vertices.len()) else {
        return Err(ClothError::NotGrid {
            vertex_count: vertices.len(),
        });
    };

    let expected = generate_cloth_mesh(res);
    if expected.indices.len() != indices.len() {
        return Err(ClothError::IndexCount {
            expected: expected.indices.len(),
            actual: indices.len(),
        });
    }

    if let Some(first_mismatch) = expected
        .indices
        .iter()
        .zip(indices)
        .position(|(e, a)| e != a)
    {
        return Err(ClothError::IndexLayout { first_mismatch });
    }

    let mut seen: HashMap<[u32; 3], usize> = HashMap::with_capacity(vertices.len());
    for (i, position) in vertices.iter().enumerate() {
        let key = [
            position.x.to_bits(),
            position.y.to_bits(),
            position.z.to_bits(),
        ];

        if let Some(&other) = seen.get(&key) {
            return Err(ClothError::DuplicateVertex { a: other, b: i });
        }
        seen.insert(key, i);
    }

    Ok(res)
}
