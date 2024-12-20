use std::{collections::BTreeMap, f32::consts::PI};

use bevy::{
    prelude::*,
    render::{
        mesh::{PrimitiveTopology, VertexAttributeValues},
        primitives::Aabb,
    },
};

#[derive(Debug, Clone, Deref, DerefMut)]
struct Triangle([Vec3A; 3]);

#[derive(Default, Resource, Deref, DerefMut)]
struct TriangleTable(BTreeMap<Entity, Vec<Triangle>>);

impl Triangle {
    pub fn aabb(&self) -> Aabb {
        let min = self.iter().copied().reduce(Vec3A::min).unwrap();
        let max = self.iter().copied().reduce(Vec3A::max).unwrap();
        Aabb::from_min_max(min.into(), max.into())
    }

    pub fn centroid(&self) -> Vec3A {
        (self[0] + self[1] + self[2]) / 3.0
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<TriangleTable>()
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(5.0, 5.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    // Only directional light is supported
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::AMBIENT_DAYLIGHT,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.5, 0.5, 0.5).with_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -PI / 8.0,
            -PI / 4.0,
            0.0,
        )),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn update_triangle_table(
    meshes: Res<Assets<Mesh>>,
    mut triangle_table: ResMut<TriangleTable>,
    query: Query<
        (Entity, &Mesh3d, &GlobalTransform),
        Or<(Added<Mesh3d>, Changed<GlobalTransform>)>,
    >,
) {
    for (entity, mesh, transform) in query.iter() {
        if let Some(mesh) = meshes.get(mesh) {
            if let Some((vertices, indices)) = mesh
                .attribute(Mesh::ATTRIBUTE_POSITION)
                .and_then(VertexAttributeValues::as_float3)
                .zip(mesh.indices())
            {
                let transform_matrix = transform.compute_matrix();
                let mut triangles: Vec<Triangle> = vec![];

                match mesh.primitive_topology() {
                    PrimitiveTopology::TriangleList => {
                        for mut chunk in &indices.iter().chunks(3) {
                            let (p0, p1, p2) = chunk.next_tuple().unwrap();
                            let vertices = [p0, p1, p2].map(|id| vertices[id]).map(|[x, y, z]| {
                                transform_matrix.transform_point3a(Vec3A::new(x, y, z))
                            });
                            triangles.push(Triangle(vertices));
                        }
                    }
                    PrimitiveTopology::TriangleStrip => {
                        for (i, (p0, p1, p2)) in indices.iter().tuple_windows().enumerate() {
                            let vertices = if i % 2 == 0 {
                                [p0, p1, p2]
                            } else {
                                [p1, p0, p2]
                            }
                            .map(|id| vertices[id])
                            .map(|[x, y, z]| {
                                transform_matrix.transform_point3a(Vec3A::new(x, y, z))
                            });
                            triangles.push(Triangle(vertices));
                        }
                    }
                    _ => (),
                }
                triangle_table.insert(entity, triangles);
            }
        }
    }
}
