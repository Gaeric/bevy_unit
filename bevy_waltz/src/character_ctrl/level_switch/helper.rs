use avian3d::prelude::{Collider, RigidBody};
use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_tnua::math::{AsF32, Float, Quaternion, Vector3};

use super::LevelObject;

#[derive(SystemParam, Deref, DerefMut)]
pub struct LevelSetupHelper<'w, 's> {
    #[deref]
    pub commands: Commands<'w, 's>,
    pub meshes: ResMut<'w, Assets<Mesh>>,
    pub materials: ResMut<'w, Assets<StandardMaterial>>,
    asset_server: Res<'w, AssetServer>,
}

pub struct LevelSetupHelperWithMaterial<'a, 'w, 's> {
    parent: &'a mut LevelSetupHelper<'w, 's>,
    material: Handle<StandardMaterial>,
}

impl<'w, 's> LevelSetupHelper<'w, 's> {
    pub fn spawn_named(&mut self, name: impl ToString) -> EntityCommands {
        self.commands
            .spawn((LevelObject, Name::new(name.to_string())))
    }

    pub fn spawn_floor(&mut self, color: impl Into<Color>) -> EntityCommands {
        let mesh = self
            .meshes
            .add(Plane3d::default().mesh().size(128.0, 128.0));
        let material = self.materials.add(color.into());
        let mut command = self.spawn_named("Floor");
        command.insert((Mesh3d(mesh), MeshMaterial3d(material)));

        command.insert(RigidBody::Static);
        command.insert(Collider::half_space(Vector3::Y));
        command
    }

    pub fn with_material<'a>(
        &'a mut self,
        material: impl Into<StandardMaterial>,
    ) -> LevelSetupHelperWithMaterial<'a, 'w, 's> {
        let material = self.materials.add(material);
        LevelSetupHelperWithMaterial {
            parent: self,
            material,
        }
    }

    pub fn with_color<'a>(
        &'a mut self,
        color: impl Into<Color>,
    ) -> LevelSetupHelperWithMaterial<'a, 'w, 's> {
        self.with_material(color.into())
    }

    pub fn spawn_scene_cuboid(
        &mut self,
        name: impl ToString,
        path: impl ToString,
        transform: Transform,
        size: Vector3,
    ) -> EntityCommands {
        let scene = self.asset_server.load(path.to_string());
        let mut cmd = self.spawn_named(name);

        cmd.insert((SceneRoot(scene), transform));

        cmd.insert(RigidBody::Static);
        cmd.insert(Collider::cuboid(size.x, size.y, size.z));

        cmd
    }
}

impl LevelSetupHelperWithMaterial<'_, '_, '_> {
    pub fn spawn_mesh_without_physics(
        &mut self,
        name: impl ToString,
        transform: Transform,
        mesh: impl Into<Mesh>,
    ) -> EntityCommands {
        let mesh = self.parent.meshes.add(mesh);
        let mut cmd = self.parent.spawn_named(name);
        cmd.insert((
            Mesh3d(mesh),
            MeshMaterial3d(self.material.clone()),
            transform,
        ));
        cmd
    }

    pub fn spawn_cuboid(
        &mut self,
        name: impl ToString,
        transform: Transform,
        size: Vector3,
    ) -> EntityCommands {
        let mut cmd =
            self.spawn_mesh_without_physics(name, transform, Cuboid::from_size(size.f32()));

        cmd.insert((RigidBody::Static, Collider::cuboid(size.x, size.y, size.z)));

        cmd
    }

    pub fn spawn_compound_cuboids(
        &mut self,
        name: impl ToString,
        transform: Transform,
        parts: &[(Vector3, Quaternion, Vector3)],
    ) -> EntityCommands {
        let child_entity_ids = parts
            .iter()
            .map(|&(pos, rot, size)| {
                self.parent
                    .commands
                    .spawn((
                        Transform {
                            translation: pos.f32(),
                            rotation: rot.f32(),
                            scale: Vec3::ONE,
                        },
                        Mesh3d(self.parent.meshes.add(Cuboid::from_size(size.f32()))),
                        MeshMaterial3d(self.material.clone()),
                    ))
                    .id()
            })
            .collect::<Vec<_>>();

        let mut cmd = self.parent.spawn_named(name);
        cmd.insert(transform);
        cmd.add_children(&child_entity_ids);

        cmd.insert((
            RigidBody::Static,
            Collider::compound(
                parts
                    .iter()
                    .map(|&(pos, rot, size)| (pos, rot, Collider::cuboid(size.x, size.y, size.z)))
                    .collect(),
            ),
        ));

        cmd
    }

    pub fn spawn_cylinder(
        &mut self,
        name: impl ToString,
        transform: Transform,
        radius: Float,
        half_height: Float,
    ) -> EntityCommands {
        let mut cmd = self.spawn_mesh_without_physics(
            name,
            transform,
            Cylinder {
                radius: radius.f32(),
                half_height: half_height.f32(),
            },
        );

        cmd.insert((
            RigidBody::Static,
            Collider::cylinder(radius, 2.0 * half_height),
        ));

        cmd
    }

    pub fn spawn_dynamic_ball(
        &mut self,
        name: impl ToString,
        transform: Transform,
        radius: Float,
    ) -> EntityCommands {
        let mut cmd = self.spawn_mesh_without_physics(
            name,
            transform,
            Sphere {
                radius: radius.f32(),
            },
        );

        cmd.insert((RigidBody::Dynamic, Collider::sphere(radius)));

        cmd
    }
}
