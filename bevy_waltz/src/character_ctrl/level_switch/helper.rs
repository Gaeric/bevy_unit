use avian3d::{
    math::Vector3,
    prelude::{Collider, RigidBody},
};
use bevy::{ecs::system::SystemParam, prelude::*};

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
}
