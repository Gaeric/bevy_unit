use bevy::{
    ecs::{component::Component, entity::Entity},
    scene::InstanceId,
};

pub trait ModularCharacter: Component {
    fn id_mut(&mut self) -> &mut usize;
    fn instance_id_mut(&mut self) -> &mut Option<InstanceId>;
    fn entities_mut(&mut self) -> &mut Vec<Entity>;
    fn id(&self) -> &usize;
    fn instance_id(&self) -> Option<&InstanceId>;
    fn entities(&self) -> &Vec<Entity>;
}
