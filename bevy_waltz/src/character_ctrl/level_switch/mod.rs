use bevy::{ecs::system::SystemId, prelude::*};

pub struct LevelSwitchingPlugin {
    levels: Vec<(String, Box<dyn Send + Sync + Fn(&mut World) -> SystemId>)>,
    default_devel: Option<String>
}
