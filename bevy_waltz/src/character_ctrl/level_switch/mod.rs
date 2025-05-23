use bevy::{ecs::system::SystemId, prelude::*};

pub struct LevelSwitchingPlugin {
    levels: Vec<(String, Box<dyn Send + Sync + Fn(&mut World) -> SystemId>)>,
    default_level: Option<String>,
}

impl LevelSwitchingPlugin {
    pub fn new(default_level: Option<impl ToString>) -> Self {
        Self {
            levels: Default::default(),
            default_level: default_level.map(|name| name.to_string()),
        }
    }

    pub fn with<M>(
        mut self,
        name: impl ToString,
        system: impl 'static + Send + Sync + Clone + IntoSystem<(), (), M>,
    ) -> Self {
        self.levels.push((
            name.to_string(),
            Box::new(move |world| world.register_system(system.clone())),
        ));
        self
    }
}
