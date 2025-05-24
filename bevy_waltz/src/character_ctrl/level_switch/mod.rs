use bevy::{ecs::system::SystemId, prelude::*};

pub struct LevelSwitchPlugin {
    levels: Vec<(String, Box<dyn Send + Sync + Fn(&mut World) -> SystemId>)>,
    default_level: Option<String>,
}

impl LevelSwitchPlugin {
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

impl Plugin for LevelSwitchPlugin {
    // register the level when build plugin
    fn build(&self, app: &mut App) {
        let levels = self
            .levels
            .iter()
            .map(|(name, system_registrar)| SwitchableLevel {
                name: name.clone(),
                level: system_registrar(app.world_mut()),
            })
            .collect::<Vec<_>>();

        let level_index = if let Some(default_level) = self.default_level.as_ref() {
            levels
                .iter()
                .position(|level| level.name() == default_level)
                .unwrap_or_else(|| panic!("level {default_level:?} not found"))
        } else {
            0
        };
    }
}

#[derive(Clone)]
pub struct SwitchableLevel {
    name: String,
    level: SystemId,
}

impl SwitchableLevel {
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Event)]
pub struct SwitchToLevel(pub usize);

#[derive(Resource)]
pub struct SwitchableLevels {
    pub current: usize,
    pub levels: Vec<SwitchableLevel>,
}

impl SwitchableLevels {
    pub fn current(&self) -> &SwitchableLevel {
        &self.levels[self.current]
    }

    pub fn iter(&self) -> impl Iterator<Item = &SwitchableLevel> {
        self.levels.iter()
    }
}

#[derive(Component)]
pub struct LevelObject;

#[derive(Component)]
pub struct PositionPlayer {
    position: Vec3,
    ttl: Timer,
}

// Observer maybe suitable for this function
fn handle_level_switch(
    mut reader: EventReader<SwitchToLevel>,
    mut switchable_levels: ResMut<SwitchableLevels>,
    query: Query<Entity, Or<(With<LevelObject>, With<PositionPlayer>)>>,
    mut commands: Commands,
) {
    let Some(SwitchToLevel(new_level_index)) = reader.read().last() else {
        return;
    };

    switchable_levels.current = *new_level_index;
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
    commands.run_system(switchable_levels.current().level);
}
