use std::time::Duration;

use avian3d::prelude::{AngularVelocity, LinearVelocity};
use bevy::{
    ecs::{query::QueryData, system::SystemId},
    prelude::*,
};

use crate::WaltzPlayer;

pub struct LevelSwitchPlugin {
    levels: Vec<(String, Box<dyn Send + Sync + Fn(&mut World) -> SystemId>)>,
    default_level: Option<String>,
}

mod helper;
pub mod jungle_gym;

#[derive(Component)]
pub struct Climable;

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

        app.insert_resource(SwitchableLevels { current: 0, levels });
        app.add_message::<SwitchToLevel>();
        app.add_systems(Update, (handle_level_switch, handle_player_position));
        app.add_systems(Startup, move |mut writer: MessageWriter<SwitchToLevel>| {
            writer.write(SwitchToLevel(level_index));
        });
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

#[derive(Message)]
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

impl From<Vec3> for PositionPlayer {
    fn from(position: Vec3) -> Self {
        Self {
            position,
            ttl: Timer::new(Duration::from_millis(500), TimerMode::Once),
        }
    }
}

// Observer maybe suitable for this function
fn handle_level_switch(
    mut reader: MessageReader<SwitchToLevel>,
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

#[derive(QueryData)]
#[query_data(mutable)]
struct PlayerQueryForPosition {
    transform: &'static mut Transform,
    avian3d_linear_velocity: Option<&'static mut LinearVelocity>,
    avian3d_angular_velocity: Option<&'static mut AngularVelocity>,
}

fn handle_player_position(
    time: Res<Time>,
    mut player_query: Query<PlayerQueryForPosition, With<WaltzPlayer>>,
    mut position_query: Query<(Entity, &mut PositionPlayer)>,
    mut commands: Commands,
) {
    let Some((position_entity, mut position_player)) = position_query.iter_mut().next() else {
        return;
    };

    for mut player in player_query.iter_mut() {
        player.transform.translation = position_player.position;

        if let Some(velocity) = player.avian3d_linear_velocity.as_mut() {
            velocity.0 = Default::default()
        }

        if let Some(velocity) = player.avian3d_angular_velocity.as_mut() {
            velocity.0 = Default::default()
        }
    }

    if position_player.ttl.tick(time.delta()).is_finished() {
        commands.entity(position_entity).despawn();
    }
}
