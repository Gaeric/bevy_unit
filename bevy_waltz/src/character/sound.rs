use std::time::Duration;

use bevy::prelude::*;
use bevy_tnua::{
    builtins::TnuaBuiltinJumpState,
    prelude::{TnuaBuiltinJump, TnuaController},
};

use crate::character::{WaltzPlayer, assets::CharacterAssets};

pub(super) fn plugin(app: &mut App) {
    app.add_systems(Update, (character_jump, character_movement, character_land));
}

fn character_jump(
    mut commands: Commands,
    character: Single<&TnuaController, With<WaltzPlayer>>,
    character_assets: ResMut<CharacterAssets>,
    mut is_jumping: Local<bool>,
    mut sound_cooldown: Local<Option<Timer>>,
    time: Res<Time>,
) {
    let sound_cooldown = sound_cooldown
        .get_or_insert_with(|| Timer::new(Duration::from_millis(1000), TimerMode::Once));
    sound_cooldown.tick(time.delta());

    if character
        .concrete_action::<TnuaBuiltinJump>()
        .is_none_or(|x| matches!(x, (_, TnuaBuiltinJumpState::FallSection)))
    {
        *is_jumping = false;
        return;
    }
    if *is_jumping {
        return;
    }

    *is_jumping = true;

    if sound_cooldown.finished() {
        // todo: play sound
        commands.spawn((
            AudioPlayer(character_assets.jump_sound.clone()),
            PlaybackSettings::ONCE,
        ));

        sound_cooldown.reset();
        info!("play jump sound");
    }
}

fn character_movement() {}

fn character_land() {}
