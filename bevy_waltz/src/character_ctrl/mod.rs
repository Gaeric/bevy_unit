/// character controller system
/// forked from the tnua shooter_like demo
use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_tnua::{control_helpers::TnuaCrouchEnforcerPlugin, prelude::TnuaControllerPlugin};
use bevy_tnua_avian3d::*;

mod ctrl_systems;

use ctrl_systems::info_system::*;

pub(super) fn plugin(app: &mut App) {
    app.add_plugins(PhysicsPlugins::new(FixedPostUpdate));
    app.add_plugins(TnuaAvian3dPlugin::new(FixedUpdate));
    app.add_plugins(TnuaControllerPlugin::new(FixedUpdate));
    app.add_plugins(TnuaCrouchEnforcerPlugin::new(FixedUpdate));

    app.add_systems(Update, character_control_radar_visualization_system);
}
