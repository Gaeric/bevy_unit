///! example follow solari

use bevy::prelude::*;
use bevy::solari::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SolariPlugins)
        .run();
}
