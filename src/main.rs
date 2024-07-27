use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins.set(AssetPlugin::default()),))
        .run();
}
