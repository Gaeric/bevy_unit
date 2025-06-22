use bevy::prelude::*;

// use bevy_shine::ShinePlugin;

use bevy_waltz::WaltzPlugin;

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(WaltzPlugin)
        // .add_plugins(dev::plugin)
        // .add_plugins(ShinePlugin)
        .run()
}
