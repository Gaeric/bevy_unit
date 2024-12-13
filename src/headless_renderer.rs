// fork from bevy official example

use bevy::{
    app::{App, Plugin},
    prelude::*,
};

use crossbeam_channel::{Receiver, Sender};

struct HeadlessRendererPlugin;

#[derive(Resource, Deref, Debug)]
struct MainWorldReceiver(Receiver<Vec<u8>>);

#[derive(Resource, Deref, Debug)]
struct RenderWorldSender(Sender<Vec<u8>>);

struct AppConfig {
    width: u32,
    height: u32,
    single_image: bool,
}

/// Capture image state
#[derive(Debug, Default)]
enum SceneState {
    #[default]
    // State before any rendering
    BuildScene,
    // Rendering state, stores the number of frames remaining before saving the image
    Render(u32),
}

/// Capture image settings and state
#[derive(Debug, Default, Resource)]
struct SceneController {
    state: SceneState,
    name: String,
    width: u32,
    height: u32,
    single_image: bool,
}

impl SceneController {
    pub fn new(width: u32, height: u32, single_image: bool) -> SceneController {
        SceneController {
            state: SceneState::BuildScene,
            name: String::from(""),
            width,
            height,
            single_image,
        }
    }
}

impl Plugin for HeadlessRendererPlugin {
    fn build(&self, app: &mut App) {
        let config = AppConfig {
            width: 1920,
            height: 1080,
            single_image: true,
        };

        // setup frame capture
        app.insert_resource(SceneController::new(
            config.width,
            config.height,
            config.single_image,
        ));
    }
}
