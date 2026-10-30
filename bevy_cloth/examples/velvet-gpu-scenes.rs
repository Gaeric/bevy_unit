//! Independent GPU entry point for the first comparison batch: Attach and Collision.
//! The CPU `velvet-scenes` example remains separate.

use bevy::prelude::*;
use bevy_cloth::{
    demo::scenes::SceneId,
    gpu::{GpuSceneConfig, SolverScenesPlugin},
};

const FIRST_BATCH: &str = "first batch supports only GPU scenes 1 (attach) and 2 (collision); scenes 3..7 are not available";
const HELP: &str = "GPU scenes 1/2: --scene 1|2|attach|collision --paused --timing --help
Defaults: scene 1 (Attach), running, GPU timing off.
The first batch supports only scenes 1 and 2; scenes 3..7 are not available.
Controls: 1/2 switch scene and pause; Space pause/resume; S step while paused;
R reset and pause; C toggle collider guides; middle drag orbit; mouse wheel zoom.";

fn parse_config(args: impl IntoIterator<Item = String>) -> Result<Option<GpuSceneConfig>, String> {
    let mut config = GpuSceneConfig::default();
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--help" | "-h" => return Ok(None),
            "--paused" => config.start_paused = true,
            "--timing" => config.gpu_timing = true,
            "--scene" => {
                let value = args.next().ok_or_else(|| {
                    format!("--scene requires 1|2|attach|collision; {FIRST_BATCH}")
                })?;
                config.scene = match value.as_str() {
                    "1" | "attach" => SceneId::Attach,
                    "2" | "collision" => SceneId::Collision,
                    _ => return Err(format!("unsupported --scene value: {value}; {FIRST_BATCH}")),
                };
            }
            _ => {
                return Err(format!(
                    "unknown argument: {flag}; {FIRST_BATCH}; use --help"
                ));
            }
        }
    }
    Ok(Some(config))
}

fn main() -> AppExit {
    let config = match parse_config(std::env::args().skip(1)) {
        Ok(Some(config)) => config,
        Ok(None) => {
            println!("{HELP}");
            return AppExit::Success;
        }
        Err(error) => {
            eprintln!("{error}");
            return AppExit::error();
        }
    };
    App::new()
        .insert_resource(config)
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "GPU scenes 1/2 — Attach / Collision".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.015, 0.02, 0.035)))
        .add_plugins(SolverScenesPlugin)
        .run()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<GpuSceneConfig>, String> {
        parse_config(args.iter().map(|arg| (*arg).to_owned()))
    }

    #[test]
    fn cli_defaults() {
        let config = parse(&[]).unwrap().unwrap();
        assert_eq!(config.scene, SceneId::Attach);
        assert!(!config.start_paused);
        assert!(!config.gpu_timing);
    }

    #[test]
    fn cli_scene_options_and_flags() {
        for (value, expected) in [
            ("1", SceneId::Attach),
            ("attach", SceneId::Attach),
            ("2", SceneId::Collision),
            ("collision", SceneId::Collision),
        ] {
            let config = parse(&["--timing", "--scene", value, "--paused"])
                .unwrap()
                .unwrap();
            assert_eq!(config.scene, expected);
            assert!(config.start_paused);
            assert!(config.gpu_timing);
        }
        let paused = parse(&["--paused"]).unwrap().unwrap();
        assert!(paused.start_paused && !paused.gpu_timing);
        let timing = parse(&["--timing"]).unwrap().unwrap();
        assert!(!timing.start_paused && timing.gpu_timing);
    }

    #[test]
    fn cli_rejects_missing_invalid_and_unknown_arguments() {
        for args in [
            vec!["--scene"],
            vec!["--scene", "0"],
            vec!["--scene", "3"],
            vec!["--scene", "4"],
            vec!["--scene", "5"],
            vec!["--scene", "6"],
            vec!["--scene", "7"],
            vec!["--scene", "8"],
            vec!["--scene", "self-collision"],
            vec!["--scene", "bad"],
            vec!["--scene", "--paused"],
            vec!["--unknown"],
            vec!["--resolution", "16"],
            vec!["--scene=1"],
            vec!["1"],
        ] {
            let error = parse(&args).unwrap_err();
            assert!(error.contains(FIRST_BATCH), "{args:?}: {error}");
        }
    }

    #[test]
    fn cli_help() {
        for flag in ["--help", "-h"] {
            assert!(parse(&[flag]).unwrap().is_none());
        }
    }
}
