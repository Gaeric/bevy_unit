//! Independent GPU entry point for all seven comparison scenes.
//! The CPU `velvet-scenes` example remains separate.

use bevy::prelude::*;
use bevy_cloth::{
    demo::scenes::SceneId,
    gpu::{GpuSceneConfig, SolverScenesPlugin},
};

const SCENE_VALUES: &str =
    "1..7 or attach|collision|self-collision|friction|multiple|high-resolution|swirl";
const HELP: &str = "GPU scenes 1..7: --scene <number|name> --paused --timing --help
Scenes: 1 attach; 2 collision; 3 self-collision; 4 friction;
        5 multiple; 6 high-resolution; 7 swirl.
Defaults: scene 1 (Attach), running, GPU timing off.
Controls: 1..7 switch scene and pause; Space pause/resume; S step while paused;
R reset and pause; C toggle solid collider bodies (simulation unchanged);
L toggle triangle-edge overlay on the lit cloth; middle drag orbit; mouse wheel zoom.";

fn parse_config(args: impl IntoIterator<Item = String>) -> Result<Option<GpuSceneConfig>, String> {
    let mut config = GpuSceneConfig::default();
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--help" | "-h" => return Ok(None),
            "--paused" => config.start_paused = true,
            "--timing" => config.gpu_timing = true,
            "--scene" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--scene requires {SCENE_VALUES}"))?;
                config.scene = match value.as_str() {
                    "1" | "attach" => SceneId::Attach,
                    "2" | "collision" => SceneId::Collision,
                    "3" | "self-collision" => SceneId::SelfCollision,
                    "4" | "friction" => SceneId::Friction,
                    "5" | "multiple" => SceneId::Multiple,
                    "6" | "high-resolution" => SceneId::HighResolution,
                    "7" | "swirl" => SceneId::Swirl,
                    _ => {
                        return Err(format!(
                            "unsupported --scene value: {value}; expected {SCENE_VALUES}"
                        ));
                    }
                };
            }
            _ => return Err(format!("unknown argument: {flag}; use --help")),
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
                title: "GPU scenes 1..7 — Attach / Collision / Self Collision / Friction / Multiple / High Resolution / Swirl".into(),
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
            ("3", SceneId::SelfCollision),
            ("self-collision", SceneId::SelfCollision),
            ("4", SceneId::Friction),
            ("friction", SceneId::Friction),
            ("5", SceneId::Multiple),
            ("multiple", SceneId::Multiple),
            ("6", SceneId::HighResolution),
            ("high-resolution", SceneId::HighResolution),
            ("7", SceneId::Swirl),
            ("swirl", SceneId::Swirl),
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
    fn cli_rejects_missing_and_invalid_scene_values() {
        let error = parse(&["--scene"]).unwrap_err();
        assert_eq!(error, format!("--scene requires {SCENE_VALUES}"));
        for value in [
            "0",
            "8",
            "bad",
            "self_collision",
            "high_resolution",
            "--paused",
        ] {
            let error = parse(&["--scene", value]).unwrap_err();
            assert_eq!(
                error,
                format!("unsupported --scene value: {value}; expected {SCENE_VALUES}")
            );
        }
    }

    #[test]
    fn cli_rejects_unknown_arguments() {
        for args in [
            vec!["--unknown"],
            vec!["--resolution", "16"],
            vec!["--scene=1"],
            vec!["1"],
        ] {
            let error = parse(&args).unwrap_err();
            assert_eq!(error, format!("unknown argument: {}; use --help", args[0]));
        }
    }

    #[test]
    fn cli_help() {
        for flag in ["--help", "-h"] {
            assert!(parse(&[flag]).unwrap().is_none());
        }
        for scene in [
            "1 attach",
            "2 collision",
            "3 self-collision",
            "4 friction",
            "5 multiple",
            "6 high-resolution",
            "7 swirl",
        ] {
            assert!(HELP.contains(scene), "help is missing {scene}");
        }
        assert!(HELP.contains("Controls: 1..7 switch scene and pause"));
    }
}
