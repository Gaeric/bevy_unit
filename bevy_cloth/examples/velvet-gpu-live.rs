//! Real-time GPU cloth, distinct from the one-shot headless `velvet-gpu` comparison.
//! Space pauses/resumes; S steps; R resets paused; C toggles collider guides.
//! Startup-only CLI workload controls; no CPU mesh writeback.

use bevy::prelude::*;
use bevy_cloth::gpu::{LiveConfig, SolverLivePlugin};

fn parse_config(args: impl IntoIterator<Item = String>) -> Result<Option<LiveConfig>, String> {
    let mut config = LiveConfig::default();
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--help" | "-h" => return Ok(None),
            "--paused" => config.start_paused = true,
            "--timing" => config.gpu_timing = true,
            "--no-self-collision" => config.self_collision = false,
            "--resolution" | "--layers" | "--substeps" | "--iterations" | "--neighbors"
            | "--hash-interval" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("{flag} requires an integer"))?;
                let value: u32 = value
                    .parse()
                    .map_err(|_| format!("invalid integer for {flag}: {value}"))?;
                match flag.as_str() {
                    "--resolution" => config.resolution = value,
                    "--layers" => config.layers = value,
                    "--substeps" => config.substeps = value,
                    "--iterations" => config.iterations = value,
                    "--neighbors" => config.neighbor_capacity = value,
                    "--hash-interval" => config.hash_interval = value,
                    _ => unreachable!(),
                }
            }
            _ => return Err(format!("unknown argument: {flag}; use --help")),
        }
    }
    config.validate().map_err(str::to_owned)?;
    Ok(Some(config))
}

fn main() -> AppExit {
    let config = match parse_config(std::env::args().skip(1)) {
        Ok(Some(config)) => config,
        Ok(None) => {
            println!(
                "GPU cloth: --resolution 2..128 --layers 1..8 --substeps 1..16 --iterations 1..64\n  --neighbors 1..256 --hash-interval 1..64 --no-self-collision --paused --timing\nDefaults: 16 / 2 / 2 / 4 / 64 / 3, self-collision on, running, timing off.\nResolution changes particle diameter/mass distribution; this is not a physically equivalent refinement.\nSpace pause/resume; S step; R reset paused; C collider guides."
            );
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
                title: "GPU cloth — Space pause / S step / R reset / C guides".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.015, 0.02, 0.035)))
        .add_plugins(SolverLivePlugin)
        .run()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<LiveConfig>, String> {
        parse_config(args.iter().map(|arg| (*arg).to_owned()))
    }

    #[test]
    fn cli_defaults_and_explicit_workload() {
        let defaults = parse(&[]).unwrap().unwrap();
        assert_eq!(defaults.particle_count(), 578);
        assert!(!defaults.start_paused && !defaults.gpu_timing);
        let config = parse(&[
            "--resolution",
            "32",
            "--layers",
            "3",
            "--substeps",
            "4",
            "--iterations",
            "6",
            "--neighbors",
            "96",
            "--hash-interval",
            "2",
            "--paused",
            "--timing",
            "--no-self-collision",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(config.particle_count(), 3267);
        assert_eq!((config.substeps, config.iterations), (4, 6));
        assert_eq!((config.neighbor_capacity, config.hash_interval), (96, 2));
        assert!(config.start_paused && config.gpu_timing && !config.self_collision);
    }

    #[test]
    fn cli_rejects_missing_invalid_and_unknown_arguments() {
        for args in [
            vec!["--layers"],
            vec!["--layers", "bad"],
            vec!["--layers", "0"],
            vec!["--resolution", "4294967296"],
            vec!["--unknown"],
        ] {
            assert!(parse(&args).is_err());
        }
        assert!(parse(&["--help"]).unwrap().is_none());
    }
}
