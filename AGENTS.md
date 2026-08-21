# AGENTS.md

## Commands
- Check/build workspace: `cargo check --workspace` / `cargo build --workspace`
- Lint: `cargo clippy --workspace --all-targets` · Format: `cargo fmt --all` (verify: `cargo fmt --all --check`)
- Run example (root package is `bevy_workflow`, Bevy 0.19): `cargo run --example example-bake`, `cargo run --example hs2_head`; other packages: `cargo run -p bevy_shine --example minimal`
- Tests: `cargo test --workspace`; single test: `cargo test -p bevy_workflow <test_name> -- --nocapture` (or `cargo test <filter>`)
- Bevy is patched to a local engine at `../bevy_engines/bevy_0.19/` via `[patch.crates-io]` in the root `Cargo.toml`; that checkout must exist before any build.
- Debug perf profiles are preconfigured in root `Cargo.toml` (opt-level 1 for our code, 3 for deps) — don't add per-crate overrides.

## Style
- Rust edition 2024, Bevy 0.19 idioms: `prelude::*`, `Commands`, observers (`add_observer`, `On<T>`), `Single<...>` for unique entities, `Extract`/`ExtractResource` for render-world data.
- Imports: group external crates first, then `std`, then local modules; use nested `bevy::{...}` groups. Keep `#[derive(Resource/Component/...)]` doc comments concise (`//!` for module docs).
- Naming: `PascalCase` types, `snake_case` fns/fields, `SCREAMING_SNAKE_CASE` consts (e.g. `EYELASH_BAKE_SHADER_PATH`).
- Error handling: prefer `Option`, early `return`/`continue`, and `let ... else`; use `info!`/`warn!` logging; `.unwrap()` only in one-shot setup/startup.
- Keep extract systems cheap: prefer single `ExtractResource` collections (change-detected) over per-frame component re-extraction for static config.
