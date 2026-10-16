# AGENTS.md

## Commands
- Check/build workspace: `cargo check --workspace` / `cargo build --workspace`
- Lint: `cargo clippy --workspace --all-targets` · Format: `cargo fmt --all` (verify: `cargo fmt --all --check`)
- Run example (root package is `bevy_workflow`, Bevy 0.19): `cargo run --example example-bake`; other packages: `cargo run -p bevy_shine --example minimal`; the `hs2_head` demo is its own crate: `cargo run -p hs2_head`, and so is the `bevy_deck` demo: `cargo run -p bevy_deck` (it is the only crate depending on the external `../libdeck/`, keeping `libdeck`/`libdeck_sample` out of the root package)
- Tests: `cargo test --workspace -j 2`; single test: `cargo test -j 2 -p bevy_workflow <test_name> -- --nocapture` (or `cargo test -j 2 <filter>`). Always pass `-j 2`: full-parallel compilation/linking of the Bevy dependency tree otherwise exhausts memory.
- Bevy is patched to a local engine at `../bevy_engines/bevy_0.20/` via `[patch.crates-io]` in the root `Cargo.toml`; that checkout must exist before any build.
- Debug perf profiles are preconfigured in root `Cargo.toml` (opt-level 1 for our code, 3 for deps) — don't add per-crate overrides.

## Style
- Rust edition 2024, Bevy 0.19 idioms: `prelude::*`, `Commands`, observers (`add_observer`, `On<T>`), `Single<...>` for unique entities, `Extract`/`ExtractResource` for render-world data.
- Imports: group external crates first, then `std`, then local modules; use nested `bevy::{...}` groups. Keep `#[derive(Resource/Component/...)]` doc comments concise (`//!` for module docs).
- Naming: `PascalCase` types, `snake_case` fns/fields, `SCREAMING_SNAKE_CASE` consts (e.g. `EYELASH_BAKE_SHADER_PATH`).
- Error handling: prefer `Option`, early `return`/`continue`, and `let ... else`; use `info!`/`warn!` logging; `.unwrap()` only in one-shot setup/startup.
- Keep extract systems cheap: prefer single `ExtractResource` collections (change-detected) over per-frame component re-extraction for static config.

## Workflow
- Every file change must be reviewable before it applies: use `edit` for targeted changes (it shows the exact old/new text), never `write` to overwrite a whole file or shell redirection / `sed -i` that hides the diff. The user confirms each change and checks the applied form matches their intent.
- One `edit` per logical change. State which file, which region, and why in the message; for multi-file work, list the plan first and apply the edits one at a time so each can be checked.
- Keep each proposal small enough to verify by reading the diff; don't batch unrelated rewrites into a single change.
