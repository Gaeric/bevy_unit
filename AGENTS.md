# AGENTS.md

## Commands
- Check/build workspace: `cargo check --workspace` / `cargo build --workspace`
- Lint: `cargo clippy --workspace --all-targets` · Format: `cargo fmt --all` (verify: `cargo fmt --all --check`)
- Run example (root package is `bevy_workflow`, Bevy 0.19): `cargo run --example example-bake`; other packages: `cargo run -p bevy_shine --example minimal`; the `hs2_head` demo is its own crate: `cargo run -p hs2_head`, and so is the `bevy_deck` demo: `cargo run -p bevy_deck` (it is the only crate depending on the external `../libdeck/`, keeping `libdeck`/`libdeck_sample` out of the root package)
- Tests: `cargo test --workspace -j 2`; single test: `cargo test -j 2 -p bevy_workflow <test_name> -- --nocapture` (or `cargo test -j 2 <filter>`). Always pass `-j 2`: full-parallel compilation/linking of the Bevy dependency tree otherwise exhausts memory.
- Container / sandboxed runs: never share `target/` with the host. This repo is bind-mounted into Docker, so `target/` holds the host toolchain's build artifacts; a container `cargo check` still writes check `.rmeta`, rebuilds proc-macro `.rlib`s, runs `-sys` build scripts and rewrites `.rustc_info.json`, which forces the host back into a full rebuild. Always set a private target dir, e.g. `CARGO_TARGET_DIR=/tmp/cargo-target cargo check -p hs2_head`; if the container lacks X11/Wayland/ALSA `.pc` files, add stub ones via `PKG_CONFIG_PATH=/tmp/pkgstub` instead of touching the repo.
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

## bevy_cloth collaboration
- Scope: high-performance GPU **cloth simulation** using Bevy compute shaders. XPBD is the current main method, not a mandate to build a general-purpose solver framework. The owner's ability to explain, verify, and modify the implementation is part of completion.
- Start with `bevy_cloth/readme.org` and `bevy_cloth/plan.org`. The plan holds the current next step; old M0–M6 labels in code are historical, not an active roadmap. CPU code is a small-scale reference and diagnostic tool, not a seven-scene prerequisite for GPU work.
- Collaborate through development and debugging: the agent advances a small agreed batch of framework code; the owner runs and validates it; both investigate observed problems before choosing the next batch. Build familiarity through participation in diagnosis, not prerequisite lessons or quizzes. The owner wrote the CPU solver; do not assume its basic calculations need reteaching.
- Each batch must state changed files, expected behavior, how to run and validate, and known limitations. When diagnosing a problem, identify the specific file/function, data to inspect, expected values or relationships, and the next check implied by each result. Provide theory or implementation references as needed for the issue, not as a mandatory course.
- Solver core code remains owner-written. Agents may explain, review, and provide agreed verification scaffolding or peripheral integration. Ask before changing algorithm-related data layouts, scheduling, or anything with an unclear boundary; do not classify core work as mere wiring.
- Velvet is a reference, not an immutable specification. Before changing a special behavior, explain its purpose and problem, agree on the change, and validate the improvement. Do not modify the sibling reference repository as part of cloth work without explicit authorization.
- Distinguish code facts, hand-calculated expectations, measured results, decisions, and unverified risks. CPU/GPU agreement alone is not proof of correctness. Never mark understanding, tests, visual behavior, or performance as verified without the corresponding evidence.
- Keep current docs short and actionable. Record settled explanations and decisions after discussion; leave future choices uncommitted. Do not expand scenes, features, or refactors beyond the agreed question. All file changes still follow the reviewable edit workflow above.
