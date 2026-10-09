//! Temporary workaround for the `dlss` + `dynamic_linking` link *and* startup failure
//! on Linux (see `docs/tips.org` for the full write-up).
//!
//! `dlss_wgpu`'s build script runs `bindgen` with `wrap_static_fns`, which emits a
//! wrapper for every function in the NGX headers — including the `NVSDK_NGX_D3D11_*`
//! and `NVSDK_NGX_D3D12_*` entry points. Those symbols are not present in the Linux
//! `libnvsdk_ngx.a` (the Linux SDK only ships the Vulkan and CUDA backends), so
//! `wrap_static_fns.o` keeps them as undefined references.
//!
//! In a fully static link the unused wrappers are dropped by `--gc-sections`. But this
//! crate's `dev` feature enables `bevy/dynamic_linking`, so bevy is linked into
//! `libbevy_dylib.so` and those references survive in the shared object. `lld` then
//! defaults to `--no-allow-shlib-undefined` when linking an executable:
//!
//! ```text
//! ld.lld: error: undefined reference: NVSDK_NGX_D3D11_CreateFeature
//!     >>> referenced by libbevy_dylib.so (disallowed by --no-allow-shlib-undefined)
//! ```
//!
//! `--allow-shlib-undefined` alone is not enough: the four references are
//! `R_X86_64_JUMP_SLOT` entries, and `libbevy_dylib.so` is built with `-z now`
//! (`DF_BIND_NOW`), so the dynamic loader resolves them eagerly at startup:
//!
//! ```text
//! target/debug/hs2_head: symbol lookup error: libbevy_dylib.so:
//!     undefined symbol: NVSDK_NGX_D3D11_CreateFeature
//! ```
//!
//! Defining the symbols in the executable itself makes them part of its dynamic
//! symbol table, so the loader can bind the shared library's PLT entries against
//! them. The functions are never called on Linux, so the value is irrelevant.
//!
//! Once `dlss_wgpu` blocklists the D3D symbols on non-Windows targets, both this file
//! and the `docs/tips.org` entry can go.

/// Never called on Linux (wgpu uses the Vulkan backend); bound to address 0 just so
/// `libbevy_dylib.so` has something to resolve its PLT entries against.
const STUBBED_NGX_SYMBOLS: [&str; 4] = [
    "NVSDK_NGX_D3D11_CreateFeature",
    "NVSDK_NGX_D3D11_EvaluateFeature_C",
    "NVSDK_NGX_D3D12_CreateFeature",
    "NVSDK_NGX_D3D12_EvaluateFeature_C",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Linux only: the msvc linker and macOS `ld` do not understand these arguments.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    // `rustc-link-arg` only applies to this package's bin/example/test/bench targets,
    // so it does not invalidate the whole bevy dependency graph like global `RUSTFLAGS`.
    println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");

    for symbol in STUBBED_NGX_SYMBOLS {
        println!("cargo:rustc-link-arg=-Wl,--defsym={symbol}=0");
    }
}
