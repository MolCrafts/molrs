fn main() {
    emit_runtime_rpath();
}

/// Emits the runtime rpath the opt-in dynamic link form needs.
///
/// The zero-argument default links statically, and there this load command is
/// simply unused — inert, not harmful — so it is emitted unconditionally
/// rather than guessed from the command line. Under the dynamic opt-in
/// (`docs/interop.md`) the `libmolrs_ffi` dylib in the shared target dir
/// records a *hashless absolute* `install_name`, so this artifact resolves the
/// shared molrs image by that path alone and needs no rpath entry for it. What
/// does need one is the dynamically linked libstd, which rustc references as
/// `@rpath/libstd-*.dylib`.
///
/// Deliberately duplicated across the four consumer build scripts: sharing it
/// would take a build-dependency crate, and molpack publishes to crates.io, so
/// that means a new published crate. Twenty lines beat a release-train
/// dependency.
fn emit_runtime_rpath() {
    // wasm has no dynamic linking at all, so neither link form applies to it
    // and there is nothing to inject.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        return;
    }

    // `-Wl,-rpath` is an ELF / Mach-O concept (the value may be a comma list,
    // e.g. "unix,wasm"). MSVC has no rpath, and the static default needs none
    // anywhere.
    let family = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    if !family.split(',').any(|f| f == "unix") {
        return;
    }

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let target = std::env::var("TARGET").expect("cargo sets TARGET for every build script");
    let fix = format!("rustc --print target-libdir --target {target}");

    // Every failure below is a contract violation, not a missing optional:
    // without libstd's directory the dynamically linked artifact loads with an
    // unresolved `@rpath/libstd-*`. Panic with the reproducing command — never
    // downgrade this path to `cargo::warning`.
    let out = std::process::Command::new(&rustc)
        .args(["--print", "target-libdir", "--target", &target])
        .output()
        .unwrap_or_else(|err| {
            panic!("could not run `{rustc}` ({err}); expected the libstd dir for `{target}`: {fix}")
        });
    if !out.status.success() {
        panic!(
            "`{rustc} --print target-libdir` failed ({}); expected the libstd dir for `{target}`: {fix}",
            out.status
        );
    }
    let libdir = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if libdir.is_empty() {
        panic!(
            "`{rustc} --print target-libdir` printed nothing; expected the libstd dir for `{target}`: {fix}"
        );
    }
    if !std::path::Path::new(&libdir).is_dir() {
        panic!(
            "rustc reported target-libdir `{libdir}`, which is not a directory; expected the libstd dir for `{target}`: {fix}"
        );
    }

    // The libstd rpath the dynamic opt-in needs. Inert under the static
    // default: nothing resolves through `@rpath` there, so the load command
    // goes unused.
    println!("cargo::rustc-link-arg=-Wl,-rpath,{libdir}");
    println!("cargo::rerun-if-env-changed=RUSTC");
    println!("cargo::rerun-if-env-changed=TARGET");
}
