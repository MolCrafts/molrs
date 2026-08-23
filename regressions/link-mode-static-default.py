"""Default form: a molrs extension built with NO flags must be STATIC.

Static is the zero-argument default. A plain `maturin build --release` injects
no rustflags at all — `.cargo/config.toml` carries only `[build] target-dir` —
so the extension links molrs in statically and the wheel is self-contained,
with no absolute path into anybody's build tree. This script asserts exactly
that: **zero** `libmolrs_ffi` dynamic-link entries.

`scripts/verify-shared-dylib.sh` and `regressions/ffi-shared-dylib.py` assert
the other form: the opt-in dynamic link, under which every native consumer
shares ONE `libmolrs_ffi` and a Frame crosses from molrs into molpack zero-copy
inside a single process. That form is flags typed on the command line, never a
committed switch — `docs/interop.md` is the one place to copy them from, and
`verify-shared-dylib.sh` carries them itself.

**These two gates assert opposite things and must never be interchanged.**
`ffi-shared-dylib.py` demands the libmolrs_ffi entry be PRESENT (and shared);
this one demands it be ABSENT. Both are correct — one for the default build,
one for the opt-in (`docs/interop.md`). If a change makes one of them green by
making the other red, that is the switch moving, not a test to refresh. Neither
golden is ever edited to make a red run green.

A published wheel that kept a dynamic entry would name
`<somebody>/target/release/deps/libmolrs_ffi.dylib` — a path that does not exist
on the installing machine — so this is the assertion standing between a release
build regime regression and an ImportError in every user's interpreter.

Failure policy, per `scripts/verify-shared-dylib.sh` lines 35-37: refuse and
explain, never silently skip. A green run that proved nothing is exactly the
"test that cannot fail" this repo forbids — so an unreadable extension, an
absent linker tool or an unrecognised platform all exit 1 with a numbered
remediation list, never 0.

Public API only. Zero third-party imports beyond numpy, which is a molrs
runtime dependency (the arrays are the documented way to fill a `Block`), and
the platform linker inspector (`otool` / `ldd`). No pytest. No network.

Run standalone:  python regressions/link-mode-static-default.py
"""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys

import numpy as np

import molrs

# ── Hard-coded goldens ───────────────────────────────────────────────────────
# The link-mode golden. ZERO, not "few", not "at most one": a static extension
# names libmolrs_ffi nowhere in its load commands. Changing this number to 1
# converts this file into a duplicate of ffi-shared-dylib.py and deletes the
# only check that a release wheel is self-contained.
FFI_LIB_STEM: str = "libmolrs_ffi"
GOLDEN_FFI_ENTRY_COUNT: int = 0

# Public-API goldens. Same three-atom frame ffi-shared-dylib.py builds, so the
# two gates exercise one shape of the API and differ only in link mode.
ELEMENTS: list[str] = ["C", "C", "O"]
GOLDEN_X: list[float] = [0.0, 1.5, 3.0]
GOLDEN_X1: float = 1.5
POSITION_TOL: float = 1e-12  # exact-arithmetic position tolerance

SUPPORTED_PLATFORMS: dict[str, list[str]] = {
    "Darwin": ["otool", "-L"],
    "Linux": ["ldd"],
}


def refuse(reason: str, detail: list[str], remediation: list[str]) -> None:
    """Print a diagnosis plus a numbered remediation list, then exit 1."""
    # Flush first: the count line CI greps goes to stdout, the diagnosis to
    # stderr, and unflushed stdout would surface after it in any captured log.
    sys.stdout.flush()
    print(f"link-mode-static-default: FAIL — {reason}", file=sys.stderr)
    if detail:
        print(file=sys.stderr)
        for line in detail:
            print(f"  {line}", file=sys.stderr)
    print(file=sys.stderr)
    print("Remediation, in this order:", file=sys.stderr)
    for i, step in enumerate(remediation, start=1):
        print(f"  {i}. {step}", file=sys.stderr)
    sys.exit(1)


# ── 1. Locate the loaded extension ───────────────────────────────────────────
spec = importlib.util.find_spec("molrs._lib")
if spec is None or not spec.origin:
    refuse(
        "could not resolve the molrs._lib extension module.",
        [
            f"importlib.util.find_spec('molrs._lib') -> {spec!r}",
            f"origin                                 -> "
            f"{getattr(spec, 'origin', None)!r}",
            f"molrs package                          -> {molrs.__file__}",
            "",
            "This gate reads the load commands of the extension the running",
            "interpreter would actually import. With no origin there is no file",
            "to interrogate, and asserting on anything else would be asserting",
            "on a stand-in.",
        ],
        [
            "cd molrs-python && maturin develop   # or install a built wheel",
            "confirm the interpreter running this script is the one molrs was "
            "installed into (python -c 'import molrs; print(molrs.__file__)')",
        ],
    )
    raise SystemExit(1)  # unreachable; narrows `spec` for type checkers

extension_path: str = spec.origin
print(f"link-mode-static-default: extension = {extension_path}")

# ── 2. Read its dynamic-link entries ─────────────────────────────────────────
observed_platform: str = platform.system()
linker_cmd: list[str] | None = SUPPORTED_PLATFORMS.get(observed_platform)
if linker_cmd is None:
    refuse(
        f"unsupported platform {observed_platform!r}.",
        [
            f"observed platform : {observed_platform!r}",
            f"expected one of   : {sorted(SUPPORTED_PLATFORMS)}",
            "",
            "Link mode is a property of the platform's binary format, read here",
            "via otool (Mach-O) or ldd (ELF). There is no portable fallback, and",
            "skipping would leave a green run that proved nothing.",
        ],
        [
            "run this gate on macOS or Linux, where the release wheels are built",
            f"or add a {observed_platform} branch to SUPPORTED_PLATFORMS here and "
            "to scripts/verify-shared-dylib.sh, which makes the same choice",
            "then rerun: python regressions/link-mode-static-default.py",
        ],
    )
    raise SystemExit(1)  # unreachable; narrows `linker_cmd` for type checkers

argv: list[str] = [*linker_cmd, extension_path]
try:
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
except FileNotFoundError:
    refuse(
        f"{linker_cmd[0]!r} is not on PATH ({observed_platform}).",
        [f"tried: {' '.join(argv)}"],
        [
            "macOS: xcode-select --install   (ships otool)",
            "Linux: install your distro's libc-bin   (ships ldd)",
            "then rerun: python regressions/link-mode-static-default.py",
        ],
    )
    raise SystemExit(1)  # unreachable

link_output: str = completed.stdout + completed.stderr
if completed.returncode != 0:
    refuse(
        f"{linker_cmd[0]} could not read the extension "
        f"(exit {completed.returncode}).",
        [f"command: {' '.join(argv)}", "", *link_output.splitlines()],
        [
            "check the extension is a real shared object for this platform "
            f"(file {extension_path})",
            "cd molrs-python && maturin develop   # rebuild it",
            "then rerun: python regressions/link-mode-static-default.py",
        ],
    )

# otool -L:  "\t/abs/path/libmolrs_ffi.dylib (compatibility version ...)"
# ldd:       "\tlibmolrs_ffi.so => /abs/path/libmolrs_ffi.so (0x00007f...)"
# Either way one entry is one line, so lines are the countable unit.
ffi_lines: list[str] = [
    line.strip() for line in link_output.splitlines() if FFI_LIB_STEM in line
]
ffi_entry_count: int = len(ffi_lines)

# Printed unconditionally, in exactly this wording: a CI acceptance criterion
# greps this line, so it must appear on every run that got far enough to count.
print(f"{FFI_LIB_STEM} dynamic-link entry count == {ffi_entry_count}")

if ffi_entry_count != GOLDEN_FFI_ENTRY_COUNT:
    refuse(
        f"expected {GOLDEN_FFI_ENTRY_COUNT} {FFI_LIB_STEM} dynamic-link "
        f"entries (static), found {ffi_entry_count}.",
        [
            f"extension : {extension_path}",
            f"command   : {' '.join(argv)}",
            "",
            "offending entries:",
            *[f"  {line}" for line in ffi_lines],
            "",
            "Every entry above is an absolute path into a build tree. A wheel",
            "carrying one imports on the machine that built it and nowhere else.",
            "",
            "Static is what a build with no flags produces, so a red here means",
            "one of exactly two things: the .so this interpreter imports was",
            "built while the opt-in dynamic link form was in effect, or this",
            "build's command line opted into that form. Either is a legitimate",
            "state to be in — but it is the shape ffi-shared-dylib.py asserts,",
            "not this one. Do NOT relax the golden to accommodate it; rebuild",
            "with no flags and rerun.",
        ],
        [
            "rebuild with no flags — that already IS the release form:\n"
            "       cd molrs-python && maturin build --release",
            "install THAT wheel into the interpreter running this script, then "
            "rerun. An editable / maturin-develop install keeps pointing at "
            "whatever regime last built it, so a wheel you build but never "
            "install changes nothing here",
            "still red? then something opted into the dynamic link form on the "
            "command line, or via an exported RUSTFLAGS. Nothing committed sets "
            "it: .cargo/config.toml carries only [build] target-dir. "
            "docs/interop.md is the one place those flags live",
            "the two forms overwrite each other's hashless libmolrs_ffi, so a "
            "stale artifact can outlive the command line that built it: touch "
            "molrs-ffi/src/lib.rs and rebuild in the form you want "
            "(see CLAUDE.md, 'Build cache')",
        ],
    )

# ── 3. The public API still works ────────────────────────────────────────────
# Static linking that broke the extension would be no win. Same Block/Frame
# shape as ffi-shared-dylib.py, with hard-coded goldens.
block = molrs.Block()
block.insert("element", ELEMENTS)
block.insert("x", np.array(GOLDEN_X, dtype=np.float64))
block.insert("y", np.array([0.0, 0.0, 0.0], dtype=np.float64))
block.insert("z", np.array([0.0, 0.0, 0.0], dtype=np.float64))
block.insert("id", np.array([1, 2, 3], dtype=np.uint32))

frame = molrs.Frame()
frame["atoms"] = block

atoms = frame["atoms"]
read_elements: list[str] = [str(e) for e in atoms["element"]]
read_x1: float = float(atoms["x"][1])

api_failures: list[str] = []
if read_elements != ELEMENTS:
    api_failures.append(f"elements {read_elements!r} != golden {ELEMENTS!r}")
if abs(read_x1 - GOLDEN_X1) > POSITION_TOL:
    api_failures.append(
        f"x[1] {read_x1!r} != golden {GOLDEN_X1!r} (tol {POSITION_TOL:g})"
    )

if api_failures:
    refuse(
        "the extension is statically linked but its public API is wrong.",
        [
            f"extension : {extension_path}",
            "",
            *api_failures,
        ],
        [
            "this is a molrs bug, not a link-mode one — the Block/Frame round "
            "trip is the same one regressions/ffi-shared-dylib.py performs; run "
            "it to see whether the dynamically-linked build agrees",
            "cargo test -p molcrafts-molrs --lib --features full,filesystem",
            "cd molrs-python && maturin develop && pytest -q",
        ],
    )

# ── 4. Green ─────────────────────────────────────────────────────────────────
print(
    f"link-mode-static-default: OK — static extension "
    f"({observed_platform}/{linker_cmd[0]}), "
    f"{len(read_elements)}-atom Frame round-trips."
)
sys.exit(0)
