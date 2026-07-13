"""Translate the antechamber parameter tables into committed Rust data structures.

molrs parses NO parameter text at runtime. This script reads the upstream tables
from `$AMBERHOME/dat/antechamber/` and emits typed Rust `const`s into
`molrs/src/ff/params/generated/`. The emitted `.rs` is the single in-repo source
of truth: the raw `.DAT` / `.DEF` files are NOT vendored.

Ten tables:

  BCCPARM.DAT          -> bccparm.rs        (corrections + CORR aliases)
  BCCPARM_ABCG2.DAT    -> bccparm_abcg2.rs  (corrections + CORR aliases)
  GASPARM.DAT          -> gasparm.rs        (Gasteiger PEOE parameters)
  ATOMTYPE_{BCC,ABCG2,GAS,GFF,GFF2,AMBER,SYBYL}.DEF -> atomtype_*.rs

All seven `.DEF` files share ONE ATD / WILDATOM grammar, so there is one parser
here and seven outputs.

An eleventh upstream table is READ but not emitted on its own: `PARMCHK.DAT`
supplies the `AtdRule.alternate` column (see `parse_parmchk`). The `.DEF` files
emit only the phase-1 name of a conjugated system (`cc`); antechamber renames one
colour of each conjugated system to a partner (`cd`) that no `.DEF` row declares,
and `PARMCHK.DAT` is where that pairing is written down.

The environment (`f9`) and atom-property (`f8`) mini-languages are PRE-PARSED
into a static AST — that is the whole point: molrs walks the AST, it never
re-parses a string. Pattern atom names are resolved here too (`EW` / WILDATOM /
element symbol -> `PatternAtom`), so an unknown name is a generator error rather
than a rule that silently never matches.

Usage:
    AMBERHOME=/path/to/amber python scripts/gen_param_tables.py
    AMBERHOME=... python scripts/gen_param_tables.py --out-dir /tmp/check
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "molrs/src/ff/params/generated"

HEADER = """//! {title}
//!
//! DO NOT HAND-EDIT — regenerate with `scripts/gen_param_tables.py`.
//!
//! Source: `$AMBERHOME/dat/antechamber/{source}` (AmberTools).
"""

# rustfmt's `struct_lit_width` default (18) explodes every flat data row across
# six lines, which destroys the one-row-per-line grep-ability that is half the
# point of committing these tables. Pin the tabular consts; the nested ATD rules
# are genuinely tree-shaped and are left to rustfmt.
RUSTFMT_SKIP = "#[rustfmt::skip]"

# --- Element symbol -> atomic number ---------------------------------------
# Only the elements the antechamber tables actually name. A symbol outside this
# map is a hard error: it means the upstream grammar grew something new.
ELEMENTS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Fe": 26, "Cu": 29,
    "Zn": 30, "Br": 35, "I": 53,
}

# Atom-property tokens, mapped to the `AtomProp` Rust variant. The lowercase
# forms (`sb`/`db`/`tb`) count aromatic + delocalized bonds too; the uppercase
# forms are strict. Anything else is a generator error.
ATOM_PROPS = {
    "RG": "Rg", "RG3": "Rg3", "RG4": "Rg4", "RG5": "Rg5", "RG6": "Rg6",
    "RG7": "Rg7", "RG8": "Rg8", "RG9": "Rg9", "RG10": "Rg10",
    "NR": "Nr",
    "AR1": "Ar1", "AR2": "Ar2", "AR3": "Ar3", "AR4": "Ar4", "AR5": "Ar5",
    "SB": "SbStrict", "sb": "SbAny",
    "DB": "DbStrict", "db": "DbAny",
    "TB": "TbStrict", "tb": "TbAny",
    "AB": "Ab", "DL": "Dl",
}

ENV_BOND_TYPES = {
    "any": "Any", "ANY": "Any", "Any": "Any",
    "SB": "Single", "sb": "Single",
    "DB": "Double", "db": "Double",
    "TB": "Triple", "tb": "Triple",
    "AB": "Aromatic", "ab": "Aromatic",
}

FLOAT_RE = re.compile(r"^[+-]?\d+\.\d+$")

# (file, module, const, gaff_namespace)
#
# `gaff_namespace` — is this `.DEF` written in the atom-type namespace that
# PARMCHK.DAT's `equivalent_flag` column describes? Only `-at gaff` / `-at gaff2`
# are, and the column may only be applied to those two.
#
# It is a NAMESPACE fact, not a per-table special case, and it is the same fact
# the WILDATOM aliases already carry: a name means what the FILE that declares it
# says it means (`XB` is `C3 N2 N3 O2 S2 P2` in ATOMTYPE_BCC.DEF and `N P` in
# ATOMTYPE_GFF.DEF). PARMCHK.DAT's lowercase PARM rows are GAFF types, so its
# `cg` is GAFF's inner sp carbon of a conjugated system — but `ATOMTYPE_GAS.DEF`
# ALSO spells a type `cg`, and there it is a guanidinium carbon (`C` bonded to
# three `N3`), which is not half of anything: GASPARM.DAT has no `ch` row at all.
# The two `cg`s are one PARM row upstream and cannot be told apart inside
# PARMCHK.DAT — every other column of that row (mass, group, atomic number) is
# identical — so the scope has to be stated here.
#
# This is the same trap as ATOMTYPE_AMBER.DEF's `CC`/`CD` (parm94's histidine
# carbons, `equivalent_flag` 0), one table over: a shared SPELLING is not a
# shared meaning. The flag column catches the AMBER pair on its own; only the
# namespace catches the GAS one.
ATOMTYPE_FILES = [
    ("ATOMTYPE_BCC.DEF", "atomtype_bcc", "ATOMTYPE_BCC", False),
    ("ATOMTYPE_ABCG2.DEF", "atomtype_abcg2", "ATOMTYPE_ABCG2", False),
    ("ATOMTYPE_GAS.DEF", "atomtype_gas", "ATOMTYPE_GAS", False),
    ("ATOMTYPE_GFF.DEF", "atomtype_gff", "ATOMTYPE_GFF", True),
    ("ATOMTYPE_GFF2.DEF", "atomtype_gff2", "ATOMTYPE_GFF2", True),
    ("ATOMTYPE_AMBER.DEF", "atomtype_amber", "ATOMTYPE_AMBER", False),
    ("ATOMTYPE_SYBYL.DEF", "atomtype_sybyl", "ATOMTYPE_SYBYL", False),
]

#: The conjugated pairing table. Read for ONE column — `equivalent_flag` — which
#: is the only in-repo source of the phase-2 names (`cd`, `cf`, ...).
PARMCHK_FILE = "PARMCHK.DAT"

#: Every upstream table this generator consumes. Hashed into MANIFEST.sha256 so
#: the provenance of the committed `.rs` is recorded, not merely asserted.
SOURCE_FILES: list[str] = [
    "BCCPARM.DAT",
    "BCCPARM_ABCG2.DAT",
    "GASPARM.DAT",
    PARMCHK_FILE,
    *[name for name, _, _, _ in ATOMTYPE_FILES],
]


class GrammarError(Exception):
    """An upstream construct this generator refuses to guess at."""


# ---------------------------------------------------------------------------
# Shared lexing helpers
# ---------------------------------------------------------------------------

def is_comment(line: str) -> bool:
    """True for the banner / rule / comment lines every antechamber table carries."""
    return not line or line[0] in "#-=" or line.startswith("//")


def rust_float(token: str) -> str:
    """Emit a source float token verbatim when it is already a valid Rust literal.

    Keeps the committed table byte-comparable with the upstream text (`-0.0753`
    stays `-0.0753`), and round-trips exactly. Anything unusual falls back to a
    shortest-round-trip repr so we never silently lose a digit.
    """
    if FLOAT_RE.match(token):
        return token
    return repr(float(token))


def rust_str(s: str) -> str:
    if '"' in s or "\\" in s:
        raise GrammarError(f"unescapable string literal {s!r}")
    return f'"{s}"'


def opt(value: str | None) -> str:
    return f"Some({value})" if value is not None else "None"


# ---------------------------------------------------------------------------
# BCCPARM.DAT / BCCPARM_ABCG2.DAT
# ---------------------------------------------------------------------------

def parse_bccparm(path: Path) -> tuple[list[tuple[str, str, int, str]], list[tuple[str, str]]]:
    """Parse a `BCCPARM*.DAT` -> (correction rows, CORR aliases).

    Row layout: `index  left  right  bond_type  delta`. A `CORR a b` row declares
    that atom type `a` borrows `b`'s corrections. Trailing `#` comments on CORR
    rows are ignored (they are prose, not fields).
    """
    corrections: list[tuple[str, str, int, str]] = []
    aliases: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0].upper() == "CORR":
            if len(f) < 3:
                raise GrammarError(f"{path.name}:{lineno}: short CORR row: {raw}")
            aliases.append((f[1], f[2]))
            continue
        if len(f) < 5:
            raise GrammarError(f"{path.name}:{lineno}: short parameter row: {raw}")
        corrections.append((f[1], f[2], int(f[3]), rust_float(f[4])))
    return corrections, aliases


def emit_bccparm(path: Path, prefix: str) -> str:
    corrections, aliases = parse_bccparm(path)
    L = [HEADER.format(
        title=f"AM1-BCC bond charge corrections — `{path.name}`.",
        source=path.name,
    )]
    w = L.append
    w("use crate::ff::params::{BccAlias, BccCorrectionRow};")
    w("")
    w(f"/// The {len(corrections)} oriented bond charge corrections of `{path.name}`.")
    w("///")
    w("/// A row `(left, right, bond_type, delta)` adds `+delta` to the `left` atom and")
    w("/// `-delta` to the `right` atom. A bond encountered in the reverse orientation")
    w("/// applies the same magnitude with the sign flipped.")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_CORRECTIONS: &[BccCorrectionRow] = &[")
    for left, right, bond_type, delta in corrections:
        w(f"    BccCorrectionRow {{ left: {rust_str(left)}, right: {rust_str(right)}, "
          f"bond_type: {bond_type}, delta: {delta} }},")
    w("];")
    w("")
    if aliases:
        w(f"/// The {len(aliases)} `CORR` alias rows of `{path.name}`.")
        w("///")
        w("/// `atom_type` carries no corrections of its own; look `reference` up instead.")
    else:
        w(f"/// `{path.name}` declares no `CORR` alias rows.")
    w(RUSTFMT_SKIP)
    w(f"pub const {prefix}_ALIASES: &[BccAlias] = &[")
    for atom_type, reference in aliases:
        w(f"    BccAlias {{ atom_type: {rust_str(atom_type)}, reference: {rust_str(reference)} }},")
    w("];")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# GASPARM.DAT
# ---------------------------------------------------------------------------

def emit_gasparm(path: Path) -> str:
    """Parse `GASPARM.DAT` -> Gasteiger PEOE rows.

    Columns are `a b c d formal_charge`. `a`/`b`/`c` are the electronegativity
    polynomial coefficients; `d` is chi+ (the normalisation denominator, NOT a
    quartic coefficient); `formal_charge` is the seed charge q0. The three
    meanings get three distinct field names — never an anonymous `[f64; 5]`.
    """
    rows = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0] != "GASPARM":
            raise GrammarError(f"{path.name}:{lineno}: unknown row kind {f[0]!r}")
        if len(f) < 7:
            raise GrammarError(f"{path.name}:{lineno}: short GASPARM row: {raw}")
        rows.append((f[1], *[rust_float(v) for v in f[2:7]]))

    L = [HEADER.format(
        title=f"Gasteiger–Marsili PEOE parameters — `{path.name}`.",
        source=path.name,
    )]
    w = L.append
    w("use crate::ff::params::GasteigerRow;")
    w("")
    w(f"/// The {len(rows)} Gasteiger PEOE parameter rows of `{path.name}`.")
    w(RUSTFMT_SKIP)
    w("pub const GASTEIGER_PARAMS: &[GasteigerRow] = &[")
    for atom_type, a, b, c, chi_plus, seed in rows:
        w(f"    GasteigerRow {{ atom_type: {rust_str(atom_type)}, a: {a}, b: {b}, c: {c}, "
          f"chi_plus: {chi_plus}, seed_charge: {seed} }},")
    w("];")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# PARMCHK.DAT — the conjugated pairing (the `alternate` column)
# ---------------------------------------------------------------------------

#: The header's own enumeration of the pairing, e.g.
#:     #   equivalent_flag:  1 for cc/ce/cg/nc/ne/pc/pe
#:     #                     2 for cd/cf/ch/nd/nf/pd/pf
#: The trailing `0 for others` line cannot match: it names no `a/b/c` list.
EQUIVALENT_FLAG_RE = re.compile(
    r"^#\s*(?:equivalent_flag:\s*)?([12])\s+for\s+(\w+(?:/\w+)+)\s*$"
)


@dataclass(frozen=True)
class Equivalents:
    """The conjugated pairing `PARMCHK.DAT` declares, read from BOTH of its halves."""

    #: phase-1 atom type -> the phase-2 type antechamber renames it to.
    pairs: dict[str, str]
    #: atom type -> its `equivalent_flag` column, verbatim (sign included).
    flags: dict[str, int]


def parse_parmchk(path: Path) -> Equivalents:
    """Read the conjugated pairing out of `PARMCHK.DAT`.

    Two halves of the same fact, and BOTH are needed:

    * the **header** enumerates the pairing by name (`1 for cc/ce/cg/nc/ne/pc/pe`
      / `2 for cd/cf/ch/nd/nf/pd/pf`), zipped position-wise into `pairs`;
    * the per-type **`equivalent_flag` column** of each `PARM` row says whether
      that particular type takes part.

    They are cross-checked against each other, so an upstream edit to either one
    is a generator error rather than a silently wrong `alternate` column.
    """
    pairs: dict[str, str] = {}
    flags: dict[str, int] = {}
    phases: dict[int, list[str]] = {}

    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        header = EQUIVALENT_FLAG_RE.match(raw.rstrip())
        if header:
            phase, names = int(header[1]), header[2].split("/")
            if phase in phases:
                raise GrammarError(f"{path.name}:{lineno}: phase {phase} declared twice")
            phases[phase] = names
            continue
        if not raw.startswith("PARM"):
            continue
        f = raw.split()
        if len(f) < 7:
            raise GrammarError(f"{path.name}:{lineno}: short PARM row: {raw}")
        flags[f[1]] = int(f[5])

    if sorted(phases) != [1, 2]:
        raise GrammarError(
            f"{path.name}: expected the header to declare both equivalent_flag phases, "
            f"found {sorted(phases)}"
        )
    if len(phases[1]) != len(phases[2]):
        raise GrammarError(
            f"{path.name}: the two equivalent_flag phases have different lengths "
            f"({phases[1]} vs {phases[2]}); the pairing is positional and no longer zips"
        )

    for phase, names in phases.items():
        for name in names:
            if name not in flags:
                raise GrammarError(f"{path.name}: header names `{name}`, which has no PARM row")
            # The sign of the column distinguishes ring (-) from chain (+) types;
            # the phase is its magnitude.
            if abs(flags[name]) != phase:
                raise GrammarError(
                    f"{path.name}: header puts `{name}` in phase {phase} but its "
                    f"equivalent_flag column says {flags[name]}"
                )
    pairs = dict(zip(phases[1], phases[2], strict=True))
    return Equivalents(pairs=pairs, flags=flags)


def alternate_of(atom_type: str, eq: Equivalents | None) -> str | None:
    """The phase-2 name `atom_type` is renamed to on the other colour, or `None`.

    `eq` is `None` for a `.DEF` outside the GAFF namespace (see `ATOMTYPE_FILES`),
    where PARMCHK.DAT's column says nothing about the types the file declares.
    Within that namespace, three independent gates — three different things that
    must NOT be paired:

    * the **flag column**, looked up case-sensitively, is what a type must carry to
      take part at all. `ATOMTYPE_AMBER.DEF` has real `CC` and `CD` rows — parm94's
      histidine carbons — whose `equivalent_flag` is `0`. Keying on the *spelling*
      of a type instead of on its column would pair them, 2-colour the AMBER table,
      and break a column that reproduces antechamber 37/37 today.
    * the **header's enumeration** is what supplies the partner's name. A nonzero
      flag alone is not a pairing: upstream flags `cp` / `cq` (-1 / -2) but does not
      list them in the header, and `antechamber -at gaff` types biphenyl `cp cp`,
      never `cp cq` — the phase renaming does not reach them.
    * the **namespace** decides whether the column applies to this file at all.
    """
    if eq is None:
        return None
    if abs(eq.flags.get(atom_type, 0)) != 1:
        return None
    return eq.pairs.get(atom_type)


# ---------------------------------------------------------------------------
# ATOMTYPE_*.DEF — the ATD / WILDATOM grammar (one parser, seven outputs)
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    """One node of an environment expression, e.g. `C3[AR1](O1,O1)`."""

    name: str
    degree: int | None
    prop: str | None
    label: str | None
    children: list["Pattern"] = field(default_factory=list)


@dataclass
class Rule:
    """One `ATD` row."""

    atom_type: str
    residue: str
    atomic_number: int | None
    degree: int | None
    hydrogen_count: int | None
    ewd_count: int | None
    prop: str | None
    env: list[Pattern] | None
    env_bonds: list[tuple[str, str, str]] | None


def find_matching(s: str, start: int, op: str, cl: str) -> int:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == op:
            depth += 1
        elif s[i] == cl:
            depth -= 1
            if depth == 0:
                return i
    raise GrammarError(f"unmatched `{op}` in `{s}`")


def split_top_level(s: str, sep: str = ",") -> list[str]:
    out, start, depth = [], 0, 0
    for i, ch in enumerate(s):
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        elif ch == sep and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
    out.append(s[start:].strip())
    return out


def parse_atom_pattern(s: str, pos: int) -> tuple[Pattern, int]:
    """`Name[Degree][Property]<Label>(Children...)` — every part after Name optional."""
    name_start = pos
    while pos < len(s) and s[pos].isalpha():
        pos += 1
    if pos == name_start:
        raise GrammarError(f"expected an atom name in `{s}`")
    name = s[name_start:pos]

    degree_start = pos
    while pos < len(s) and s[pos].isdigit():
        pos += 1
    degree = int(s[degree_start:pos]) if pos > degree_start else None

    prop = label = None
    children: list[Pattern] = []
    while pos < len(s):
        ch = s[pos]
        if ch == "[":
            end = find_matching(s, pos, "[", "]")
            prop = s[pos:end + 1]
            pos = end + 1
        elif ch == "<":
            end = find_matching(s, pos, "<", ">")
            label = s[pos + 1:end]
            pos = end + 1
        elif ch == "(":
            end = find_matching(s, pos, "(", ")")
            # Consecutive groups CONCATENATE: `S4(O1)(O1)(O2)` means an S of
            # degree 4 with three neighbours, exactly like `S4(O1,O1,O2)`.
            # (ATOMTYPE_ABCG2.DEF:62, the sulfonate/sulfate-ester oxygen `3S`,
            # is the only row upstream that spells it this way.)
            children.extend(parse_pattern_list(s[pos + 1:end]))
            pos = end + 1
        else:
            break
    return Pattern(name, degree, prop, label, children), pos


def parse_pattern_list(s: str) -> list[Pattern]:
    out = []
    for part in split_top_level(s):
        if not part:
            continue
        pattern, pos = parse_atom_pattern(part, 0)
        if pos != len(part):
            raise GrammarError(f"trailing junk in environment fragment `{part}`")
        out.append(pattern)
    return out


def parse_environment(s: str) -> list[Pattern]:
    if not (s.startswith("(") and s.endswith(")")):
        raise GrammarError(f"environment must be parenthesised: `{s}`")
    return parse_pattern_list(s[1:-1])


def parse_env_bonds(s: str) -> list[tuple[str, str, str]]:
    """`a1:a2:any[,...]` — a bond constraint between two <labelled> environment atoms."""
    out = []
    for raw in s.split(","):
        parts = raw.split(":")
        if len(parts) != 3:
            raise GrammarError(f"environment bond must be `a:b:TYPE`: `{raw}`")
        if parts[2] not in ENV_BOND_TYPES:
            raise GrammarError(f"unknown environment bond type `{parts[2]}`")
        out.append((parts[0], parts[1], parts[2]))
    return out


def parse_prop_expr(s: str) -> list[list[tuple[int | None, str, bool | None]]]:
    """`[A.B,C]` -> AND of ORs: every comma-group must hold; a group holds if any
    dot-unit holds. A unit is `[count]PROP[' | '']`, where `'` additionally requires
    the bond back to the previous atom to be of that type and `''` requires it not to be.
    """
    if not (s.startswith("[") and s.endswith("]")):
        raise GrammarError(f"atom property must be bracketed: `{s}`")
    constraints = []
    for group in s[1:-1].split(","):
        units = []
        for unit in group.split("."):
            digits = 0
            while digits < len(unit) and unit[digits].isdigit():
                digits += 1
            count = int(unit[:digits]) if digits else None
            prop = unit[digits:]
            relation: bool | None = None
            if prop.endswith("''"):
                prop, relation = prop[:-2], False
            elif prop.endswith("'"):
                prop, relation = prop[:-1], True
            if prop not in ATOM_PROPS:
                raise GrammarError(f"unknown atom property `{prop}` in `{s}`")
            units.append((count, prop, relation))
        constraints.append(units)
    return constraints


def parse_wildatom_spec(token: str) -> tuple[int, int | None]:
    """`C3` -> (6, Some(3)); `O` -> (8, None)."""
    split = next((i for i, c in enumerate(token) if c.isdigit()), len(token))
    symbol, degree = token[:split], token[split:]
    if symbol not in ELEMENTS:
        raise GrammarError(f"unknown WILDATOM element symbol `{symbol}`")
    return ELEMENTS[symbol], int(degree) if degree else None


def parse_atomtype_def(path: Path) -> tuple[dict[str, list[tuple[int, int | None]]], list[Rule]]:
    wildatoms: dict[str, list[tuple[int, int | None]]] = {}
    rules: list[Rule] = []

    def num(f: list[str], i: int) -> int | None:
        v = f[i] if i < len(f) else None
        return None if v in (None, "*", "&") else int(v)

    def text(f: list[str], i: int) -> str | None:
        v = f[i] if i < len(f) else None
        return None if v in (None, "*", "&") else v

    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if is_comment(line):
            continue
        f = line.split()
        if f[0] == "WILDATOM":
            if len(f) < 3:
                raise GrammarError(f"{path.name}:{lineno}: short WILDATOM row")
            wildatoms[f[1]] = [parse_wildatom_spec(t) for t in f[2:]]
            continue
        if f[0] != "ATD":
            continue  # prose in the trailing "rules of defining an atom" section
        if len(f) < 3:
            raise GrammarError(f"{path.name}:{lineno}: short ATD row: {raw}")
        atom_type = f[1]
        # Every table ends with a constraint-free fall-through row — `ATD DU &`
        # (BCC / ABCG2 / GAS / GFF / GFF2 / AMBER) or `ATD ANY &` (SYBYL). It is
        # a NAMED RULE OF THE TABLE, not an invented fallback, and antechamber
        # reaches it: `-at amber` types nitromethane's two nitro oxygens and
        # DMSO's sulfoxide sulfur `DU`. Dropping it made molrs error where
        # antechamber answers, so it is emitted like any other row — with every
        # column unconstrained, which is what `&` in the residue column means.
        residue = text(f, 2) or "*"
        env = text(f, 8)
        env_bonds = text(f, 9)
        prop = text(f, 7)
        rules.append(Rule(
            atom_type=atom_type,
            residue=residue,
            atomic_number=num(f, 3),
            degree=num(f, 4),
            hydrogen_count=num(f, 5),
            ewd_count=num(f, 6),
            prop=prop,
            env=parse_environment(env) if env else None,
            env_bonds=parse_env_bonds(env_bonds) if env_bonds else None,
        ))
    return wildatoms, rules


# --- ATD -> Rust ------------------------------------------------------------

def wild_const(name: str) -> str:
    return f"WILD_{name.upper()}"


def emit_prop_expr(s: str) -> str:
    constraints = parse_prop_expr(s)
    out = []
    for units in constraints:
        rendered = []
        for count, prop, relation in units:
            rel = None
            if relation is True:
                rel = "PropRelation::BondedToPrev"
            elif relation is False:
                rel = "PropRelation::NotBondedToPrev"
            rendered.append(
                f"PropUnit {{ count: {opt(str(count) if count is not None else None)}, "
                f"prop: AtomProp::{ATOM_PROPS[prop]}, relation: {opt(rel)} }}"
            )
        out.append(f"PropConstraint {{ units: &[{', '.join(rendered)}] }}")
    return f"PropExpr {{ constraints: &[{', '.join(out)}] }}"


def emit_pattern(p: Pattern, wildatoms: dict) -> str:
    """Resolve the pattern's name here, so molrs never looks a name up at runtime."""
    if p.name == "EW":
        atom = "PatternAtom::ElectronWithdrawing"
    elif p.name in wildatoms:
        atom = f"PatternAtom::Wild({wild_const(p.name)})"
    elif p.name in ELEMENTS:
        atom = f"PatternAtom::Element({ELEMENTS[p.name]})"
    else:
        raise GrammarError(
            f"pattern atom `{p.name}` is neither EW, a WILDATOM of this file, nor an element"
        )
    children = ", ".join(emit_pattern(c, wildatoms) for c in p.children)
    return (
        f"AtomPattern {{ atom: {atom}, "
        f"degree: {opt(str(p.degree) if p.degree is not None else None)}, "
        f"property: {opt(emit_prop_expr(p.prop) if p.prop else None)}, "
        f"label: {opt(rust_str(p.label) if p.label else None)}, "
        f"children: &[{children}] }}"
    )


PARAM_TYPES = [
    "AtdRule", "AtdTable", "AtomPattern", "AtomProp", "EnvBond", "EnvBondType",
    "PatternAtom", "PropConstraint", "PropExpr", "PropRelation", "PropUnit",
    "WildAtom", "WildAtomSpec",
]


def emit_atomtype(path: Path, const: str, eq: Equivalents | None) -> str:
    wildatoms, rules = parse_atomtype_def(path)

    L: list[str] = []
    w = L.append

    for name, specs in wildatoms.items():
        pretty = " ".join(
            f"{sym}{d if d is not None else ''}"
            for sym, d in (
                (next(s for s, z in ELEMENTS.items() if z == zz), dd) for zz, dd in specs
            )
        )
        w(f"/// `WILDATOM {name} {pretty}`")
        w(RUSTFMT_SKIP)
        w(f"const {wild_const(name)}: &[WildAtomSpec] = &[")
        for z, degree in specs:
            w(f"    WildAtomSpec {{ z: {z}, degree: {opt(str(degree) if degree is not None else None)} }},")
        w("];")
        w("")

    w(f"/// The {len(wildatoms)} `WILDATOM` aliases declared by `{path.name}`.")
    w(RUSTFMT_SKIP)
    w("pub const WILDATOMS: &[WildAtom] = &[")
    for name in wildatoms:
        w(f"    WildAtom {{ name: {rust_str(name)}, specs: {wild_const(name)} }},")
    w("];")
    w("")

    paired = sum(1 for r in rules if alternate_of(r.atom_type, eq))
    w(f"/// The {len(rules)} `ATD` rules of `{path.name}`, in file order.")
    w("///")
    w("/// Order is significant: the FIRST rule that matches wins — which is why the")
    w("/// table's own last row is the constraint-free fall-through (`DU`, or `ANY` in")
    w("/// `ATOMTYPE_SYBYL.DEF`). It matches anything nothing above it matched, and")
    w("/// antechamber does reach it: `-at amber` types nitromethane's nitro oxygens")
    w("/// `DU`. It is a rule of the table, not a fallback the engine invents.")
    w("///")
    if paired:
        w(f"/// {paired} of these rules carry an `alternate`: the phase-2 name")
        w("/// `PARMCHK.DAT` pairs their atom type with. The rule emits the phase-1 name;")
        w("/// the typifier's 2-colouring pass renames one colour of each conjugated")
        w("/// system to the alternate, which is the only way a type no ATD row declares")
        w("/// (`cd`) is ever assigned.")
    else:
        w("/// No rule here carries an `alternate`: `PARMCHK.DAT`'s `equivalent_flag`")
        w("/// column describes the GAFF atom-type namespace, and this file is not written")
        w("/// in it. Nothing in this table is ever renamed by the 2-colouring pass.")
    w("pub const RULES: &[AtdRule] = &[")
    for r in rules:
        alternate = alternate_of(r.atom_type, eq)
        env = (
            "&[" + ", ".join(emit_pattern(p, wildatoms) for p in r.env) + "]"
            if r.env is not None else None
        )
        env_bonds = None
        if r.env_bonds is not None:
            bonds = ", ".join(
                f"EnvBond {{ a: {rust_str(a)}, b: {rust_str(b)}, "
                f"bond: EnvBondType::{ENV_BOND_TYPES[t]} }}"
                for a, b, t in r.env_bonds
            )
            env_bonds = f"&[{bonds}]"
        w("    AtdRule {")
        w(f"        atom_type: {rust_str(r.atom_type)},")
        w(f"        alternate: {opt(rust_str(alternate)) if alternate else 'None'},")
        w(f"        residue: {rust_str(r.residue)},")
        w(f"        atomic_number: {opt(str(r.atomic_number) if r.atomic_number is not None else None)},")
        w(f"        degree: {opt(str(r.degree) if r.degree is not None else None)},")
        w(f"        hydrogen_count: {opt(str(r.hydrogen_count) if r.hydrogen_count is not None else None)},")
        w(f"        ewd_count: {opt(str(r.ewd_count) if r.ewd_count is not None else None)},")
        w(f"        atom_property: {opt(emit_prop_expr(r.prop) if r.prop else None)},")
        w(f"        environment: {opt(env)},")
        w(f"        environment_bonds: {opt(env_bonds)},")
        w("    },")
    w("];")
    w("")
    w(f"/// `{path.name}` as one typed table.")
    w(f"pub const {const}: AtdTable = AtdTable {{")
    w(f"    name: {rust_str(path.name)},")
    w("    wildatoms: WILDATOMS,")
    w("    rules: RULES,")
    w("};")
    w("")

    # Import exactly the row types this file mentions — `PropRelation` only shows
    # up in the tables that actually use a `'` / `''` unit, and an unused import
    # is a `-D warnings` build failure.
    body = "\n".join(L)
    used = [t for t in PARAM_TYPES if re.search(rf"\b{t}\b", body)]
    head = HEADER.format(
        title=f"Antechamber atom-type definition rules — `{path.name}`.",
        source=path.name,
    )
    return f"{head}\nuse crate::ff::params::{{{', '.join(used)}}};\n\n{body}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def emit_mod(modules: list[str]) -> str:
    L = ["//! Antechamber parameter tables, compiled to typed Rust `const`s.",
         "//!",
         "//! DO NOT HAND-EDIT — regenerate with `scripts/gen_param_tables.py`.",
         "//!",
         "//! The row types live in [`crate::ff::params`]; this module holds only data.",
         ""]
    for m in sorted(modules):
        L.append(f"pub mod {m};")
    L.append("")
    return "\n".join(L)


def rustfmt(paths: list[Path]) -> None:
    """Format in place, so the committed output is `cargo fmt --check`-clean and the
    drift guard compares bytes that a formatter cannot perturb."""
    exe = shutil.which("rustfmt")
    if exe is None:
        raise SystemExit("rustfmt not found on PATH; cannot emit fmt-stable tables")
    subprocess.run([exe, "--edition", "2024", *[str(p) for p in paths]], check=True)


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, as lowercase hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where to write the generated .rs (default: the committed tree)")
    args = ap.parse_args()

    amberhome = os.environ.get("AMBERHOME")
    if not amberhome:
        raise SystemExit("AMBERHOME is not set; point it at an AmberTools install")
    src = Path(amberhome) / "dat/antechamber"
    if not src.is_dir():
        raise SystemExit(f"no antechamber tables under {src}")

    # Emit into a tempdir first: a mid-run GrammarError must not leave a
    # half-written table behind in the repo.
    # The conjugated pairing feeds every ATD table's `alternate` column, so it is
    # read once, up front: a malformed PARMCHK.DAT must fail before anything is
    # emitted, not leave six good tables and one silently unpaired.
    equivalents = parse_parmchk(src / PARMCHK_FILE)

    with tempfile.TemporaryDirectory(prefix="param-tables-") as tmp:
        staged: dict[str, str] = {}
        staged["bccparm.rs"] = emit_bccparm(src / "BCCPARM.DAT", "BCC")
        staged["bccparm_abcg2.rs"] = emit_bccparm(src / "BCCPARM_ABCG2.DAT", "ABCG2")
        staged["gasparm.rs"] = emit_gasparm(src / "GASPARM.DAT")
        for filename, module, const, gaff_namespace in ATOMTYPE_FILES:
            staged[f"{module}.rs"] = emit_atomtype(
                src / filename, const, equivalents if gaff_namespace else None
            )
        modules = [name[:-3] for name in staged]
        staged["mod.rs"] = emit_mod(modules)

        tmp_paths = []
        for name, text in staged.items():
            p = Path(tmp) / name
            p.write_text(text)
            tmp_paths.append(p)
        rustfmt(tmp_paths)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        for p in tmp_paths:
            shutil.copyfile(p, args.out_dir / p.name)

        # MANIFEST.sha256 — the ONLY drift check that works without AmberTools.
        # The byte-regeneration guard needs $AMBERHOME, which CI does not have,
        # so it is permanently skipped there. Hashing the emitted files lets CI
        # still catch a hand-edit to the 27k lines of generated tables.
        # Source hashes record provenance and are checked only when $AMBERHOME
        # is available.
        lines = ["# Generated by scripts/gen_param_tables.py — DO NOT HAND-EDIT.",
                 "# emitted <sha256>  <file>"]
        for p in sorted(tmp_paths, key=lambda q: q.name):
            lines.append(f"emitted {sha256_of(args.out_dir / p.name)}  {p.name}")
        lines.append("# source  <sha256>  <upstream table>")
        for name in sorted(SOURCE_FILES):
            lines.append(f"source  {sha256_of(src / name)}  {name}")
        (args.out_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    for name in sorted(staged):
        out = args.out_dir / name
        print(f"  wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}"
              f"  ({len(out.read_text().splitlines())} lines)")
    print(f"{len(staged)} files -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
