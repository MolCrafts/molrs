#!/usr/bin/env python3
"""Parse RDKit's MMFF parameter tables from Params.cpp and generate XML files.

Reads the C++ source containing embedded MMFF94 force field parameter tables,
extracts all 16 string literal tables, parses their tab-separated data,
and writes data/mmff94.xml and data/mmff94s.xml.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data containers (immutable NamedTuples)
# ---------------------------------------------------------------------------

class AtomTypeDef(NamedTuple):
    symbol: str
    mmff_type: int
    eq2: int
    eq3: int
    eq4: int
    eq5: int


class AtomProp(NamedTuple):
    atype: int
    aspec: int
    crd: int
    val: int
    pilp: int
    mltb: int
    arom: int
    linh: int
    sbmb: int


class PBCI(NamedTuple):
    type_: int
    atype: int
    pbci: float
    fcadj: float


class BondCharge(NamedTuple):
    bond_type: int
    i_type: int
    j_type: int
    bci: float


class BondStretch(NamedTuple):
    bond_type: int
    i_type: int
    j_type: int
    kb: float
    r0: float


class Bndk(NamedTuple):
    atno1: int
    atno2: int
    r0: float
    kb: float


class HerschbachLaurie(NamedTuple):
    i: int
    j: int
    a_ij: float
    d_ij: float
    dp_ij: float


class CovRadPauEle(NamedTuple):
    atno: int
    r0: float
    chi: float


class AngleBend(NamedTuple):
    angle_type: int
    i_type: int
    j_type: int
    k_type: int
    ka: float
    theta0: float


class StretchBend(NamedTuple):
    stbn_type: int
    i_type: int
    j_type: int
    k_type: int
    kba_ijk: float
    kba_kji: float


class DefaultStretchBend(NamedTuple):
    row_i: int
    row_j: int
    row_k: int
    kba_ijk: float
    kba_kji: float


class OutOfPlane(NamedTuple):
    i_type: int
    j_type: int
    k_type: int
    l_type: int
    koop: float


class Torsion(NamedTuple):
    tor_type: int
    i_type: int
    j_type: int
    k_type: int
    l_type: int
    v1: float
    v2: float
    v3: float


class VdWGlobal(NamedTuple):
    power: float
    B: float
    Beta: float
    DARAD: float
    DAEPS: float


class VdW(NamedTuple):
    atype: int
    alpha_i: float
    n_i: float
    a_i: float
    g_i: float
    da: str
    symbol: str


# ---------------------------------------------------------------------------
# C++ string literal extraction
# ---------------------------------------------------------------------------

def extract_simple_string(src: str, var_name: str) -> str:
    """Extract a single C++ string literal assigned to *var_name*.

    Handles multi-line concatenated string literals like:
        const std::string foo =
            "line1\\n"
            "line2\\n";
    """
    pattern = rf'const\s+std::string\s+{re.escape(var_name)}\s*=\s*'
    match = re.search(pattern, src)
    if match is None:
        raise ValueError(f"Could not find variable {var_name}")

    pos = match.end()
    fragments: list[str] = []

    while pos < len(src):
        # Skip whitespace
        while pos < len(src) and src[pos] in ' \t\n\r':
            pos += 1

        if pos >= len(src):
            break

        if src[pos] == '"':
            # Find the closing quote (handle escaped chars)
            end = pos + 1
            while end < len(src):
                if src[end] == '\\':
                    end += 2
                    continue
                if src[end] == '"':
                    break
                end += 1
            fragment = src[pos + 1:end]
            fragments.append(fragment)
            pos = end + 1
        elif src[pos] == ';':
            break
        else:
            pos += 1

    raw = ''.join(fragments)
    return _unescape_cpp(raw)


def extract_array_strings(src: str, var_name: str) -> str:
    """Extract a C++ array of string literals (like defaultMMFFAngleData[]).

    Returns the concatenated, unescaped content of all array elements
    (excluding the "EOS" sentinel).
    """
    pattern = rf'const\s+std::string\s+{re.escape(var_name)}\s*\[\s*\]\s*=\s*\{{'
    match = re.search(pattern, src)
    if match is None:
        raise ValueError(f"Could not find array variable {var_name}")

    pos = match.end()
    brace_depth = 1
    all_fragments: list[str] = []
    current_element_fragments: list[str] = []

    while pos < len(src) and brace_depth > 0:
        ch = src[pos]

        if ch in ' \t\n\r':
            pos += 1
            continue

        if ch == '"':
            end = pos + 1
            while end < len(src):
                if src[end] == '\\':
                    end += 2
                    continue
                if src[end] == '"':
                    break
                end += 1
            fragment = src[pos + 1:end]
            current_element_fragments.append(fragment)
            pos = end + 1
            continue

        if ch == ',':
            element_text = ''.join(current_element_fragments)
            if element_text != 'EOS':
                all_fragments.append(element_text)
            current_element_fragments = []
            pos += 1
            continue

        if ch == '}':
            brace_depth -= 1
            if brace_depth == 0:
                element_text = ''.join(current_element_fragments)
                if element_text != 'EOS':
                    all_fragments.append(element_text)
            pos += 1
            continue

        if ch == '{':
            brace_depth += 1
            pos += 1
            continue

        pos += 1

    raw = ''.join(all_fragments)
    return _unescape_cpp(raw)


def _unescape_cpp(raw: str) -> str:
    """Convert C++ escape sequences to Python strings."""
    result = raw.replace('\\n', '\n')
    result = result.replace('\\t', '\t')
    result = result.replace('\\"', '"')
    result = result.replace('\\\\', '\\')
    return result


# ---------------------------------------------------------------------------
# Table parsers
# ---------------------------------------------------------------------------

def _data_lines(text: str):
    """Yield non-empty, non-comment lines from extracted table text."""
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('*'):
            continue
        yield stripped


def parse_def(text: str) -> list[AtomTypeDef]:
    rows: list[AtomTypeDef] = []
    seen_types: set[int] = set()
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 7:
            continue
        symbol = cols[0].strip()
        mmff_type = int(cols[1])
        if mmff_type in seen_types:
            continue
        seen_types.add(mmff_type)
        rows.append(AtomTypeDef(
            symbol=symbol,
            mmff_type=mmff_type,
            eq2=int(cols[2]),
            eq3=int(cols[3]),
            eq4=int(cols[4]),
            eq5=int(cols[5]),
        ))
    return rows


def parse_prop(text: str) -> list[AtomProp]:
    rows: list[AtomProp] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 9:
            continue
        rows.append(AtomProp(
            atype=int(cols[0]),
            aspec=int(cols[1]),
            crd=int(cols[2]),
            val=int(cols[3]),
            pilp=int(cols[4]),
            mltb=int(cols[5]),
            arom=int(cols[6]),
            linh=int(cols[7]),
            sbmb=int(cols[8]),
        ))
    return rows


def parse_pbci(text: str) -> list[PBCI]:
    rows: list[PBCI] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 4:
            continue
        rows.append(PBCI(
            type_=int(cols[0]),
            atype=int(cols[1]),
            pbci=float(cols[2]),
            fcadj=float(cols[3]),
        ))
    return rows


def parse_chg(text: str) -> list[BondCharge]:
    rows: list[BondCharge] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 4:
            continue
        rows.append(BondCharge(
            bond_type=int(cols[0]),
            i_type=int(cols[1]),
            j_type=int(cols[2]),
            bci=float(cols[3]),
        ))
    return rows


def parse_bond(text: str) -> list[BondStretch]:
    rows: list[BondStretch] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 5:
            continue
        rows.append(BondStretch(
            bond_type=int(cols[0]),
            i_type=int(cols[1]),
            j_type=int(cols[2]),
            kb=float(cols[3]),
            r0=float(cols[4]),
        ))
    return rows


def parse_bndk(text: str) -> list[Bndk]:
    rows: list[Bndk] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 4:
            continue
        rows.append(Bndk(
            atno1=int(cols[0]),
            atno2=int(cols[1]),
            r0=float(cols[2]),
            kb=float(cols[3]),
        ))
    return rows


def parse_herschbach_laurie(text: str) -> list[HerschbachLaurie]:
    rows: list[HerschbachLaurie] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 5:
            continue
        rows.append(HerschbachLaurie(
            i=int(cols[0]),
            j=int(cols[1]),
            a_ij=float(cols[2]),
            d_ij=float(cols[3]),
            dp_ij=float(cols[4]),
        ))
    return rows


def parse_covrad(text: str) -> list[CovRadPauEle]:
    rows: list[CovRadPauEle] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 3:
            continue
        rows.append(CovRadPauEle(
            atno=int(cols[0]),
            r0=float(cols[1]),
            chi=float(cols[2]),
        ))
    return rows


def parse_angle(text: str) -> list[AngleBend]:
    rows: list[AngleBend] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 6:
            continue
        rows.append(AngleBend(
            angle_type=int(cols[0]),
            i_type=int(cols[1]),
            j_type=int(cols[2]),
            k_type=int(cols[3]),
            ka=float(cols[4]),
            theta0=float(cols[5]),
        ))
    return rows


def parse_stbn(text: str) -> list[StretchBend]:
    rows: list[StretchBend] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 6:
            continue
        rows.append(StretchBend(
            stbn_type=int(cols[0]),
            i_type=int(cols[1]),
            j_type=int(cols[2]),
            k_type=int(cols[3]),
            kba_ijk=float(cols[4]),
            kba_kji=float(cols[5]),
        ))
    return rows


def parse_dfsb(text: str) -> list[DefaultStretchBend]:
    rows: list[DefaultStretchBend] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 5:
            continue
        rows.append(DefaultStretchBend(
            row_i=int(cols[0]),
            row_j=int(cols[1]),
            row_k=int(cols[2]),
            kba_ijk=float(cols[3]),
            kba_kji=float(cols[4]),
        ))
    return rows


def parse_oop(text: str) -> list[OutOfPlane]:
    rows: list[OutOfPlane] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 5:
            continue
        rows.append(OutOfPlane(
            i_type=int(cols[0]),
            j_type=int(cols[1]),
            k_type=int(cols[2]),
            l_type=int(cols[3]),
            koop=float(cols[4]),
        ))
    return rows


def parse_tor(text: str) -> list[Torsion]:
    rows: list[Torsion] = []
    for line in _data_lines(text):
        cols = line.split('\t')
        if len(cols) < 8:
            continue
        rows.append(Torsion(
            tor_type=int(cols[0]),
            i_type=int(cols[1]),
            j_type=int(cols[2]),
            k_type=int(cols[3]),
            l_type=int(cols[4]),
            v1=float(cols[5]),
            v2=float(cols[6]),
            v3=float(cols[7]),
        ))
    return rows


def parse_vdw(text: str) -> tuple[VdWGlobal, list[VdW]]:
    rows: list[VdW] = []
    vdw_global: VdWGlobal | None = None

    for line in _data_lines(text):
        cols = line.split('\t')

        # First data line is global params: power, B, Beta, DARAD, DAEPS
        if vdw_global is None:
            if len(cols) < 5:
                continue
            vdw_global = VdWGlobal(
                power=float(cols[0]),
                B=float(cols[1]),
                Beta=float(cols[2]),
                DARAD=float(cols[3]),
                DAEPS=float(cols[4]),
            )
            continue

        if len(cols) < 7:
            continue
        rows.append(VdW(
            atype=int(cols[0]),
            alpha_i=float(cols[1]),
            n_i=float(cols[2]),
            a_i=float(cols[3]),
            g_i=float(cols[4]),
            da=cols[5].strip(),
            symbol=cols[6].strip(),
        ))

    if vdw_global is None:
        raise ValueError("No global VdW parameters found")

    return vdw_global, rows


# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Format a float: strip trailing zeros but keep at least one decimal."""
    s = f"{value:.4f}"
    # Remove trailing zeros after decimal point, keep at least X.0
    if '.' in s:
        s = s.rstrip('0')
        if s.endswith('.'):
            s += '0'
    return s


def _fmt3(value: float) -> str:
    """Format a float with 3 decimal places."""
    s = f"{value:.3f}"
    if '.' in s:
        s = s.rstrip('0')
        if s.endswith('.'):
            s += '0'
    return s


def build_xml(
    ff_name: str,
    defs: list[AtomTypeDef],
    props: list[AtomProp],
    pbcis: list[PBCI],
    charges: list[BondCharge],
    bonds: list[BondStretch],
    angles: list[AngleBend],
    stretch_bends: list[StretchBend],
    default_stbn: list[DefaultStretchBend],
    oops: list[OutOfPlane],
    torsions: list[Torsion],
    vdw_global: VdWGlobal,
    vdws: list[VdW],
    bndks: list[Bndk],
    herschbach: list[HerschbachLaurie],
    covrad: list[CovRadPauEle],
) -> ET.Element:
    root = ET.Element("ForceField", name=ff_name)

    # AtomTypes
    atom_types_el = ET.SubElement(root, "AtomTypes")
    for d in defs:
        ET.SubElement(atom_types_el, "Type", id=str(d.mmff_type), name=d.symbol)

    # AtomProperties
    atom_props_el = ET.SubElement(root, "AtomProperties")
    for p in props:
        ET.SubElement(atom_props_el, "Prop",
                      type=str(p.atype),
                      atno=str(p.aspec),
                      crd=str(p.crd),
                      val=str(p.val),
                      pilp=str(p.pilp),
                      mltb=str(p.mltb),
                      arom=str(p.arom),
                      linh=str(p.linh),
                      sbmb=str(p.sbmb))

    # EquivalenceTable
    equiv_el = ET.SubElement(root, "EquivalenceTable")
    for d in defs:
        ET.SubElement(equiv_el, "Def",
                      type=str(d.mmff_type),
                      eq1=str(d.eq2),
                      eq2=str(d.eq3),
                      eq3=str(d.eq4),
                      eq4=str(d.eq5))

    # BondChargeIncrements
    bci_el = ET.SubElement(root, "BondChargeIncrements")
    for c in charges:
        ET.SubElement(bci_el, "BCI",
                      bond_type=str(c.bond_type),
                      type1=str(c.i_type),
                      type2=str(c.j_type),
                      bci=_fmt(c.bci))

    # PartialBondChargeIncrements
    pbci_el = ET.SubElement(root, "PartialBondChargeIncrements")
    for p in pbcis:
        ET.SubElement(pbci_el, "PBCI",
                      type=str(p.atype),
                      pbci=_fmt(p.pbci),
                      fcadj=_fmt(p.fcadj))

    # BondStretchParams
    bond_el = ET.SubElement(root, "BondStretchParams")
    for b in bonds:
        ET.SubElement(bond_el, "Bond",
                      bond_type=str(b.bond_type),
                      type1=str(b.i_type),
                      type2=str(b.j_type),
                      kb=_fmt3(b.kb),
                      r0=_fmt3(b.r0))

    # AngleBendParams
    angle_el = ET.SubElement(root, "AngleBendParams")
    for a in angles:
        ET.SubElement(angle_el, "Angle",
                      angle_type=str(a.angle_type),
                      type1=str(a.i_type),
                      type2=str(a.j_type),
                      type3=str(a.k_type),
                      ka=_fmt3(a.ka),
                      theta0=_fmt3(a.theta0))

    # StretchBendParams
    stbn_el = ET.SubElement(root, "StretchBendParams")
    for s in stretch_bends:
        ET.SubElement(stbn_el, "StretchBend",
                      stbn_type=str(s.stbn_type),
                      type1=str(s.i_type),
                      type2=str(s.j_type),
                      type3=str(s.k_type),
                      kba_ijk=_fmt3(s.kba_ijk),
                      kba_kji=_fmt3(s.kba_kji))

    # DefaultStretchBend
    dfsb_el = ET.SubElement(root, "DefaultStretchBend")
    for d in default_stbn:
        ET.SubElement(dfsb_el, "Dfsb",
                      row_i=str(d.row_i),
                      row_j=str(d.row_j),
                      row_k=str(d.row_k),
                      kba_ijk=_fmt3(d.kba_ijk),
                      kba_kji=_fmt3(d.kba_kji))

    # OutOfPlaneParams
    oop_el = ET.SubElement(root, "OutOfPlaneParams")
    for o in oops:
        ET.SubElement(oop_el, "Oop",
                      type1=str(o.i_type),
                      type2=str(o.j_type),
                      type3=str(o.k_type),
                      type4=str(o.l_type),
                      koop=_fmt3(o.koop))

    # TorsionParams
    tor_el = ET.SubElement(root, "TorsionParams")
    for t in torsions:
        ET.SubElement(tor_el, "Torsion",
                      tor_type=str(t.tor_type),
                      type1=str(t.i_type),
                      type2=str(t.j_type),
                      type3=str(t.k_type),
                      type4=str(t.l_type),
                      v1=_fmt3(t.v1),
                      v2=_fmt3(t.v2),
                      v3=_fmt3(t.v3))

    # VdWParams
    vdw_el = ET.SubElement(root, "VdWParams",
                           B=_fmt(vdw_global.B),
                           Beta=_fmt(vdw_global.Beta),
                           DARAD=_fmt(vdw_global.DARAD),
                           DAEPS=_fmt(vdw_global.DAEPS))
    for v in vdws:
        ET.SubElement(vdw_el, "VdW",
                      type=str(v.atype),
                      alpha=_fmt3(v.alpha_i),
                      n_eff=_fmt3(v.n_i),
                      a_i=_fmt3(v.a_i),
                      g_i=_fmt3(v.g_i),
                      da=v.da,
                      symbol=v.symbol)

    # EmpiricalBondRules
    emp_el = ET.SubElement(root, "EmpiricalBondRules")
    for c in covrad:
        ET.SubElement(emp_el, "CovRadPauEle",
                      atno=str(c.atno),
                      r0=_fmt3(c.r0),
                      chi=_fmt3(c.chi))
    for h in herschbach:
        ET.SubElement(emp_el, "HerschbachLaurie",
                      row_i=str(h.i),
                      row_j=str(h.j),
                      a_ij=_fmt3(h.a_ij),
                      d_ij=_fmt3(h.d_ij),
                      dp_ij=_fmt3(h.dp_ij))
    for b in bndks:
        ET.SubElement(emp_el, "Bndk",
                      atno_i=str(b.atno1),
                      atno_j=str(b.atno2),
                      r0=_fmt3(b.r0),
                      kb=_fmt3(b.kb))

    return root


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to an ElementTree in-place for readable output."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
    if level == 0:
        elem.tail = "\n"


def write_xml(root: ET.Element, path: Path) -> None:
    indent_xml(root)
    tree = ET.ElementTree(root)
    with open(path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    # Append final newline
    with open(path, 'a') as f:
        f.write('\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    default_cpp = Path("/Users/roykid/work/rdkit/Code/ForceField/MMFF/Params.cpp")

    if len(sys.argv) > 1:
        cpp_path = Path(sys.argv[1])
    else:
        cpp_path = default_cpp

    if not cpp_path.exists():
        print(f"Error: {cpp_path} not found", file=sys.stderr)
        sys.exit(1)

    src = cpp_path.read_text(encoding='utf-8')
    print(f"Read {len(src)} bytes from {cpp_path}")

    # Create output directory (repo_root/data/)
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Extract and parse all tables
    # -----------------------------------------------------------------------

    print("\n--- Extracting tables ---")

    def_text = extract_simple_string(src, "defaultMMFFDef")
    defs = parse_def(def_text)
    print(f"  AtomTypeDef:         {len(defs)} rows")

    prop_text = extract_simple_string(src, "defaultMMFFProp")
    props = parse_prop(prop_text)
    print(f"  AtomProp:            {len(props)} rows")

    pbci_text = extract_simple_string(src, "defaultMMFFPBCI")
    pbcis = parse_pbci(pbci_text)
    print(f"  PBCI:                {len(pbcis)} rows")

    chg_text = extract_simple_string(src, "defaultMMFFChg")
    charges = parse_chg(chg_text)
    print(f"  BondCharge:          {len(charges)} rows")

    bond_text = extract_simple_string(src, "defaultMMFFBond")
    bonds = parse_bond(bond_text)
    print(f"  BondStretch:         {len(bonds)} rows")

    bndk_text = extract_simple_string(src, "defaultMMFFBndk")
    bndks = parse_bndk(bndk_text)
    print(f"  Bndk:                {len(bndks)} rows")

    hl_text = extract_simple_string(src, "defaultMMFFHerschbachLaurie")
    herschbach = parse_herschbach_laurie(hl_text)
    print(f"  HerschbachLaurie:    {len(herschbach)} rows")

    covrad_text = extract_simple_string(src, "defaultMMFFCovRadPauEle")
    covrad = parse_covrad(covrad_text)
    print(f"  CovRadPauEle:        {len(covrad)} rows")

    angle_text = extract_array_strings(src, "defaultMMFFAngleData")
    angles = parse_angle(angle_text)
    print(f"  AngleBend:           {len(angles)} rows")

    stbn_text = extract_simple_string(src, "defaultMMFFStbn")
    stretch_bends = parse_stbn(stbn_text)
    print(f"  StretchBend:         {len(stretch_bends)} rows")

    dfsb_text = extract_simple_string(src, "defaultMMFFDfsb")
    default_stbn = parse_dfsb(dfsb_text)
    print(f"  DefaultStretchBend:  {len(default_stbn)} rows")

    oop_text = extract_simple_string(src, "defaultMMFFOop")
    oops_94 = parse_oop(oop_text)
    print(f"  OutOfPlane (94):     {len(oops_94)} rows")

    soop_text = extract_simple_string(src, "defaultMMFFsOop")
    oops_94s = parse_oop(soop_text)
    print(f"  OutOfPlane (94s):    {len(oops_94s)} rows")

    tor_text = extract_simple_string(src, "defaultMMFFTor")
    torsions_94 = parse_tor(tor_text)
    print(f"  Torsion (94):        {len(torsions_94)} rows")

    stor_text = extract_simple_string(src, "defaultMMFFsTor")
    torsions_94s = parse_tor(stor_text)
    print(f"  Torsion (94s):       {len(torsions_94s)} rows")

    vdw_text = extract_simple_string(src, "defaultMMFFVdW")
    vdw_global, vdws = parse_vdw(vdw_text)
    print(f"  VdW:                 {len(vdws)} rows (global: power={vdw_global.power})")

    # -----------------------------------------------------------------------
    # Generate MMFF94 XML
    # -----------------------------------------------------------------------

    print("\n--- Generating XML ---")

    root_94 = build_xml(
        ff_name="MMFF94",
        defs=defs,
        props=props,
        pbcis=pbcis,
        charges=charges,
        bonds=bonds,
        angles=angles,
        stretch_bends=stretch_bends,
        default_stbn=default_stbn,
        oops=oops_94,
        torsions=torsions_94,
        vdw_global=vdw_global,
        vdws=vdws,
        bndks=bndks,
        herschbach=herschbach,
        covrad=covrad,
    )
    out_94 = data_dir / "mmff94.xml"
    write_xml(root_94, out_94)
    print(f"  Wrote {out_94}")

    # -----------------------------------------------------------------------
    # Generate MMFF94s XML (different OOP and torsion tables)
    # -----------------------------------------------------------------------

    root_94s = build_xml(
        ff_name="MMFF94s",
        defs=defs,
        props=props,
        pbcis=pbcis,
        charges=charges,
        bonds=bonds,
        angles=angles,
        stretch_bends=stretch_bends,
        default_stbn=default_stbn,
        oops=oops_94s,
        torsions=torsions_94s,
        vdw_global=vdw_global,
        vdws=vdws,
        bndks=bndks,
        herschbach=herschbach,
        covrad=covrad,
    )
    out_94s = data_dir / "mmff94s.xml"
    write_xml(root_94s, out_94s)
    print(f"  Wrote {out_94s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
