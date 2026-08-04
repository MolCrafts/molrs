"""Smoke tests for the ``molrs.ff.ForceField.subset`` PyO3 binding.

The deep per-type projection correctness is covered by the Rust unit tests in
``molrs-ff/src/forcefield/mod.rs``. These tests only verify the binding is wired
end-to-end: a typed :class:`molrs.Frame` goes in, a smaller :class:`ForceField`
comes out, and the source force field is left unmodified.

``style_names()`` reports one entry per *style* (``category:name``), not per
type, so the Python-observable signal of pruning is a whole style being dropped
once all of its types are unused. The fixture is built so the angle style is
pruned away while the pair and bond styles survive.
"""

import numpy as np

import molrs

# A force field spanning three styles. The angle style has a single type whose
# atoms are never referenced by the typed frame below, so subset drops it.
_FF_XML = """<ForceField name="fixture">
 <PairStyle name="lj/cut" cutoff="10.0">
  <Type name="A" epsilon="0.1" sigma="3.0"/>
  <Type name="B" epsilon="0.2" sigma="3.5"/>
  <Type name="C" epsilon="0.3" sigma="4.0"/>
 </PairStyle>
 <BondStyle name="harmonic">
  <Type name="A-B" k="300" r0="1.5"/>
  <Type name="A-C" k="320" r0="1.6"/>
 </BondStyle>
 <AngleStyle name="harmonic">
  <Type name="A-B-A" k="40" theta0="1.9"/>
 </AngleStyle>
</ForceField>"""


def _full_ff():
    return molrs.ff.read_forcefield_xml_str(_FF_XML)


def _typed_frame():
    """Atoms {A, B} and bonds {A-B}; no angles block (angle style unused)."""
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("type", np.array(["A", "B", "A"], dtype=str))
    frame["atoms"] = atoms
    bonds = molrs.Block()
    bonds.insert("type", np.array(["A-B"], dtype=str))
    frame["bonds"] = bonds
    return frame


def test_subset_returns_a_forcefield():
    mini = _full_ff().subset(_typed_frame())
    assert isinstance(mini, molrs.ff.ForceField)


def test_subset_is_no_larger_than_source():
    ff = _full_ff()
    mini = ff.subset(_typed_frame())
    assert len(mini.style_names()) <= len(ff.style_names())


def test_subset_excludes_unused_style():
    mini = _full_ff().subset(_typed_frame())
    names = set(mini.style_names())
    # the pair and bond styles are referenced and survive ...
    assert "pair:lj/cut" in names
    assert "bond:harmonic" in names
    # ... while the unreferenced angle style is pruned away entirely.
    assert "angle:harmonic" not in names


def test_subset_does_not_mutate_source():
    ff = _full_ff()
    before = sorted(ff.style_names())
    ff.subset(_typed_frame())
    assert sorted(ff.style_names()) == before
