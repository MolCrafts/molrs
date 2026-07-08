"""Python-binding coverage for the native reaction-SMARTS (SMIRKS) engine.

Exposes the validated Rust ``Reaction`` (``molrs/src/core/chem/smarts/reaction.rs``)
to Python: parse a Daylight reaction SMARTS ``reactants >> products``, read back
the reactant components + forming-bond map pairs, and apply the compiled
transform to one matched occurrence in place. The transform compilation itself
(map diff, unmapped-atom pairing, form/break/set-order) is exercised in Rust;
these tests assert the PyO3 surface and Daylight atom-map semantics.
"""

import pytest

import molrs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _amine_plus_ester():
    """A single Atomistic holding two disconnected fragments.

    Amine  : N0 with two explicit H (matches ``[N;H2:1]``).
    Ester  : carbonyl C0 =O1, single -O2- to methyl C3
             (matches ``[C:2](=O)OC``; leaving group = O2 + C3).
    """
    mol = molrs.Atomistic()
    # amine
    n0 = mol.add_atom("N", 0.0, 0.0, 0.0)
    h1 = mol.add_atom("H", 0.6, 0.8, 0.0)
    h2 = mol.add_atom("H", -0.6, 0.8, 0.0)
    mol.add_bond(n0, h1)
    mol.add_bond(n0, h2)
    # ester
    c0 = mol.add_atom("C", 5.0, 0.0, 0.0)  # carbonyl C
    o1 = mol.add_atom("O", 6.0, 0.8, 0.0)  # =O (preserved)
    o2 = mol.add_atom("O", 5.0, -1.3, 0.0)  # ester O (leaves)
    c3 = mol.add_atom("C", 5.0, -2.6, 0.0)  # alkyl C (leaves)
    bo = mol.add_bond(c0, o1)
    mol.set_bond_order(bo, 2.0)  # carbonyl double bond
    mol.add_bond(c0, o2)
    mol.add_bond(o2, c3)
    return mol, {"n0": n0, "c0": c0, "o1": o1, "o2": o2, "c3": c3}


def _bind(reaction, mol):
    """Match each reactant component and merge the map->handle dicts."""
    binding = {}
    for pat in reaction.reactant_patterns:
        matches = pat.find_matches(mol, mapped=True)
        assert matches, f"pattern {pat!r} did not match the fixture"
        binding.update(matches[0])
    return binding


# ---------------------------------------------------------------------------
# ac-001: parse (>> split + multi-component LHS + pairwise maps + agent)
# ---------------------------------------------------------------------------


def test_parse_two_reactant_components_and_product():
    rxn = molrs.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    pats = rxn.reactant_patterns
    assert len(pats) == 2
    assert all(isinstance(p, molrs.SmartsPattern) for p in pats)
    # component 0 is the amine (1 query atom), component 1 the ester (4 atoms)
    assert pats[0].num_query_atoms == 1
    assert pats[1].num_query_atoms == 4


def test_parse_tolerates_agent_field():
    # "A > agent > B" three-part form must not crash; the agent is ignored.
    rxn = molrs.Reaction("[C:1]=[C:2].[S;H1:3] > [Pt] > [C:1][C:2][S:3]")
    assert len(rxn.reactant_patterns) == 2


def test_parse_rejects_missing_arrow():
    with pytest.raises(ValueError):
        molrs.Reaction("[C:1][O:2]")


# ---------------------------------------------------------------------------
# ac-002: forming-bond map pairs + one-sided-map validation
# ---------------------------------------------------------------------------


def test_forming_bonds_amide():
    rxn = molrs.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    assert rxn.forming_bonds == [(1, 2)]


def test_forming_bonds_thiol_ene_is_only_the_new_bond():
    # (1,2) merely changes order (2->1) so it is NOT a forming bond; (2,3) is.
    rxn = molrs.Reaction("[C:1]=[C:2].[S;H1:3] >> [C:1][C:2][S:3]")
    assert rxn.forming_bonds == [(2, 3)]


def test_one_sided_map_is_an_error():
    # :2 appears only on the reactant side -> Daylight pairwise rule violated.
    with pytest.raises(ValueError):
        molrs.Reaction("[C:1][O:2] >> [C:1]")


# ---------------------------------------------------------------------------
# ac-003: single-occurrence apply + topology refresh
# ---------------------------------------------------------------------------


def test_apply_amide_forms_bond_and_drops_leaving_group():
    rxn = molrs.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    mol, h = _amine_plus_ester()
    n_before = mol.n_atoms
    assert not molrs.SmartsPattern("[N][C]=O").has_match(mol)

    binding = _bind(rxn, mol)
    assert set(binding) == {1, 2}
    touched = rxn.apply(mol, binding)

    # apply now reports the touched (surviving) atom handles as a list[int]
    assert isinstance(touched, list)
    assert all(isinstance(t, int) for t in touched)
    # leaving atoms (ester O + alkyl C) removed -> exactly 2 fewer atoms
    assert mol.n_atoms == n_before - 2
    # new N-C(=O) amide linkage present; carbonyl O preserved (not re-added)
    assert molrs.SmartsPattern("[N][C]=O").has_match(mol)
    # topology regenerated around the new bond
    assert mol.n_relations("angles") > 0


def test_apply_reuses_core_and_leaves_binding_atoms_alive():
    rxn = molrs.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    mol, h = _amine_plus_ester()
    binding = _bind(rxn, mol)
    touched = rxn.apply(mol, binding)
    assert isinstance(touched, list)
    # the mapped atoms survive and are now bonded
    n_nbrs = {other for (_, other) in mol.incident_relations(binding[1], "bonds")}
    assert binding[2] in n_nbrs


# ---------------------------------------------------------------------------
# ac-004: unmapped RHS atom is added (element from template, no coordinates)
# ---------------------------------------------------------------------------


def test_apply_adds_unmapped_product_atom():
    # substitution: C-Br -> C-O ; Br leaves, O is a brand-new atom.
    rxn = molrs.Reaction("[C:1]Br >> [C:1]O")
    mol = molrs.Atomistic()
    c0 = mol.add_atom("C", 0.0, 0.0, 0.0)
    br = mol.add_atom("Br", 1.9, 0.0, 0.0)
    mol.add_bond(c0, br)

    binding = _bind(rxn, mol)
    assert set(binding) == {1}
    touched = rxn.apply(mol, binding)
    assert isinstance(touched, list)

    # Br removed, O added -> still 2 atoms
    assert mol.n_atoms == 2
    elements = {mol.get(e, "element") for e in mol.entities()}
    assert elements == {"C", "O"}
    # the new O is bonded to C0 and carries no coordinates
    o_handle = next(e for e in mol.entities() if mol.get(e, "element") == "O")
    c_nbrs = {other for (_, other) in mol.incident_relations(c0, "bonds")}
    assert o_handle in c_nbrs
    assert not mol.has(o_handle, "x")


# ---------------------------------------------------------------------------
# ac-005: no regression of the chain-01 matcher
# ---------------------------------------------------------------------------


def test_smarts_matcher_still_works():
    mol = molrs.Atomistic()
    c = mol.add_atom("C", 0.0, 0.0, 0.0)
    o = mol.add_atom("O", 1.4, 0.0, 0.0)
    mol.add_bond(c, o)
    assert molrs.SmartsPattern("[C:1][O:2]").has_match(mol)


# ---------------------------------------------------------------------------
# region-support-02: apply reports the touched (surviving) atom handles
# ---------------------------------------------------------------------------


def test_apply_touched_amide_endpoints_and_leaving_neighbor():
    """ac-001: touched has the formed-bond endpoints N/C plus the leaving
    group's surviving neighbor; the deleted atoms' own handles are absent; the
    result is deduped."""
    rxn = molrs.Reaction("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O")
    mol, h = _amine_plus_ester()
    binding = _bind(rxn, mol)
    touched = rxn.apply(mol, binding)
    touched_set = set(touched)

    # formed N-C bond endpoints (C is also the ester O's surviving neighbor)
    assert binding[1] in touched_set  # amine N
    assert binding[2] in touched_set  # ester carbonyl C
    # deleted atoms (ester O + alkyl C) own handles must NOT appear
    assert h["o2"] not in touched_set
    assert h["c3"] not in touched_set
    # deduped
    assert len(touched) == len(touched_set)


def test_apply_touched_includes_added_atom():
    """ac-002: a reaction adding an RHS atom includes the new atom's handle;
    the surviving carbon is touched, the deleted Br handle is not."""
    rxn = molrs.Reaction("[C:1]Br >> [C:1]O")
    mol = molrs.Atomistic()
    c0 = mol.add_atom("C", 0.0, 0.0, 0.0)
    br = mol.add_atom("Br", 1.9, 0.0, 0.0)
    mol.add_bond(c0, br)

    binding = _bind(rxn, mol)
    touched = set(rxn.apply(mol, binding))

    o_handle = next(e for e in mol.entities() if mol.get(e, "element") == "O")
    assert o_handle in touched  # newly added atom
    assert c0 in touched  # surviving neighbor of the leaving Br
    assert br not in touched  # deleted atom's own handle absent


def test_apply_touched_thiol_ene_two_carbons_and_sulfur():
    """ac-002: thiol-ene touched = the two carbons (order change) + the sulfur
    (formed bond); the sulfur's spectator H is not touched."""
    rxn = molrs.Reaction("[C:1]=[C:2].[S;H1:3] >> [C:1][C:2][S:3]")
    mol = molrs.Atomistic()
    c1 = mol.add_atom("C", 0.0, 0.0, 0.0)
    c2 = mol.add_atom("C", 1.3, 0.0, 0.0)
    s3 = mol.add_atom("S", 3.0, 0.0, 0.0)
    hs = mol.add_atom("H", 3.0, 1.0, 0.0)  # explicit H so [S;H1] matches
    b = mol.add_bond(c1, c2)
    mol.set_bond_order(b, 2.0)
    mol.add_bond(s3, hs)

    binding = _bind(rxn, mol)
    touched = set(rxn.apply(mol, binding))
    assert touched == {c1, c2, s3}
