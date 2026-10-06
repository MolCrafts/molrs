"""The ``cmap`` category and array-valued params across the FFI seam.

A cmap type names five atom types (two consecutive dihedrals) and carries its
correction table as the array param ``grid``. Array params cross as float64
numpy arrays, compare exactly, survive the ``forcefield`` section and a
``*.mrec`` store, and pickle; a frame's ``cmaps`` block renumbers
``atomi`` … ``atomm``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import molrs
import numpy as np
import pytest

ENDS = ("C", "NH1", "CT1", "C2", "NH2")


def _grid(n: int = 24, scale: float = 0.125) -> np.ndarray:
    return scale * np.arange(n * n, dtype=np.float64).reshape(n, n)


def _cmap_ff(grid: np.ndarray | None = None) -> molrs.ff.ForceField:
    ff = molrs.ff.ForceField("charmm", units="real")
    atoms = ff.def_style("atom", "full")
    ends = [atoms.def_type(name, mass=12.0) for name in ENDS]
    ff.def_style("cmap", "charmm").def_type(
        "-".join(ENDS), *ends, grid=_grid() if grid is None else grid
    )
    return ff


def test_a_cmap_style_takes_five_endpoints_and_a_grid() -> None:
    ff = _cmap_ff()
    style = ff.get_style("cmap", "charmm")
    assert isinstance(style, molrs.ff.CmapStyle)
    (cmap,) = style.types
    assert isinstance(cmap, molrs.ff.CmapType)
    assert cmap.category == "cmap"
    assert [t.name for t in cmap.endpoints] == list(ENDS)
    assert (cmap.itom.name, cmap.mtom.name) == ("C", "NH2")
    grid = cmap["grid"]
    assert isinstance(grid, np.ndarray) and grid.dtype == np.float64
    np.testing.assert_array_equal(grid, _grid())
    assert ff.get_types(molrs.ff.CmapType) == [cmap]
    assert ff.get_styles("cmap") == [style]


def test_an_array_param_is_taken_from_any_numeric_array_or_nested_list() -> None:
    ff = _cmap_ff(grid=[[1, 2], [3, 4]])
    (cmap,) = ff.get_types("cmap")
    np.testing.assert_array_equal(cmap["grid"], [[1.0, 2.0], [3.0, 4.0]])
    assert cmap["grid"].dtype == np.float64
    cmap["grid"] = np.zeros((3, 3), dtype=np.int32)
    assert cmap["grid"].shape == (3, 3) and cmap["grid"].dtype == np.float64
    with pytest.raises(TypeError):
        cmap["grid"] = [[1.0, 2.0], [3.0]]
    # A 0-d array is a number.
    cmap["scale"] = np.float64(0.5)
    assert cmap["scale"] == 0.5


def test_a_restatement_compares_its_arrays_exactly() -> None:
    ff = _cmap_ff()
    style = ff.get_style("cmap", "charmm")
    ends = ff.get_style("atom", "full").types
    name = "-".join(ENDS)
    style.def_type(name, *ends, grid=_grid())
    nudged = _grid()
    nudged[3, 4] = np.nextafter(nudged[3, 4], np.inf)
    with pytest.raises(ValueError):
        style.def_type(name, *ends, grid=nudged)
    with pytest.raises(TypeError):
        style.def_type("short", *ends[:4], grid=_grid())


def test_a_cmap_grid_round_trips_through_the_section_and_a_store(
    tmp_path: Path,
) -> None:
    ff = _cmap_ff()
    section = ff.to_section()
    table = section.table("cmap", "charmm")
    assert list(table["mtom"]) == ["NH2"]
    assert table["grid"].shape == (1, 24, 24)

    path = tmp_path / "ff.mrec"
    molrs.io.write_mrec_forcefield(path, ff)
    back = molrs.ff.ForceField.from_section(molrs.io.read_mrec_forcefield(path))
    (cmap,) = back.get_types("cmap")
    assert cmap["grid"].tobytes() == _grid().tobytes()


def test_a_grid_the_section_cannot_hold_is_refused() -> None:
    with pytest.raises(ValueError, match="cmap grid"):
        _cmap_ff(grid=np.zeros((2, 3))).to_section()


def test_a_cmap_force_field_pickles_with_its_grid() -> None:
    ff = _cmap_ff()
    back = pickle.loads(pickle.dumps(ff, protocol=pickle.HIGHEST_PROTOCOL))
    (cmap,) = back.get_types("cmap")
    assert cmap["grid"].tobytes() == _grid().tobytes()
    assert [t.name for t in cmap.endpoints] == list(ENDS)


def _two_cmaps() -> molrs.Frame:
    return molrs.Frame(
        {
            "atoms": {"x": np.arange(6, dtype=np.float64)},
            "cmaps": {
                key: np.array([i, i + 1], dtype=np.uint64)
                for i, key in enumerate(["atomi", "atomj", "atomk", "atoml", "atomm"])
            }
            | {"type": np.array(["c1", "c2"])},
        }
    )


def test_a_cmaps_block_renumbers_atomi_through_atomm(tmp_path: Path) -> None:
    assert [str(key) for key in molrs.keys.ENDPOINTS][-1] == "atomm"
    assert str(molrs.keys.ATOMM) == "atomm"
    assert molrs.schema.CMAPS == "cmaps"
    assert molrs.schema.relation_endpoints("cmaps", [])[-1] == ("atomm", "atoms")
    frame = _two_cmaps()
    frame.validate()

    out = frame.subset([1, 2, 3, 4, 5])
    assert list(out["cmaps"]["atomm"]) == [4]
    assert list(out["cmaps"]["type"]) == ["c2"]
    two = frame.replicate(2)
    assert list(two["cmaps"]["atomm"]) == [4, 5, 10, 11]

    path = tmp_path / "cmaps.mrec"
    molrs.io.write_mrec(path, frame)
    back = molrs.io.read_mrec(path)
    assert list(back["cmaps"]["atomm"]) == [4, 5]
