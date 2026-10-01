import molrs
import pytest


def _forcefield():
    ff = molrs.ff.ForceField()
    atom_style = ff.def_style("atom", "full")
    cr = atom_style.def_type("CR", type_=1.0)
    b = atom_style.def_type("B", type_=2.0)
    pair_style = ff.def_style("pair", "lj/cut")
    pair_style.def_type("CR-B", cr, b, epsilon=1.0, sigma=3.5)
    pair_style.def_type("CR", cr, epsilon=2.0, sigma=3.6)
    return ff


def _epsilon(ff, name):
    return ff.get_style("pair", "lj/cut").get_type_by_name(name)["epsilon"]


def test_native_scale_lj_clones_and_scales_cross_pair():
    ff = _forcefield()
    fragments = {
        "c2c1im": (["CR"], [(0.0, 0.0, 0.0)], [12.0]),
        "bf4": (["B"], [(4.0, 0.0, 0.0)], [11.0]),
    }
    output = molrs.ff.scale_lj(ff, fragments)
    expected = molrs.ff.compute_k_ij(
        molrs.ff.fragment_scaling_data()["c2c1im"],
        molrs.ff.fragment_scaling_data()["bf4"],
        4.0,
    )
    assert isinstance(output, molrs.ff.ForceField)
    assert _epsilon(output, "CR-B") == pytest.approx(expected)
    assert _epsilon(output, "CR") == 2.0
    assert _epsilon(ff, "CR-B") == 1.0


def test_native_scale_lj_missing_data_is_key_error():
    fragments = {"missing": (["CR"], [(0.0, 0.0, 0.0)], [12.0])}
    with pytest.raises(KeyError, match="no scaling data"):
        molrs.ff.scale_lj(_forcefield(), fragments)
