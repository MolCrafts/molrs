import pytest

import molrs


def _forcefield():
    ff = molrs.ForceField()
    ff.def_atomstyle("full").def_type("CR", type_=1.0)
    ff.def_atomstyle("full").def_type("B", type_=2.0)
    pairs = ff.def_pairstyle("lj/cut")
    cr = ff.def_atomstyle("full").types[0]
    b = ff.def_atomstyle("full").types[1]
    pairs.def_type(cr, b, epsilon=1.0, sigma=3.5)
    pairs.def_type(cr, cr, epsilon=2.0, sigma=3.6)
    return ff


def _epsilon(ff, name):
    return dict(ff.types("pair", "lj/cut"))[name]["epsilon"]


def test_native_scale_lj_clones_and_scales_cross_pair():
    ff = _forcefield()
    fragments = {
        "c2c1im": (["CR"], [(0.0, 0.0, 0.0)], [12.0]),
        "bf4": (["B"], [(4.0, 0.0, 0.0)], [11.0]),
    }
    output = molrs.scale_lj(ff, fragments)
    expected = molrs.compute_k_ij(
        molrs.fragment_scaling_data()["c2c1im"],
        molrs.fragment_scaling_data()["bf4"],
        4.0,
    )
    assert isinstance(output, molrs.ForceField)
    assert _epsilon(output, "CR-B") == pytest.approx(expected)
    assert _epsilon(output, "CR") == 2.0
    assert _epsilon(ff, "CR-B") == 1.0


def test_native_scale_lj_missing_data_is_key_error():
    fragments = {"missing": (["CR"], [(0.0, 0.0, 0.0)], [12.0])}
    with pytest.raises(KeyError, match="no scaling data"):
        molrs.scale_lj(_forcefield(), fragments)
