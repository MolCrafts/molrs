import pytest

import molrs


def test_registry_parses_and_converts_molecular_units():
    units = molrs.UnitRegistry()
    energy = units.quantity(1.0, "kilocalorie_per_mole")
    assert energy.to("kilojoule_per_mole").magnitude == pytest.approx(4.184)
    assert energy.to("eV").magnitude == pytest.approx(0.0433641, rel=1e-5)


def test_spelled_out_si_prefixes_are_native():
    units = molrs.UnitRegistry()
    assert units.parse("nanometer").factor_to(units.angstrom) == pytest.approx(10.0)
    assert units.parse("femtosecond").factor_to(units.second) == pytest.approx(1e-15)


def test_quantity_arithmetic_and_dimension_errors():
    units = molrs.UnitRegistry()
    total = 1.0 * units.nanometer + 5.0 * units.angstrom
    assert total.magnitude == pytest.approx(1.5)
    assert total.units == units.nanometer

    speed = (2.0 * units.nanometer) / (4.0 * units.picosecond)
    assert speed.to("meter / second").magnitude == pytest.approx(500.0)

    with pytest.raises(molrs.UnitsError, match="dimension mismatch"):
        (1.0 * units.meter).to("second")


def test_custom_definition_is_registry_local():
    units = molrs.UnitRegistry()
    units.define("smoot", 1.7018, units.meter.dimension)
    assert (1.0 * units.smoot).to(units.meter).magnitude == pytest.approx(1.7018)
    with pytest.raises(AttributeError):
        getattr(molrs.UnitRegistry(), "smoot")


def test_affine_temperature_conversion():
    units = molrs.UnitRegistry()
    assert units.quantity(25.0, "degC").to("K").magnitude == pytest.approx(298.15)
