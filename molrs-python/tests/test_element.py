import pytest

import molrs


@pytest.mark.parametrize(
    ("identifier", "number", "name", "symbol"),
    [
        ("h", 1, "Hydrogen", "H"),
        ("CARBON", 6, "Carbon", "C"),
        (8, 8, "Oxygen", "O"),
        ("oganesson", 118, "Oganesson", "Og"),
    ],
)
def test_element_lookup(identifier, number, name, symbol):
    element = molrs.Element(identifier)
    assert element.number == number
    assert element.name == name
    assert element.symbol == symbol
    assert repr(element) == f"<Element {symbol}>"


def test_all_real_elements_round_trip_through_rust_table():
    for number in range(1, 119):
        element = molrs.Element(number)
        assert molrs.Element(element.symbol) == element
        assert molrs.Element(element.name.upper()) == element
        assert element.mass > 0.0
        assert element.vdw > 0.0
        assert element.covalent > 0.0


def test_element_convenience_lookups():
    assert molrs.Element.get_symbols([1, "carbon", "o", 7]) == ["H", "C", "O", "N"]
    assert molrs.Element.get_atomic_number("fe") == 26


@pytest.mark.parametrize(
    "identifier", [-1, 0, 119, 999, "X", "unknown", "not-an-element"]
)
def test_invalid_element_fails_fast(identifier):
    with pytest.raises(KeyError, match="Element not found"):
        molrs.Element(identifier)
