"""Responsable: Silva. La llave de hogar debe ser única."""
import pytest

@pytest.mark.skip(reason="activar cuando io.py esté listo")
def test_hogar_id_unico():
    from ses.io import read_module, make_hogar_id
    h = read_module("hogar_vivienda")
    assert make_hogar_id(h).is_unique
