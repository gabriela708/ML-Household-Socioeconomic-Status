"""Responsable: Juanita."""
import pytest
from ses.features.leakage import assert_no_leakage

def test_detecta_variable_prohibida():
    with pytest.raises(AssertionError):
        assert_no_leakage(["edad_jefe", "INGLABO"])
