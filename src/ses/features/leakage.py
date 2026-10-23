"""Lista negra de variables monetarias y assert_no_leakage(X).

Responsable: Juanita. Día 1.
"""

PROHIBIDAS = {"INGLABO", "P6500", "P5100", "P5130", "P5140"}  # completar con la matriz de fuga


def assert_no_leakage(columns):
    usadas = {c.split("__")[0] for c in columns}
    malas = usadas & PROHIBIDAS
    assert not malas, f"Variables prohibidas en X: {malas}"
