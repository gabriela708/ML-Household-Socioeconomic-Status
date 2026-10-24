"""Lista negra de variables monetarias y assert_no_leakage(X).

Responsable: Juanita. Día 1.
"""
EN_TARGET = {
    "INGLABO",               # ingreso laboral
    "P6500",                 # salario
    "P6750",                 # ganancia del trabajo independiente
    "P7500S2", "P7500S2A1",  # recibió pensión / monto
    "P5130",                 # arriendo imputado (confirmar con target.py)
}
MONETARIAS = {
    "P5100",  # cuota mensual de la vivienda
    "P5110",  # cuota de amortización
    "P5140",  # arriendo pagado
}
IDENTIFICADORES = {"DIRECTORIO", "SECUENCIA_P", "ORDEN", "HOGAR", "FEX_C18", "PERIODO", "MES"}


PROHIBIDAS = EN_TARGET | MONETARIAS | IDENTIFICADORES

def assert_no_leakage(columns):
    malas = {c for c in columns if set(c.split("__")) & PROHIBIDAS}
    assert not malas, f"Variables prohibidas en X: {malas}"

#PRUEBAS 
if __name__ == "__main__":
    # Prueba 1: columnas permitidas, no debe fallar
    assert_no_leakage(["P5000", "P4010", "edad_jefe"])
    print("Prueba 1 OK: sin fuga")

    # Prueba 2: una columna prohibida con prefijo del Pipeline, debe fallar
    try:
        assert_no_leakage(["P5000", "num__P5140"])
        print("Prueba 2 FALLÓ: el guardia no detectó la fuga")
    except AssertionError as e:
        print("Prueba 2 OK: detectó la fuga ->", e)