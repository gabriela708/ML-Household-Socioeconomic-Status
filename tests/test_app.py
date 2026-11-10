"""Responsable: Juanita. Pruebas de la app de Streamlit (lógica de app/procesar.py y del monitoreo).

No necesitan Streamlit ni los datos procesados: usan hogares inventados. La última prueba usa el
modelo guardado en models/ y se salta si todavía no existe.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import opciones as o  # noqa: E402
import procesar as p  # noqa: E402
from ses.monitoring.drift import detectar_drift, semaforo  # noqa: E402
from ses.pipeline import BINARIAS, BLOQUES, NOMINALES  # noqa: E402

HOGAR = {
    "n_personas": 4, "n_menores15": 2, "n_mayores65": 0, "edad_jefe": 35, "sexo_jefe": 2,
    "educ_jefe": 4, "tipo_vivienda": 1, "material_paredes": 1, "material_pisos": 2, "estrato": 1,
    "n_cuartos": 3, "n_dormitorios": 2, "tipo_sanitario": 1, "lugar_cocina": 1,
    "combustible_cocina": 4, "tenencia_vivienda": 3, "tiene_gas_natural": 0,
    "tiene_alcantarillado": 1, "tiene_recoleccion_basuras": 1, "tiene_acueducto": 1,
    # bloque C
    "posicion_ocup_jefe": 4, "n_ocupados": 2, "n_cotizantes": 1, "n_afiliados_salud": 4,
    "n_subsidiados": 2, "fin_corriente_o_cdt": 0, "fin_cuenta_ahorros": 1,
    "fin_credito_vivienda_vehiculo": 0, "fin_prestamo_libre_inversion": 0,
    "fin_tarjeta_credito": 0, "fin_billetera_movil": 1,
}


def hogares_falsos(n=600, seed=42):
    """Tabla con las 32 columnas, códigos válidos de opciones.py y hogares coherentes."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "n_personas": rng.integers(1, 7, n), "edad_jefe": rng.integers(18, 90, n),
        "educ_jefe": rng.integers(1, 14, n), "n_cuartos": rng.integers(2, 6, n),
    })
    d["n_menores15"] = rng.integers(0, d["n_personas"] + 1)
    d["n_mayores65"] = 0
    d["prop_dependientes"] = d["n_menores15"] / d["n_personas"]
    d["n_dormitorios"] = rng.integers(1, d["n_cuartos"] + 1)
    d["n_ocupados"] = rng.integers(0, d["n_personas"] + 1)
    d["n_cotizantes"] = rng.integers(0, d["n_ocupados"] + 1)
    d["prop_afiliados_salud"] = 1.0
    d["prop_subsidiado"] = rng.random(n).round(2)
    for col in NOMINALES:
        if col in o.OPCIONES:
            d[col] = rng.choice(list(o.OPCIONES[col].values()), n)
    for col in BINARIAS:
        d[col] = rng.integers(0, 2, n)
    return d[BLOQUES["C"]]


# --- validar -----------------------------------------------------------------
def test_hogar_valido_no_tiene_errores():
    assert p.validar(HOGAR, "B") == []
    assert p.validar(HOGAR, "C") == []


def test_mas_menores_que_personas_es_error():
    assert p.validar({**HOGAR, "n_menores15": 5}, "B")


def test_mas_dormitorios_que_cuartos_es_error():
    assert p.validar({**HOGAR, "n_dormitorios": 5}, "B")


def test_cocina_incoherente_es_error():
    # "no preparan alimentos" (6) pero cocinan con gas en cilindro (4)
    assert p.validar({**HOGAR, "lugar_cocina": 6}, "B")


def test_mas_cotizantes_que_ocupados_es_error_en_c():
    assert p.validar({**HOGAR, "n_cotizantes": 3}, "C")


def test_valor_por_fuera_del_entrenamiento_da_aviso():
    assert p.avisos(HOGAR) == []
    assert p.avisos({**HOGAR, "n_personas": 15})


# --- construir_fila: coherencia entre entrenamiento y uso ----------------------
@pytest.mark.parametrize("bloque, n_columnas", [("B", 21), ("C", 32)])
def test_fila_tiene_las_columnas_del_bloque_en_orden(bloque, n_columnas):
    fila = p.construir_fila(HOGAR, bloque)
    assert fila.shape == (1, n_columnas)
    assert list(fila.columns) == BLOQUES[bloque]


def test_proporciones_se_calculan_como_en_build_features():
    fila = p.construir_fila(HOGAR, "C")
    assert fila.loc[0, "prop_dependientes"] == pytest.approx(2 / 4)
    assert fila.loc[0, "prop_afiliados_salud"] == pytest.approx(4 / 4)
    assert fila.loc[0, "prop_subsidiado"] == pytest.approx(2 / 4)


def test_opciones_solo_tienen_codigos_unicos():
    for var, dic in o.OPCIONES.items():
        assert len(set(dic.values())) == len(dic), f"{var} tiene códigos repetidos"


# --- CSV y prioridad -------------------------------------------------------------
def test_csv_valido_y_csv_con_errores():
    d = hogares_falsos()
    assert p.validar_csv(d, "C") == []
    assert p.validar_csv(d.drop(columns="estrato"), "B")            # falta una columna
    malo = d.copy()
    malo.loc[malo.index[0], "estrato"] = 99                          # código que no existe
    assert p.validar_csv(malo, "B")


def test_niveles_de_prioridad():
    assert p.mensaje_prioridad([0.6, 0.2, 0.15, 0.05], umbral=0.5)[0] == "alta"
    assert p.mensaje_prioridad([0.2, 0.1, 0.6, 0.1], umbral=0.5)[0] == "media"
    assert p.mensaje_prioridad([0.05, 0.05, 0.5, 0.4], umbral=0.5)[0] == "baja"


# --- monitoreo de drift ------------------------------------------------------------
def test_mismos_datos_dan_semaforo_verde():
    d = hogares_falsos()
    tabla, tasa = detectar_drift(d, d.copy(), "B")
    assert tasa == 0
    assert semaforo(tabla, tasa)[0] == "verde"


def test_datos_muy_distintos_dan_semaforo_rojo():
    ref = hogares_falsos(seed=1)
    nuevos = hogares_falsos(seed=2)
    nuevos["estrato"] = 1
    nuevos["tenencia_vivienda"] = 3
    nuevos["n_personas"] = nuevos["n_personas"] + 3
    tabla, tasa = detectar_drift(ref, nuevos, "B")
    assert semaforo(tabla, tasa)[0] == "rojo"


# --- el modelo guardado predice igual desde la app ----------------------------------
def test_app_predice_igual_que_el_modelo():
    import joblib
    rutas = [ROOT / "models" / n for n in ("pom_AB.joblib", "provisional_pom_B.joblib")]
    ruta = next((r for r in rutas if r.exists()), None)
    if ruta is None:
        pytest.skip("Todavía no hay modelo guardado en models/")
    modelo = joblib.load(ruta)
    d = hogares_falsos(n=5)
    for i in range(len(d)):
        resp = d.iloc[i].to_dict()
        fila = p.construir_fila(resp, "B")
        directo = modelo.predict_proba(d.iloc[[i]][BLOQUES["B"]])
        assert np.allclose(modelo.predict_proba(fila), directo)
        