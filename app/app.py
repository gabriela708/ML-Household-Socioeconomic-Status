"""App de Streamlit: priorizar hogares para evaluación de ingreso.

Responsable: Juanita. Entrega final.

Correr desde la raíz del repositorio:
    streamlit run app/app.py

Este archivo solo tiene la interfaz. Los textos y códigos están en opciones.py y la lógica
(validar, armar la fila, traducir la salida) en procesar.py.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Rutas: la app funciona desde cualquier carpeta y aunque no se haya hecho `pip install -e .`
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for ruta in (APP_DIR, ROOT / "src"):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

import opciones as o  # noqa: E402
import procesar as p  # noqa: E402

# Modelos. Provisionales mientras llegan los finales de Gaby (models/pom_AB.joblib, pom_ABC.joblib):
# cuando lleguen, solo se cambian estas dos rutas.
RUTAS_MODELO = {
    "B": ROOT / "models" / "provisional_pom_B.joblib",
    "C": ROOT / "models" / "provisional_pom_C.joblib",
}

# Valores con los que arranca el formulario: un hogar típico del entrenamiento (medianas y modas)
VALORES_INICIALES = {
    "n_personas": 3, "n_menores15": 0, "n_mayores65": 0, "edad_jefe": 49,
    "sexo_jefe": 2, "educ_jefe": 5,
    "tipo_vivienda": 2, "material_paredes": 1, "material_pisos": 4, "estrato": 2,
    "n_cuartos": 3, "n_dormitorios": 2, "tipo_sanitario": 1, "lugar_cocina": 1,
    "combustible_cocina": 3, "tenencia_vivienda": 3,
    "tiene_acueducto": 1, "tiene_alcantarillado": 1, "tiene_gas_natural": 1,
    "tiene_recoleccion_basuras": 1,
    "posicion_ocup_jefe": 1, "n_ocupados": 1, "n_cotizantes": 1,
    "n_afiliados_salud": 3, "n_subsidiados": 0,
    "fin_cuenta_ahorros": 1, "fin_billetera_movil": 1, "fin_tarjeta_credito": 0,
    "fin_prestamo_libre_inversion": 0, "fin_credito_vivienda_vehiculo": 0,
    "fin_corriente_o_cdt": 0,
}

COLOR_NIVEL = {"alta": st.error, "media": st.warning, "baja": st.success}

# Archivos de ejemplo (los genera scripts/crear_referencia.py)
EJEMPLOS = APP_DIR / "ejemplos"
PLANTILLA = EJEMPLOS / "plantilla.csv"
EJEMPLO_SIN_DRIFT = EJEMPLOS / "hogares_sin_drift.csv"
EJEMPLO_CON_DRIFT = EJEMPLOS / "hogares_con_drift.csv"
REFERENCIA = ROOT / "data" / "reference" / "referencia_train.csv"


def leer_csv(fuente):
    """Lee un CSV de hogares; si trae hogar_id, lo usa como índice para identificar cada fila."""
    df = pd.read_csv(fuente)
    return df.set_index("hogar_id") if "hogar_id" in df.columns else df


# ---------------------------------------------------------------------------
# Carga del modelo (una sola vez, no en cada clic)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo(bloque):
    return joblib.load(RUTAS_MODELO[bloque])


def campo(var):
    """Dibuja la pregunta de una variable y devuelve el código que recibe el modelo."""
    inicial = VALORES_INICIALES[var]
    if var in o.OPCIONES:                                   # lista desplegable
        textos = list(o.OPCIONES[var].keys())
        codigos = list(o.OPCIONES[var].values())
        elegido = st.selectbox(o.PREGUNTAS[var], textos, index=codigos.index(inicial), key=var)
        return o.OPCIONES[var][elegido]
    if var in o.RANGOS:                                     # número entero
        minimo, _, maximo = o.RANGOS[var]
        return int(st.number_input(o.PREGUNTAS[var], min_value=minimo, max_value=maximo,
                                   value=inicial, step=1, key=var))
    return int(st.checkbox(o.PREGUNTAS[var], value=bool(inicial), key=var))  # sí / no


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Priorización de hogares", page_icon="🏠", layout="wide")
st.title("🏠 ¿A qué hogar visitar primero?")
st.markdown(
    "Con datos fáciles de observar (quiénes viven en el hogar y cómo es la vivienda), el modelo "
    "estima la probabilidad de que un hogar sea **Pobre, Vulnerable, Media o Alta** según las "
    "clases del DANE, y sugiere a cuáles programar primero una **evaluación completa de ingreso**."
)
st.info("⚠️ Herramienta de priorización. **No reemplaza la medición oficial de pobreza** del DANE: "
        "ordena qué hogares evaluar primero cuando todavía no se conoce su ingreso.")

with st.sidebar:
    st.header("Sobre el modelo")
    st.markdown(
        "- **Modelo:** regresión logística ordinal (POM) con pesos balanceados.\n"
        "- **Datos:** GEIH diciembre 2025, 15.728 hogares de cabeceras (80 % de entrenamiento).\n"
        f"- **Regla de prioridad:** P(Pobre) + P(Vulnerable) ≥ {p.UMBRAL_PRIORIDAD:.0%}.\n"
        "- **No usa el ingreso** ni ninguna variable monetaria."
    )
    st.caption("Proyecto integrador Machine Learning I · Universidad Externado de Colombia")

tab_hogar, tab_lista, tab_monitoreo = st.tabs(
    ["🏠 Priorizar un hogar", "📋 Lista de visitas (CSV)", "📈 Monitoreo del modelo"])

# ---------------------------------------------------------------------------
# Pestaña 1: un hogar
# ---------------------------------------------------------------------------
with tab_hogar:
    usar_c = st.toggle(
        "Tengo datos de trabajo, salud y productos financieros del hogar",
        help="Activa el modelo A+B+C. Es más preciso, pero usa información más cercana al ingreso "
             "y más difícil de verificar en una visita corta.")
    bloque = "C" if usar_c else "B"

    with st.form("formulario_hogar"):
        col_a, col_b = st.columns(2)
        resp = {}
        with col_a:
            st.subheader("👨‍👩‍👧 Quiénes viven aquí")
            for var in o.GRUPO_A:
                resp[var] = campo(var)
        with col_b:
            st.subheader("🏘️ Cómo es la vivienda")
            for var in o.GRUPO_B:
                resp[var] = campo(var)
            st.markdown("**Servicios públicos con los que cuenta**")
            for var in o.SERVICIOS:
                resp[var] = campo(var)

        if usar_c:
            st.subheader("💼 Trabajo, salud y productos financieros")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                for var in o.GRUPO_C:
                    resp[var] = campo(var)
            with col_c2:
                st.markdown("**Productos financieros que tiene alguien del hogar**")
                for var in o.FINANCIEROS:
                    resp[var] = campo(var)

        enviado = st.form_submit_button("Calcular prioridad", type="primary")

    errores = p.validar(resp, bloque) if enviado else []
    for e in errores:
        st.error(e)

    # Solo se predice si se envió el formulario y no hay errores. No se usa st.stop() para que
    # las otras pestañas sigan funcionando.
    if enviado and not errores:
        for a in p.avisos(resp):
            st.warning(a)

        fila = p.construir_fila(resp, bloque)
        probas = cargar_modelo(bloque).predict_proba(fila)[0]
        nivel, mensaje = p.mensaje_prioridad(probas)

        st.divider()
        COLOR_NIVEL[nivel](f"### {mensaje}")

        col_g, col_t = st.columns([2, 1])
        with col_g:
            grafico = pd.DataFrame({"Probabilidad": probas},
                                   index=pd.Index(list(o.CLASES.values()), name="Clase"))
            st.bar_chart(grafico, y="Probabilidad", horizontal=True)
        with col_t:
            for k, nombre in o.CLASES.items():
                st.metric(nombre, f"{probas[k]:.0%}")

        with st.expander("Ver los datos exactos que recibió el modelo"):
            st.caption("Mismas columnas y códigos que los datos de entrenamiento (X_hogar). "
                       "La imputación, la codificación y el escalamiento los hace el Pipeline guardado.")
            st.dataframe(fila, hide_index=True)

# ---------------------------------------------------------------------------
# Pestañas 2 y 3: siguientes pasos
# ---------------------------------------------------------------------------
with tab_lista:
    st.markdown(
        "Sube un archivo CSV con varios hogares (una fila por hogar) y descarga la **lista de visitas "
        "ordenada de mayor a menor prioridad**. El archivo debe tener las mismas columnas y códigos "
        "que la plantilla."
    )
    usar_c_lista = st.toggle("El archivo trae datos de trabajo, salud y productos financieros",
                             key="toggle_lista")
    bloque_lista = "C" if usar_c_lista else "B"

    col_p, col_e = st.columns(2)
    with col_p:
        st.download_button("⬇️ Descargar plantilla CSV", data=PLANTILLA.read_bytes(),
                           file_name="plantilla_hogares.csv", mime="text/csv")
    with col_e:
        usar_ejemplo = st.checkbox("Usar el archivo de ejemplo (1.000 hogares de entrenamiento)")

    archivo = st.file_uploader("Archivo CSV de hogares", type="csv", key="csv_lista")
    errores_csv = []
    if archivo is not None or usar_ejemplo:
        hogares = leer_csv(archivo if archivo is not None else EJEMPLO_SIN_DRIFT)
        errores_csv = p.validar_csv(hogares, bloque_lista)
        for e in errores_csv:
            st.error(e)

    if (archivo is not None or usar_ejemplo) and not errores_csv:
        lista = p.priorizar_lote(cargar_modelo(bloque_lista), hogares, bloque_lista)
        conteo = lista["prioridad"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Prioridad alta", int(conteo.get("alta", 0)))
        c2.metric("🟡 Prioridad media", int(conteo.get("media", 0)))
        c3.metric("🟢 Prioridad baja", int(conteo.get("baja", 0)))

        columnas_ver = ["prioridad", "P_Pobre_o_Vulnerable"] + [f"P_{c}" for c in o.CLASES.values()]
        st.dataframe(lista[columnas_ver])
        st.download_button("⬇️ Descargar lista de visitas", data=lista.to_csv().encode("utf-8"),
                           file_name="lista_de_visitas.csv", mime="text/csv", type="primary")

with tab_monitoreo:
    st.info("En construcción: comparar hogares nuevos con los de entrenamiento (data drift).")