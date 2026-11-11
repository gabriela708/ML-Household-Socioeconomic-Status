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
from ses.monitoring.drift import (  # noqa: E402
    detectar_drift, drift_categorico, drift_predicciones, semaforo)

# Modelos finales (Gaby, decisiones 43-45): PPOM entrenado con el 80 % completo, guardado como
# Pipeline (prep + modelo) por scripts/evaluar_test.py.
RUTAS_MODELO = {
    "B": ROOT / "models" / "ppom_AB.joblib",
    "C": ROOT / "models" / "ppom_ABC.joblib",
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

# Streamlit ordena las barras alfabéticamente: se numeran las clases para que salgan en su orden
# ordinal (Pobre -> Alta) y no como Alta, Media, Pobre, Vulnerable.
ETIQUETA_CLASE = {k: f"{k + 1}. {nombre}" for k, nombre in o.CLASES.items()}

# Archivos de ejemplo (los genera scripts/crear_referencia.py)
EJEMPLOS = APP_DIR / "ejemplos"
PLANTILLA = EJEMPLOS / "plantilla.csv"
EJEMPLO_SIN_DRIFT = EJEMPLOS / "hogares_sin_drift.csv"
EJEMPLO_CON_DRIFT = EJEMPLOS / "hogares_con_drift.csv"
REFERENCIA = ROOT / "data" / "reference" / "referencia_train.csv"


@st.cache_data
def cargar_referencia():
    return leer_csv(REFERENCIA)


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
    "Con datos del hogar que no son el ingreso (quiénes viven ahí, cómo es la vivienda y, si se "
    "tienen, su situación de trabajo, salud y productos financieros), el modelo "
    "estima la probabilidad de que un hogar sea **Pobre, Vulnerable, Media o Alta** según las "
    "clases del DANE, y sugiere a cuáles programar primero una **evaluación completa de ingreso**."
)
st.info("⚠️ Herramienta de priorización. **No reemplaza la medición oficial de pobreza** del DANE: "
        "ordena qué hogares evaluar primero cuando todavía no se conoce su ingreso.")

with st.sidebar:
    st.header("Sobre el modelo")
    st.markdown(
        "- **Modelo:** regresión logística ordinal de odds proporcionales parciales (PPOM).\n"
        "- **Escenario por defecto:** A+B+C (demografía, vivienda, trabajo, salud y finanzas); "
        "A+B al apagar el interruptor.\n"
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
        "Tengo datos de trabajo, salud y productos financieros del hogar", value=True,
        help="Encendido: modelo final A+B+C (decisión 44). Apagado: modelo A+B, solo con demografía y "
             "vivienda, para cuando no se tiene esa información (por ejemplo, en una visita corta).")
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
                                   index=pd.Index(list(ETIQUETA_CLASE.values()), name="Clase"))
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
                             value=True, key="toggle_lista")
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
    st.markdown(
        "Un modelo entrenado con datos de **diciembre de 2025 en cabeceras** puede dejar de servir si "
        "los hogares que llegan son distintos (otra zona, otro año, otra forma de recoger los datos). "
        "Aquí se comparan los hogares nuevos con una muestra de **1.000 hogares de entrenamiento**."
    )
    with st.expander("¿Cómo se decide si hay cambio (drift)?"):
        st.markdown(
            "- **Variables numéricas:** prueba de Kolmogorov-Smirnov; hay cambio si el p-valor es menor que 0,05.\n"
            "- **Variables categóricas y sí/no:** suma de las diferencias de proporciones; hay cambio si pasa de 0,20.\n"
            "- **Semáforo:** 🟢 hasta 30 % de variables con cambio y ninguna crítica · 🟡 más de 30 % o una "
            "variable crítica · 🔴 más de 50 % o dos o más variables críticas.\n"
            "- **Variables críticas:** las de mayor efecto en el modelo (estrato, tamaño del hogar, tenencia, "
            "pisos, educación del jefe y ocupados)."
        )

    usar_c_mon = st.toggle("Revisar también las variables de trabajo, salud y finanzas", value=True,
                           key="toggle_mon")
    bloque_mon = "C" if usar_c_mon else "B"
    fuente = st.radio("Hogares a revisar",
                      ["Ejemplo sin cambios", "Ejemplo con cambios simulados", "Subir mi archivo"],
                      horizontal=True)
    nuevos = None
    if fuente == "Ejemplo sin cambios":
        nuevos = leer_csv(EJEMPLO_SIN_DRIFT)
    elif fuente == "Ejemplo con cambios simulados":
        nuevos = leer_csv(EJEMPLO_CON_DRIFT)
        st.caption("Simulación: más hogares de estrato 1 y 2, más arriendo, hogares con un niño más y "
                   "menos conexiones de gas natural (supuestos en scripts/crear_referencia.py).")
    else:
        archivo_mon = st.file_uploader("Archivo CSV de hogares nuevos", type="csv", key="csv_mon")
        if archivo_mon is not None:
            nuevos = leer_csv(archivo_mon)

    errores_mon = p.validar_csv(nuevos, bloque_mon) if nuevos is not None else []
    for e in errores_mon:
        st.error(e)

    if nuevos is not None and not errores_mon:
        referencia = cargar_referencia()
        tabla, tasa = detectar_drift(referencia, nuevos, bloque_mon)
        color, mensaje_sem, accion = semaforo(tabla, tasa)
        {"rojo": st.error, "amarillo": st.warning, "verde": st.success}[color](f"### {mensaje_sem}")
        st.markdown(f"**Acción recomendada:** {accion}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Variables revisadas", len(tabla))
        m2.metric("Variables con cambio", int(tabla["drift"].sum()))
        m3.metric("Críticas con cambio", int((tabla["drift"] & tabla["crítica"]).sum()))

        st.subheader("Detalle por variable")
        st.dataframe(tabla, hide_index=True)

        variable = st.selectbox("Ver cómo cambió una variable", tabla["variable"].tolist())
        if variable.startswith("prop_"):
            st.dataframe(pd.DataFrame({"referencia": referencia[variable].describe(),
                                       "nuevos": nuevos[variable].describe()}).round(3))
        else:
            _, detalle = drift_categorico(referencia[variable], nuevos[variable])
            etiquetas = {v: k for k, v in o.OPCIONES.get(variable, {}).items()}
            # "código · texto" para que las barras salgan en el orden de los códigos
            detalle.index = [f"{int(i)} · {etiquetas[i]}" if i in etiquetas else i for i in detalle.index]
            st.bar_chart(detalle[["referencia", "nuevos"]], stack=False)

        st.subheader("¿Cambiaron también las predicciones?")
        valor_pred, detalle_pred = drift_predicciones(cargar_modelo(bloque_mon), referencia, nuevos,
                                                      bloque_mon, o.CLASES)
        detalle_pred.index = list(ETIQUETA_CLASE.values())
        st.bar_chart(detalle_pred[["referencia", "nuevos"]], stack=False)
        st.caption(f"Suma de diferencias en la proporción de clases predichas: {valor_pred:.2f}. "
                   "El monitoreo detecta cambios en los datos, pero no dice si el modelo acierta: para eso "
                   "se necesitaría el ingreso real de los hogares nuevos.")