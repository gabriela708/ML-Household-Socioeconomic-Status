"""Lógica de la app, sin Streamlit: validar respuestas, armar la fila del modelo y traducir la salida.

Responsable: Juanita. Entrega final.

Se separa de app.py para poder probarla con pytest y desde el cuaderno, sin abrir la app.

Flujo de un hogar:
    respuestas (dict del formulario)
      -> validar()            lista de errores; si no está vacía, no se predice
      -> avisos()             valores raros que dejan seguir, pero con advertencia
      -> construir_fila()     DataFrame de 1 fila con las columnas y códigos de X_hogar
      -> modelo.predict_proba(fila)
      -> mensaje_prioridad()  texto de acción para el usuario
"""
# %%
import pandas as pd

from ses.pipeline import BINARIAS, BLOQUES, NOMINALES
from opciones import CLASES, CODIGOS_POCO_FRECUENTES, OPCIONES, PREGUNTAS, RANGOS

# Umbral de prioridad: 0,5, confirmado por Silva con el análisis de costos (decisión 49)..
UMBRAL_PRIORIDAD = 0.5


# ---------------------------------------------------------------------------
# 1. Validar: errores que impiden predecir
# ---------------------------------------------------------------------------
def validar(resp, bloque="B"):
    """Devuelve la lista de errores del formulario (vacía si todo está bien)."""
    errores = []
    n = resp.get("n_personas") or 0

    if n < 1:
        errores.append("El hogar debe tener al menos 1 persona.")
    if resp.get("n_menores15", 0) + resp.get("n_mayores65", 0) > n:
        errores.append("Los menores de 15 más los mayores de 65 no pueden ser más que "
                       "el total de personas del hogar.")
    if resp.get("n_dormitorios", 0) > resp.get("n_cuartos", 0):
        errores.append("Los cuartos para dormir no pueden ser más que el total de cuartos.")

    if bloque == "C":
        for var in ["n_ocupados", "n_cotizantes", "n_afiliados_salud", "n_subsidiados"]:
            if resp.get(var, 0) > n:
                errores.append(f"«{PREGUNTAS[var]}» no puede ser mayor que el total de personas.")
        if resp.get("n_cotizantes", 0) > resp.get("n_ocupados", 0):
            errores.append("Las personas que cotizan a pensión no pueden ser más que las que trabajan.")
        if resp.get("n_subsidiados", 0) > resp.get("n_afiliados_salud", 0):
            errores.append("Las personas en régimen subsidiado no pueden ser más que las afiliadas a salud.")
        if resp.get("n_ocupados", 0) == 0 and resp.get("posicion_ocup_jefe", 0) != 0:
            errores.append("Si nadie del hogar trabaja, el jefe no puede tener una posición ocupacional.")

    # Coherencia de la cocina (decisión 7): "no preparan alimentos" <-> "no cocinan"
    no_prepara = resp.get("lugar_cocina") == 6
    no_cocina = resp.get("combustible_cocina") == 0
    if no_prepara != no_cocina:
        errores.append("Si no preparan alimentos en el hogar, el combustible debe ser «No cocinan en "
                       "el hogar» (y al revés).")
    return errores


# ---------------------------------------------------------------------------
# 2. Avisos: se puede predecir, pero con menos confianza
# ---------------------------------------------------------------------------
def avisos(resp):
    """Valores fuera de lo visto en entrenamiento o categorías con muy pocos hogares."""
    lista = []
    for var, (_, max_visto, _) in RANGOS.items():
        if var in resp and resp[var] > max_visto:
            lista.append(f"«{PREGUNTAS[var]}» = {resp[var]}: en los datos de entrenamiento el máximo "
                         f"fue {max_visto}. La predicción puede ser menos confiable.")
    for var, codigos in CODIGOS_POCO_FRECUENTES.items():
        if resp.get(var) in codigos:
            texto = next(t for t, c in OPCIONES[var].items() if c == resp[var])
            lista.append(f"«{texto}» aparece en muy pocos hogares de entrenamiento. "
                         "La predicción puede ser menos confiable.")
    return lista


# ---------------------------------------------------------------------------
# 3. Construir la fila que recibe el modelo
# ---------------------------------------------------------------------------
def construir_fila(resp, bloque="B"):
    """DataFrame de 1 fila con exactamente las columnas de BLOQUES[bloque], en ese orden.

    No escala ni hace one-hot: eso lo hace el Pipeline guardado en el .joblib.
    Solo calcula las variables derivadas igual que build_features.py.
    """
    datos = dict(resp)
    n = datos["n_personas"]

    # build_features.py, línea 204
    datos["prop_dependientes"] = (datos["n_menores15"] + datos["n_mayores65"]) / n

    if bloque == "C":
        # build_features.py, líneas 141-149: proporciones sobre las personas del hogar
        datos["prop_afiliados_salud"] = datos["n_afiliados_salud"] / n
        datos["prop_subsidiado"] = datos["n_subsidiados"] / n

    columnas = BLOQUES[bloque]
    faltan = [c for c in columnas if c not in datos]
    if faltan:
        raise KeyError(f"Faltan respuestas para: {faltan}")
    return pd.DataFrame([{c: datos[c] for c in columnas}], columns=columnas)


# ---------------------------------------------------------------------------
# 4. Revisar un CSV con varios hogares (lista de visitas y monitoreo)
# ---------------------------------------------------------------------------
def validar_csv(df, bloque="B"):
    """Errores de un CSV subido: columnas faltantes y códigos que el modelo no conoce."""
    errores = []
    columnas = BLOQUES[bloque]
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        return [f"Faltan columnas en el archivo: {', '.join(faltan)}"]

    for var in [c for c in columnas if c in NOMINALES and c in OPCIONES]:
        validos = set(OPCIONES[var].values())
        malos = sorted(set(df[var].dropna().astype(int)) - validos)
        if malos:
            errores.append(f"{var}: códigos no válidos {malos}")
    for var in [c for c in columnas if c in BINARIAS]:
        if not set(df[var].dropna().unique()) <= {0, 1}:
            errores.append(f"{var}: solo puede tener 0 o 1")
    if (df["n_personas"] < 1).any():
        errores.append("n_personas: hay hogares con menos de 1 persona")
    return errores


# ---------------------------------------------------------------------------
# 5. Traducir la salida del modelo a una acción
# ---------------------------------------------------------------------------
def mensaje_prioridad(probas, umbral=UMBRAL_PRIORIDAD):
    """probas: las 4 probabilidades (Pobre, Vulnerable, Media, Alta) de un hogar.

    Devuelve (nivel, mensaje). La regla es P(Pobre) + P(Vulnerable) >= umbral.
    """
    p_riesgo = probas[0] + probas[1]
    clase = CLASES[int(max(range(4), key=lambda k: probas[k]))]
    if p_riesgo >= umbral:
        return ("alta", f"Prioridad alta: programar evaluación completa de ingreso. "
                        f"Probabilidad de ser Pobre o Vulnerable: {p_riesgo:.0%} "
                        f"(clase más probable: {clase}).")
    if p_riesgo >= umbral / 2:
        return ("media", f"Prioridad media: evaluar si hay cupos disponibles. "
                         f"Probabilidad de ser Pobre o Vulnerable: {p_riesgo:.0%} "
                         f"(clase más probable: {clase}).")
    return ("baja", f"Prioridad baja. Probabilidad de ser Pobre o Vulnerable: {p_riesgo:.0%} "
                    f"(clase más probable: {clase}).")


def priorizar_lote(modelo, df, bloque="B", umbral=UMBRAL_PRIORIDAD):
    """Predice un CSV de hogares y lo ordena de mayor a menor prioridad (lista de visitas)."""
    X = df[BLOQUES[bloque]]
    probas = modelo.predict_proba(X)
    salida = df.copy()
    for k, nombre in CLASES.items():
        salida[f"P_{nombre}"] = probas[:, k].round(3)
    salida["P_Pobre_o_Vulnerable"] = (probas[:, 0] + probas[:, 1]).round(3)
    salida["prioridad"] = [mensaje_prioridad(p, umbral)[0] for p in probas]
    return salida.sort_values("P_Pobre_o_Vulnerable", ascending=False)
# %%
