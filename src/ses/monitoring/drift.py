"""Detección de data drift: ¿los hogares nuevos se parecen a los de entrenamiento?

Responsable: Juanita. Entrega final.

Adaptado del cuaderno de clase "MLOps: Data Drift, Alertas y Despliegue con Streamlit"
(secciones 11, 16, 17 y 22) a las variables del proyecto:
- numéricas: prueba de Kolmogorov-Smirnov (ks_2samp); drift si p < 0,05;
- categóricas y binarias: suma de diferencias absolutas de proporciones; drift si > 0,20;
- alerta global si más del 30 % de las variables tiene drift;
- semáforo verde / amarillo / rojo y una acción recomendada.

Limitaciones (para el informe):
- El drift dice que los datos cambiaron, no que el modelo se equivoque: para saber eso hace
  falta el ingreso real de los hogares nuevos.
- Con muestras grandes, KS marca como drift diferencias muy pequeñas en variables con pocos
  valores (n_personas, n_cuartos). Por eso se compara con muestras de tamaño parecido (~1.000).
"""

import warnings

import pandas as pd
from scipy.stats import ks_2samp

from ses.pipeline import BINARIAS, BLOQUES, NUMERICAS

ALPHA = 0.05            # KS: drift si p-valor < ALPHA
UMBRAL_CATEGORICO = 0.20
UMBRAL_GLOBAL = 0.30    # alerta si más del 30 % de las variables tiene drift

# Variables críticas: las de mayor efecto en el POM (reports/tables/tabla2_pom_coeficientes.csv)
CRITICAS = ["estrato", "n_personas", "tenencia_vivienda", "material_pisos", "educ_jefe", "n_ocupados"]


def drift_categorico(ref, cur):
    """Suma de diferencias absolutas de proporciones y tabla de detalle (cuaderno, sección 11)."""
    p_ref = ref.value_counts(normalize=True)
    p_cur = cur.value_counts(normalize=True)
    categorias = sorted(set(p_ref.index) | set(p_cur.index))
    p_ref = p_ref.reindex(categorias, fill_value=0)
    p_cur = p_cur.reindex(categorias, fill_value=0)
    detalle = pd.DataFrame({"referencia": p_ref, "nuevos": p_cur, "diferencia": (p_cur - p_ref).abs()})
    return float(detalle["diferencia"].sum()), detalle


def detectar_drift(referencia, nuevos, bloque="B", alpha=ALPHA,
                   umbral_categorico=UMBRAL_CATEGORICO, criticas=CRITICAS):
    """Tabla con una fila por variable del bloque y el % de variables con drift (cuaderno, sección 16)."""
    columnas = BLOQUES[bloque]
    filas = []
    for col in columnas:
        if col in NUMERICAS:
            with warnings.catch_warnings():   # aviso de scipy por valores repetidos; no cambia el resultado
                warnings.simplefilter("ignore")
                _, valor = ks_2samp(referencia[col].dropna(), nuevos[col].dropna())
            tipo, metrica, drift = "numérica", "p-valor KS", valor < alpha
        else:
            valor, _ = drift_categorico(referencia[col], nuevos[col])
            tipo = "binaria" if col in BINARIAS else "categórica"
            metrica, drift = "suma de diferencias", valor > umbral_categorico
        filas.append({
            "variable": col, "tipo": tipo, "métrica": metrica, "valor": round(valor, 4),
            "drift": bool(drift), "crítica": col in criticas,
            "media_referencia": round(referencia[col].mean(), 3) if col in NUMERICAS else None,
            "media_nuevos": round(nuevos[col].mean(), 3) if col in NUMERICAS else None,
        })
    tabla = pd.DataFrame(filas).sort_values(["drift", "crítica"], ascending=False, ignore_index=True)
    return tabla, float(tabla["drift"].mean())


def semaforo(tabla, drift_rate, umbral_global=UMBRAL_GLOBAL):
    """(color, mensaje, acción) para comunicar el estado (cuaderno, secciones 17 y 22)."""
    criticas = tabla[tabla["drift"] & tabla["crítica"]]["variable"].tolist()
    if drift_rate > 0.5 or len(criticas) >= 2:
        return ("rojo", f"🔴 ROJO: {drift_rate:.0%} de las variables cambió"
                        + (f"; variables críticas afectadas: {', '.join(criticas)}" if criticas else ""),
                "Investigar de inmediato la causa (¿otra zona, otro periodo, error de captura?). "
                "Si el cambio es real, reentrenar con datos recientes antes de seguir priorizando.")
    if drift_rate > umbral_global or len(criticas) == 1:
        return ("amarillo", f"🟡 AMARILLO: {drift_rate:.0%} de las variables cambió"
                            + (f"; variable crítica afectada: {criticas[0]}" if criticas else ""),
                "Usar las prioridades con cautela, revisar las variables afectadas y repetir el "
                "monitoreo con el siguiente lote.")
    return ("verde", f"🟢 VERDE: {drift_rate:.0%} de las variables cambió",
            "Operación normal: los hogares nuevos se parecen a los de entrenamiento.")


def drift_predicciones(modelo, referencia, nuevos, bloque, clases):
    """Compara la distribución de clases predichas: si el modelo de pronto predice muchos más
    Pobres, es una señal aunque cada variable por separado parezca estable."""
    pred_ref = pd.Series(modelo.predict(referencia[BLOQUES[bloque]])).map(clases)
    pred_new = pd.Series(modelo.predict(nuevos[BLOQUES[bloque]])).map(clases)
    valor, detalle = drift_categorico(pred_ref, pred_new)
    detalle = detalle.reindex(list(clases.values()), fill_value=0)
    return valor, detalle