"""Línea base: clasificador de clase mayoritaria (DummyClassifier).

Siempre predice la clase más frecuente del entrenamiento de cada fold (Media, ~50 %).
Es el desempeño mínimo que cualquier modelo real debe superar (corrección 6, 1.ª entrega).
Valores esperados en validación cruzada: exactitud ~0,50, macro-F1 ~0,17,
recall de Pobre y de Alta = 0, QWK = 0.

Uso:
    from ses.models.baseline import construir_baseline, evaluar_baseline
    tabla = evaluar_baseline()      # DataFrame: modelo, bloque, fold, metrica, valor

    python -m ses.models.baseline   # imprime media ± desviación
"""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

from ses.evaluation.metrics import SCORERS, a_signo_normal
from ses.pipeline import cargar_entrenamiento, construir_pipeline


def construir_baseline():
    """Modelo de clase mayoritaria (sin parámetros que ajustar)."""
    return DummyClassifier(strategy="most_frequent")

def evaluar_baseline(X=None, y=None, cv=None):
    """Validación cruzada de la línea base, en el formato largo del contrato de datos."""
    if X is None:
        X, y, cv = cargar_entrenamiento()
    # Se pasa por el mismo Pipeline que los demás modelos para comparar en igualdad.
    # El DummyClassifier ignora X, así que el bloque no cambia el resultado.
    pipe = construir_pipeline(construir_baseline(), bloque="C")
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORERS)
    filas = []
    for metrica in SCORERS:
        for fold, valor in enumerate(res[f"test_{metrica}"], start=1):
            filas.append({
                "modelo": "baseline_mayoritaria",
                "bloque": "ninguno",
                "fold": fold,
                "metrica": metrica,
                "valor": a_signo_normal(metrica, valor),
            })
    return pd.DataFrame(filas)

if __name__ == "__main__":
    tabla = evaluar_baseline()
    resumen = tabla.groupby("metrica")["valor"].agg(["mean", "std"]).round(3)
    print("Línea base (clase mayoritaria), validación cruzada 5 folds:")
    print(resumen.to_string())

