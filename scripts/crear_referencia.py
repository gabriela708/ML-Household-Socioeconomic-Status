"""Datos de referencia y de demostración para la app (lista de visitas y monitoreo de drift).

Responsable: Juanita. Entrega final.

Uso (desde la raíz del repositorio):
    python scripts/crear_referencia.py

Genera, usando SOLO el 80 % de entrenamiento (el 20 % de prueba no se toca, decisión 7):
    data/reference/referencia_train.csv   1.000 hogares: "cómo eran los datos al entrenar"
    app/ejemplos/hogares_sin_drift.csv    1.000 hogares distintos, del mismo entrenamiento
    app/ejemplos/hogares_con_drift.csv    los mismos 1.000, con cambios simulados
    app/ejemplos/plantilla.csv            solo los encabezados, para que el usuario sepa qué subir

Drift simulado ("llegó población de otra zona", supuestos explícitos):
    - 40 % de los hogares de estrato 3 o 4 pasan a estrato 1 o 2;
    - 30 % de los hogares con vivienda propia pasan a arriendo;
    - 30 % de los hogares ganan una persona menor de 15 años (máximo 13 personas);
    - 20 % de los hogares con gas natural pierden la conexión y cocinan con gas en cilindro.
"""

import numpy as np
import pandas as pd

from ses.config import CFG, ROOT
from ses.pipeline import BLOQUES, cargar_entrenamiento

SEED = CFG["seed"]
N = 1000
COLUMNAS = BLOQUES["C"]  # las 32: sirven para el modelo B y el C


def simular_drift(df, seed=SEED):
    """Copia de df con los cambios descritos arriba, manteniendo hogares coherentes."""
    rng = np.random.default_rng(seed)
    d = df.copy()

    est = d["estrato"].isin([3, 4]) & (rng.random(len(d)) < 0.40)
    d.loc[est, "estrato"] = rng.choice([1, 2], size=est.sum())

    propia = d["tenencia_vivienda"].isin([1, 2]) & (rng.random(len(d)) < 0.30)
    d.loc[propia, "tenencia_vivienda"] = 3

    crece = (d["n_personas"] < 13) & (rng.random(len(d)) < 0.30)
    d.loc[crece, "n_personas"] += 1
    d.loc[crece, "n_menores15"] += 1
    # las proporciones se recalculan con el nuevo tamaño del hogar
    d["prop_dependientes"] = (d["n_menores15"] + d["n_mayores65"]) / d["n_personas"]
    for col in ["prop_afiliados_salud", "prop_subsidiado"]:
        d.loc[crece, col] = df.loc[crece, col] * df.loc[crece, "n_personas"] / d.loc[crece, "n_personas"]

    gas = (d["tiene_gas_natural"] == 1) & (rng.random(len(d)) < 0.20)
    d.loc[gas, "tiene_gas_natural"] = 0
    d.loc[gas & (d["combustible_cocina"] == 3), "combustible_cocina"] = 4
    return d


def main():
    X, _, _ = cargar_entrenamiento()
    X = X[COLUMNAS]

    muestra = X.sample(2 * N, random_state=SEED)
    referencia, sin_drift = muestra.iloc[:N], muestra.iloc[N:]
    con_drift = simular_drift(sin_drift)

    carpeta_ref = ROOT / "data" / "reference"
    carpeta_ej = ROOT / "app" / "ejemplos"
    carpeta_ref.mkdir(parents=True, exist_ok=True)
    carpeta_ej.mkdir(parents=True, exist_ok=True)

    referencia.to_csv(carpeta_ref / "referencia_train.csv")
    sin_drift.to_csv(carpeta_ej / "hogares_sin_drift.csv")
    con_drift.to_csv(carpeta_ej / "hogares_con_drift.csv")
    pd.DataFrame(columns=["hogar_id"] + COLUMNAS).to_csv(carpeta_ej / "plantilla.csv", index=False)

    print(f"Referencia: {len(referencia)} hogares -> {carpeta_ref / 'referencia_train.csv'}")
    print(f"Ejemplos sin y con drift: {len(sin_drift)} hogares cada uno -> {carpeta_ej}")


if __name__ == "__main__":
    main()
    