# Predicción proxy del nivel socioeconómico de hogares urbanos en Colombia (GEIH 2025)

Proyecto integrador — Machine Learning I, Universidad Externado de Colombia.
Equipo: Sara Gabriela Chisaba Cárdenas (Gaby), Juana Sofía Gómez Sánchez (Juanita), Juan Esteban Silva Arévalo (Silva).

Artículo base: Lelisho, Wogi y Tareke (2022). *Ordinal Logistic Regression Analysis in Determining Factors Associated with Socioeconomic Status of Household in Tepi Town, Southwest Ethiopia*. https://doi.org/10.1155/2022/2415692

## Pregunta

¿Qué tan bien puede predecirse la clase socioeconómica de un hogar urbano colombiano (pobre, vulnerable, media, alta, según el ingreso per cápita y los umbrales del DANE) usando **solo información no monetaria** del hogar? Es un enfoque de *proxy means test*: el ingreso se usa únicamente para construir la variable objetivo, nunca como predictor.

## Datos

- GEIH del DANE, diciembre de 2025, módulos en `data/raw/`.
- Población: hogares de cabeceras (`CLASE == 1`). Unidad de observación: el hogar.
- Llave del hogar: `hogar_id = DIRECTORIO_SECUENCIA_P_HOGAR`.

## Estructura

```
config/          parámetros (semilla, rutas, líneas de pobreza, umbrales)
data/raw/        módulos GEIH originales (solo lectura, no se modifican)
data/external/   fuentes externas (pobreza monetaria 2025, líneas de pobreza)
data/interim/    salidas intermedias (no se suben)
data/processed/  y_hogar, X_hogar, índices de partición (no se suben, excepto split_indices.json)
docs/            contrato de datos, matriz de fuga, diccionario, decisiones, informe
src/ses/         código fuente reutilizable (paquete de Python)
notebooks/       análisis por persona (01_ Silva, 02_ Juanita, 03_ Gaby, ...)
scripts/         scripts que regeneran los artefactos de data/processed
reports/         figuras y tablas finales del informe
tests/           pruebas mínimas (llaves, fuga, conjunto de prueba sellado)
app/             despliegue en Streamlit (entrega final)
```

## Reproducir

## Reproducir

**1. Preparar el entorno** (una sola vez)

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**2. Correr el flujo completo, en este orden**

```bash
python scripts/make_target.py
python scripts/make_features.py
python scripts/ajuste_hiperparametros.py
python scripts/run_ablacion.py
python scripts/run_cv.py
pytest
```

| Paso | Qué genera | Responsable |
|---|---|---|
| `make_target.py` | `data/processed/y_hogar.parquet` (clase socioeconómica de cada hogar) | Silva |
| `make_features.py` | `data/processed/X_hogar.parquet` y `split_indices.json` (partición 80/20 y 5 folds) | Juanita |
| `ajuste_hiperparametros.py` | `reports/tables/mejores_C.csv`: el mejor C de la logística por bloque | Juanita |
| `run_ablacion.py` | Ablación A → A+B → A+B+C: tablas en `reports/tables/` y figuras en `reports/figures/` | Juanita |
| `run_cv.py` | Comparación de modelos candidatos | Gaby |
| `pytest` | Pruebas: llaves, fuga y conjunto de prueba sellado | Equipo |

> `ajuste_hiperparametros.py` debe correr **antes** de `run_ablacion.py`, porque la ablación lee `mejores_C.csv`.

## Reglas del equipo

1. Nadie trabaja sobre `main`: una rama por tarea (`silva/dia1-io`, `juanita/dia1-matriz-fuga`, `gaby/dia1-metricas`) y *pull request* revisado por otra persona.
2. Semilla única: `SEED = 42` en `config/config.yaml`.
3. El 20 % de prueba se separa el día 2 y **no se usa en la segunda entrega**: todas las cifras salen de validación cruzada.
4. Ninguna variable monetaria entra como predictor (ver `docs/matriz_fuga.md`).
5. Uso de IA generativa declarado en el informe.
