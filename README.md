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
python scripts/run_candidatos_nuevos.py lda qda
python scripts/run_ppom.py
python scripts/crear_referencia.py
pytest
```

| Paso | Qué genera | Responsable |
|---|---|---|
| `make_target.py` | `data/processed/y_hogar.parquet` (clase socioeconómica de cada hogar) | Silva |
| `make_features.py` | `data/processed/X_hogar.parquet` y `split_indices.json` (partición 80/20 y 5 folds) | Juanita |
| `ajuste_hiperparametros.py` | `reports/tables/mejores_C.csv`: el mejor C de la logística por bloque | Juanita |
| `run_ablacion.py` | Ablación A → A+B → A+B+C: tablas en `reports/tables/` y figuras en `reports/figures/` | Juanita |
| `run_cv.py` | Comparación de modelos candidatos | Gaby |
| `run_candidatos_nuevos.py` | Agrega LDA y QDA (o cualquier candidato nuevo) a las tablas de `run_cv.py` sin volver a correr los demás | Gaby |
| `run_ppom.py` | PPOM en A+B y A+B+C: agrega sus filas a las tablas de `run_cv.py` y de la ablación, diagnóstico por fold y tablas 5–6 (POM vs. PPOM, prueba LR y AIC) | Gaby |
| `crear_referencia.py` | `data/reference/referencia_train.csv` y los CSV de ejemplo de `app/ejemplos/` (solo con el 80 % de entrenamiento) | Juanita |
| `pytest` | Pruebas: llaves, fuga, conjunto de prueba sellado y lógica de la app | Equipo |

> `ajuste_hiperparametros.py` debe correr **antes** de `run_ablacion.py`, porque la ablación lee `mejores_C.csv`.
> `run_ppom.py` debe correr **después** de `run_ablacion.py` y `run_cv.py`, porque agrega filas a sus tablas (si se vuelve a correr `run_ablacion.py`, se borran las filas del PPOM de la ablación y hay que correr `run_ppom.py` de nuevo).


## App de Streamlit

La app **prioriza qué hogares visitar primero** para una evaluación completa de ingreso. No reemplaza la medición oficial de pobreza del DANE.

**Correr la app** (desde la raíz del repositorio, después de preparar el entorno):

```bash
python -m streamlit run app/app.py
```

Se abre en el navegador (`http://localhost:8501`). Necesita los modelos en `models/` y los archivos que genera `scripts/crear_referencia.py`.

| Pestaña | Qué hace |
|---|---|
| 🏠 Priorizar un hogar | Formulario en lenguaje normal → probabilidad de cada clase y prioridad (alta, media o baja). Por defecto usa el modelo final A+B+C (decisión 44); al apagar el interruptor usa el A+B (solo demografía y vivienda) |
| 📋 Lista de visitas | Sube un CSV con varios hogares y descarga la lista ordenada por prioridad. Plantilla en `app/ejemplos/plantilla.csv` |
| 📈 Monitoreo | Compara hogares nuevos con 1.000 de entrenamiento (prueba KS y diferencia de proporciones) y muestra un semáforo de drift con la acción recomendada |

| Archivo | Qué tiene |
|---|---|
| `app/app.py` | Solo la interfaz |
| `app/opciones.py` | Textos del formulario y códigos de cada variable (diccionario GEIH 2025 del DANE) |
| `app/procesar.py` | Validaciones, construcción de la fila que recibe el modelo y regla de prioridad |
| `src/ses/monitoring/drift.py` | Detección de drift y semáforo (adaptado del cuaderno de clase de MLOps) |
| `tests/test_app.py` | Pruebas de la app (`pytest`) |

**Coherencia entre entrenamiento y uso:** la app arma una fila con las mismas columnas y códigos que `X_hogar`; la imputación, la codificación y el escalamiento los hace el `Pipeline` guardado en el `.joblib`. Solo se ofrecen las categorías vistas en entrenamiento (decisión 47).

## Reglas del equipo

1. Nadie trabaja sobre `main`: una rama por tarea (`silva/dia1-io`, `juanita/dia1-matriz-fuga`, `gaby/dia1-metricas`) y *pull request* revisado por otra persona.
2. Semilla única: `SEED = 42` en `config/config.yaml`.
3. El 20 % de prueba se separa el día 2 y **se usa una sola vez**: todas las cifras salen de validación cruzada.
4. Ninguna variable monetaria entra como predictor (ver `docs/matriz_fuga.md`).
5. Uso de IA generativa declarado en el informe.