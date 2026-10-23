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

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/make_target.py
python scripts/make_features.py
python scripts/run_cv.py
pytest
```

## Reglas del equipo

1. Nadie trabaja sobre `main`: una rama por tarea (`silva/dia1-io`, `juanita/dia1-matriz-fuga`, `gaby/dia1-metricas`) y *pull request* revisado por otra persona.
2. Semilla única: `SEED = 42` en `config/config.yaml`.
3. El 20 % de prueba se separa el día 2 y **no se usa en la segunda entrega**: todas las cifras salen de validación cruzada.
4. Ninguna variable monetaria entra como predictor (ver `docs/matriz_fuga.md`).
5. Uso de IA generativa declarado en el informe.
