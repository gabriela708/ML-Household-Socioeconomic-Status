# Contrato de datos (firmado el día 1)

Este archivo permite trabajar en paralelo. Cambiarlo requiere PR aprobado por las tres personas.

## Llave
`hogar_id = f"{DIRECTORIO}_{SECUENCIA_P}_{HOGAR}"` (texto).

## `data/processed/y_hogar.parquet` — responsable: Silva
| columna | tipo | descripción |
|---|---|---|
| hogar_id | str | llave |
| DIRECTORIO | int | para agrupar particiones |
| ses | int (0–3) | 0 Pobre, 1 Vulnerable, 2 Media, 3 Alta |
| ypc | float | ingreso per cápita (solo análisis, **nunca predictor**) |

## `data/processed/X_hogar.parquet` — responsable: Juanita
| columna | tipo | descripción |
|---|---|---|
| hogar_id | str | llave |
| (predictores) | numérico / nominal / ordinal | nombres en snake_case, tipo en `diccionario_variables.md` |

## `data/processed/split_indices.json` — responsable: Juanita
`{"seed": 42, "train_idx": [...hogar_id...], "test_idx": [...hogar_id...], "folds": [[...], ...]}`

## `data/processed/cv_resultados.csv` — responsable: Gaby
`modelo, bloque, fold, metrica, valor`

Firmas: Silva ☐ · Juanita ☐ · Gaby ☐
