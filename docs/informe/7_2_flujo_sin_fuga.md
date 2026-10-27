## 7.2. Flujo reproducible sin fuga de datos

*Responsable: Juanita. *

**Partición.** Se debe tener en cuenta que antes de cualquier transformación, los 19.660 hogares se dividieron en 80 % de entrenamiento (15.728) y 20 % de prueba (3.932) con `StratifiedGroupKFold`, estratificando por clase y agrupando por vivienda (`DIRECTORIO`). La estratificación conserva la proporción de las cuatro clases en cada parte, incluida Alta (4,4 %), y el agrupamiento impide que hogares de una misma vivienda queden a ambos lados de la partición, lo que inflaría el desempeño. Dentro del 80 % se definieron cinco folds de validación cruzada con el mismo criterio; el 20 % de prueba queda sellado hasta la entrega final; la Tabla 1 muestra que la distribución de clases es la misma en todas las partes. La partición se guarda como identificadores de hogar en `split_indices.json`, con semilla 42, de modo que los tres integrantes usan exactamente los mismos folds (decisiones 20–22).



| Parte | Hogares | Pobre | Vulnerable | Media | Alta |
|---|---|---|---|---|---|
| Población | 19.660 | 21,6 | 23,5 | 50,4 | 4,4 |
| Entrenamiento (80 %) | 15.728 | 21,6 | 23,5 | 50,4 | 4,4 |
| Prueba (20 %) | 3.932 | 21,6 | 23,5 | 50,4 | 4,4 |
| Cada fold (5) | 3.145–3.146 | 21,6 | 23,5–23,6 | 50,4 | 4,4–4,5 |

**Tabla 1.** Distribución de las clases (%) en la población, la partición 80/20 y los folds (salida de `make_features.py`).

**Qué se hace antes y qué dentro del flujo.** Antes de partir solo se aplican reglas fijas que no aprenden de los datos: recodificación de códigos de la encuesta y depuración de columnas colineales o casi constantes, que deja 32 predictores (decisión 23). Todo lo que se estima a partir de los datos (medianas, modas, categorías y parámetros de escala) se ajusta dentro de un `Pipeline` de scikit-learn, por lo que se recalcula en cada fold usando solo su parte de entrenamiento. A esto se suma el control de la fuga por variables: ninguna variable monetaria ni componente del ingreso con el que se construye la clase entra como predictor, y la función `assert_no_leakage()` lo verifica en código sobre las columnas finales de X y en las pruebas automáticas (decisión 4, `docs/matriz_fuga.md`).

**Orden de las etapas.** Nosotros aplicamos una ruta, en este orden: selección de las variables del bloque, imputación, codificación one-hot de las nominales, escalamiento de las numéricas y modelo (Tabla 2). El orden garantiza que cada etapa aprenda solo del entrenamiento del fold y transforme la validación con lo aprendido. Tres elementos completan el control de la fuga. Primero, el desbalance se trata con `class_weight="balanced"` dentro del modelo, en lugar de sobremuestrear hogares antes de partir, lo que filtraría copias de un hogar hacia la validación (decisiones 31 y 33). Segundo, una categoría que aparece en validación pero no en el entrenamiento del fold se codifica como ceros, sin romper el flujo (decisión 32). Tercero, el hiperparámetro C de la logística se elige con `GridSearchCV` sobre el mismo `Pipeline` y los mismos folds, nunca con la prueba (decisión 34).


| Etapa | Qué aprende de los datos | De qué datos lo aprende |
|---|---|---|
| Selección del bloque (A, A+B o A+B+C) | Nada (lista fija de columnas) | — |
| Imputación | Mediana (numéricas) y moda (nominales y binarias) | Entrenamiento del fold |
| One-hot (`drop="first"`) | Categorías de cada variable nominal | Entrenamiento del fold |
| Escalamiento | Media y desviación de cada numérica | Entrenamiento del fold |
| Modelo (logística o POM) | Coeficientes y pesos de clase | Entrenamiento del fold |
| Ajuste de C (`GridSearchCV`) | Mejor C por bloque | Los 5 folds del 80 % |
| Evaluación | Nada | Fold de validación |

**Tabla 2.** Etapas del `Pipeline` y origen de lo que aprende cada una.

**Verificación y reproducibilidad.** Para asegurarnos de esto se realizan las pruebas automáticas (`pytest`), que verifican que el escalador aprende solo del entrenamiento, que una categoría nueva no rompe el flujo, que no pasan columnas prohibidas y que el conjunto de prueba no se lee. El flujo completo se regenera con los comandos de la sección *Reproducir* del README, y todos los modelos (línea base, logística y POM) se evalúan con los mismos folds, las mismas métricas y el mismo `Pipeline`, lo que hace comparable cualquier diferencia entre ellos.