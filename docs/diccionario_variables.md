# Diccionario de variables (32 predictores de X_hogar.parquet)

Unidad: hogar de cabecera (CLASE = 1) con cobertura en la base de pobreza del DANE (19.690). Llave: `hogar_id` (no es predictor).
Faltantes: hogares de la población con el dato vacío antes del filtro de casos completos.
Los vacíos con significado se recodificaron (ver columna Valores) y no cuentan como faltantes.

| Nombre final | Código GEIH | Módulo | Pregunta | Tipo | Valores / recodificación | Faltantes | Bloque |
|---|---|---|---|---|---|---|---|
| n_personas | P6008 | hogar_vivienda | Total de personas en el hogar | numérico | | 0 | A |
| edad_jefe | P6040 | personas | Edad del jefe de hogar (P6050 = 1) | numérico | años | 0 | A |
| sexo_jefe | P3271 | personas | Sexo del jefe | nominal | 1 hombre, 2 mujer | 0 | A |
| educ_jefe | P3042 | personas | Nivel educativo más alto alcanzado por el jefe | ordinal | 1 ninguno … 13 doctorado. El 9 es un nivel válido (tecnológica), no "no sabe" | 0 | A |
| n_menores15 | P6040 | personas | Personas del hogar con menos de 15 años | numérico | conteo por hogar | 0 | A |
| n_mayores65 | P6040 | personas | Personas del hogar con 65 años o más | numérico | conteo por hogar | 0 | A |
| prop_dependientes | P6040, P6008 | personas | (menores de 15 + mayores de 65) / total de personas | numérico | 0 a 1. Reemplaza la razón de dependencia, que divide entre 0 en hogares sin personas de 15–64 años | 0 | A |
| tipo_vivienda | P4005 | hogar_vivienda | Tipo de vivienda | nominal | códigos DANE (casa, apartamento, cuarto…) | 0 | B |
| material_paredes | P4010 | hogar_vivienda | Material predominante de las paredes exteriores | nominal | 1 ladrillo/bloque … 8 zinc/cartón/desechos. El 9 = sin paredes (válido) | 0 | B |
| material_pisos | P4020 | hogar_vivienda | Material predominante de los pisos | nominal | códigos DANE | 0 | B |
| tiene_gas_natural | P4030S2 | hogar_vivienda | ¿Cuenta con gas natural conectado a red pública? | binario | 1 sí / 0 no | 0 | B |
| tiene_alcantarillado | P4030S3 | hogar_vivienda | ¿Cuenta con alcantarillado? | binario | 1 sí / 0 no | 0 | B |
| tiene_recoleccion_basuras | P4030S4 | hogar_vivienda | ¿Cuenta con recolección de basuras? | binario | 1 sí / 0 no | 0 | B |
| tiene_acueducto | P4030S5 | hogar_vivienda | ¿Cuenta con acueducto? | binario | 1 sí / 0 no | 0 | B |
| estrato | P4030S1A1 | hogar_vivienda | Estrato para tarifa (energía) | nominal | 0 sin estrato, 1–6 estrato, **7 = sin energía** (código creado: la pregunta no se hace si P4030S1 = 2). 9 = no sabe → vacío | 3 | B |
| n_cuartos | P5000 | hogar_vivienda | Número de cuartos de la vivienda | numérico | | 0 | B |
| n_dormitorios | P5010 | hogar_vivienda | Cuartos en que duermen las personas del hogar | numérico | | 0 | B |
| tipo_sanitario | P5020 | hogar_vivienda | Tipo de servicio sanitario | nominal | códigos DANE | 0 | B |
| lugar_cocina | P5070 | hogar_vivienda | Lugar donde preparan los alimentos | nominal | códigos DANE; 6 = no preparan alimentos | 0 | B |
| combustible_cocina | P5080 | hogar_vivienda | Energía o combustible para cocinar | nominal | códigos DANE. **0 = no cocina** (código creado: no se pregunta si P5070 = 6) | 0 | B |
| tenencia_vivienda | P5090 | hogar_vivienda | La vivienda es propia, en arriendo, usufructo… | nominal | códigos DANE | 0 | B |
| fin_corriente_o_cdt | P5222S1, P5222S3 | hogar_vivienda | Productos financieros: cuenta corriente o CDT | binario | 1 si tiene alguno de los dos, 0 si ninguno (decisión 23) | 25 | C |
| fin_cuenta_ahorros | P5222S2 | hogar_vivienda | Productos financieros: cuenta de ahorros | binario | selección múltiple: 1 marcó, vacío → 0 | 25 | C |
| fin_credito_vivienda_vehiculo | P5222S4, P5222S5 | hogar_vivienda | Productos financieros: préstamo de vivienda o de vehículo | binario | 1 si tiene alguno de los dos, 0 si ninguno (decisión 23) | 25 | C |
| fin_prestamo_libre_inversion | P5222S6 | hogar_vivienda | Productos financieros: préstamo de libre inversión | binario | igual | 25 | C |
| fin_tarjeta_credito | P5222S7 | hogar_vivienda | Productos financieros: tarjeta de crédito | binario | igual | 25 | C |
| fin_billetera_movil | P5222S11 | hogar_vivienda | Productos financieros: plataforma móvil (Daviplata, Nequi…) | binario | igual | 25 | C |
| prop_afiliados_salud | P6090 | personas | Proporción de personas del hogar afiliadas a salud | numérico | 0 a 1. P6090: 1 sí → 1, 2 no → 0, 9 no sabe → vacío (no cuenta en el promedio) | 2 | C |
| prop_subsidiado | P6100 | personas | Proporción de personas en régimen subsidiado | numérico | 0 a 1. P6100 = 3 → 1 | 0 | C |
| n_ocupados | (registros en Ocupados) | ocupados | Personas ocupadas del hogar | numérico | conteo. **Vacío → 0** (hogar sin ocupados) | 0 | C |
| n_cotizantes | P6920 | ocupados | Ocupados del hogar que cotizan a pensión | numérico | conteo de P6920 = 1. **Vacío → 0** | 0 | C |
| posicion_ocup_jefe | P6430 | ocupados | Posición ocupacional del jefe | nominal | códigos DANE. **0 = jefe no ocupado** (código creado) | 0 | C |

**Notas**
- `P5222S10` ("no sabe") no es predictor. Los 8 hogares que la marcaron y los 17 que no marcaron ninguna opción quedan con vacío en todas las `fin_*` (los 25 faltantes); esos hogares se excluyen en el filtro de casos completos.
- Las variables `fin_*`, `estrato`, `prop_subsidiado` y `n_cotizantes` son proxies fuertes de capacidad económica: ver justificación en `matriz_fuga.md`.
- Depuración (decisión 23): se eliminaron `n_edad_trabajo` y `fin_ninguno` (combinación exacta de otras variables: colinealidad perfecta) y `tiene_energia` (99,9 % con energía) y `fin_otro` (0,08 %) por ser casi constantes. Cuenta corriente + CDT y préstamo de vivienda + vehículo se unieron por ser muy poco frecuentes por separado. Se aplica después del filtro de casos completos.