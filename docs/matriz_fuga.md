# Matriz de fuga

Regla de admisión: una variable entra como predictor solo si
1. no es un monto en pesos,
2. no se usa en la fórmula del target,
3. puede observarse sin preguntar por el ingreso.

Bloques: **A** (paper: jefe y tamaño del hogar), **B** (A + vivienda), **C** (B + laborales/administrativas).

| Variable | Módulo | Descripción | ¿Monetaria? | ¿En el target? | Decisión | Bloque | Justificación |
|---|---|---|---|---|---|---|---|
| P6040 | personas | edad del jefe | no | no | Permitida | A | |
| P3271 | personas | sexo del jefe | no | no | Permitida | A | |
| P3042 | personas | nivel educativo más alto del jefe | no | no | Permitida | A | ordinal |
| P6008 | hogar_vivienda | número de personas del hogar | no | no | Permitida | A | |
| P4005 | hogar_vivienda | tipo de vivienda | no | no | Permitida | B | |
| P4010 | hogar_vivienda | material de las paredes | no | no | Permitida | B | |
| P4020 | hogar_vivienda | material de los pisos | no | no | Permitida | B | |
| P4030S1–S5 | hogar_vivienda | servicios públicos (energía, gas, acueducto, alcantarillado, basuras) | no | no | Permitida | B | una variable por servicio. Energía (S1) se elimina por casi constante, 99,9 % (decisión 23) |
| P5000 | hogar_vivienda | número de cuartos | no | no | Permitida | B | |
| P5010 | hogar_vivienda | cuartos para dormir | no | no | Permitida | B | |
| P5020 | hogar_vivienda | tipo de sanitario | no | no | Permitida | B | |
| P5070 / P5080 | hogar_vivienda | lugar y combustible para cocinar | no | no | Permitida | B | |
| P5090 | hogar_vivienda | tenencia de la vivienda | no | no | Permitida | B | |
| P5222S1–S11 | hogar_vivienda | productos financieros que usa el hogar (cuenta corriente, ahorros, CDT, préstamos, tarjeta de crédito, billetera móvil…) | no | no | Permitida con justificación | C | selección múltiple: vacío = no lo usa (0). Proxy fuerte de capacidad económica. S8 (otro) y S9 (ninguno) se eliminan; S1+S3 y S4+S5 se unen (decisión 23) |
| P5222S10 | hogar_vivienda | productos financieros: "no sabe" | no | no | Excluida | — | no es predictor; los hogares con "no sabe" o sin ninguna opción marcada quedan con vacío y salen en casos completos (25 hogares) |
| P5222S8A1 | hogar_vivienda | texto "¿cuál?" de la opción Otro | no | no | Excluida | — | texto libre, no es predictor |
| P4030S1A1 | hogar_vivienda | estrato | no | no | Permitida con justificación | B | describe el entorno de la vivienda, no el ingreso; se observa en el recibo de servicios. 0 = sin estrato (categoría), 9 = no sabe |
| P6430 | ocupados | posición ocupacional del jefe | no | no | Permitida | C | se agrega a hogar como posición del jefe |
| P6090 | personas | afiliación a salud | no | no | Permitida | C | se agrega como proporción de afiliados del hogar |
| P6100 | personas | régimen de salud | no | no | Permitida con justificación | C | dato administrativo ligado al SISBÉN; se revisa su aporte en la ablación |
| P6920 | ocupados | cotiza a pensión | no | no | Permitida con justificación | C | proxy de formalidad laboral; cumple las 3 reglas. Su aporte se evalúa en la ablación B → C |
| INGLABO | ocupados | ingreso laboral | sí | sí | Prohibida | — | construye el target |
| P6500 | ocupados | salario | sí | sí | Prohibida | — | construye el target |
| P6750 | ocupados | ganancia del trabajo independiente | sí | sí | Prohibida | — | construye el target |
| P7500S2 / P7500S2A1 | otros_ingresos | recibió pensión / monto | sí | sí | Prohibida | — | la pensión se suma al ingreso del hogar |
| P5100 | hogar_vivienda | cuota mensual de la vivienda | sí | no | Prohibida | — | monto en pesos |
| P5110 | hogar_vivienda | cuota de amortización | sí | no | Prohibida | — | monto en pesos |
| P5130 | hogar_vivienda | arriendo estimado (vivienda propia) | sí | sí | Prohibida | — | monto en pesos; ingpcug incluye la imputación de arriendo (decisión 1) |
| P5140 | hogar_vivienda | arriendo pagado | sí | no | Prohibida | — | monto en pesos |
| ingpcug, ypc, li, lp, pobre, indigente | base pobreza DANE | ingreso per cápita y líneas de pobreza | sí | sí | Prohibida | — | columnas exclusivas del target (decisiones 1 y 9) |
| DPTO | hogar_vivienda | departamento | no | no | No es predictor | — | solo se usa para filtrar la población con cobertura en pobreza DANE (decisión 2) |