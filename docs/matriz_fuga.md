# Matriz de fuga

Regla de admisión: una variable entra como predictor solo si
1. no es un monto en pesos,
2. no se usa en la fórmula del target,
3. puede observarse sin preguntar por el ingreso.

Bloques: **A** (paper: jefe y tamaño del hogar), **B** (A + vivienda), **C** (B + laborales/administrativas).

| Variable | Módulo | Descripción | ¿Monetaria? | ¿En el target? | Decisión | Bloque | Justificación |
|---|---|---|---|---|---|---|---|
| INGLABO | ocupados | ingreso laboral | sí | sí | Prohibida | — | construye el target |
| P6500 | ocupados | salario | sí | sí | Prohibida | — | |
| P5100 / P5130 / P5140 | hogar_vivienda | cuota / arriendo estimado / arriendo pagado | sí | no | Prohibida | — | monto en pesos |
| P6040 | personas | edad del jefe | no | no | Permitida | A | |
| P6430 | ocupados | posición ocupacional del jefe | no | no | Permitida | C | |
| P6100 | personas | régimen de salud | no | no | Permitida con justificación | C | ligada al SISBÉN |
| P6920 | ocupados | cotiza pensión | no | no | Permitida con justificación | C | proxy de formalidad |
| P4030S1A1 | hogar_vivienda | estrato | no | no | Permitida | B | 0 = sin estrato (categoría), 9 = no sabe |
| ... | | | | | | | |
