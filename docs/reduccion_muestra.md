# Reducción de la muestra

# Reducción de la muestra

## Embudo de hogares

| Paso | Hogares | Criterio |
|---|---|---|
| Hogares GEIH (diciembre 2025) | 24.159 | Módulo Datos del hogar y la vivienda |
| Cabeceras | 21.037 | CLASE = 1 (zona urbana) |
| Casos completos | 20.991 | Sin vacíos en ninguno de los 38 predictores |

Se excluyen 46 hogares de cabecera (0,2 %). El tamaño final supera ampliamente el mínimo de 5.000 registros.

## Causas de exclusión

| Causa | Hogares |
|---|---|
| Productos financieros (P5222): respondió "no sabe" (9) o no marcó ninguna opción (19) | 28 |
| Estrato (P4030S1A1): "no sabe" | 16 |
| Afiliación a salud: todos los miembros respondieron "no sabe" | 2 |

Antes de este filtro se recodificaron los vacíos que tenían significado, para no perder hogares de forma injustificada: las opciones no marcadas de P5222 (= 0), el combustible de hogares que no cocinan (= 0), el estrato de hogares sin energía (= 7) y los hogares sin ocupados (= 0). Sin esa revisión, la pregunta de productos financieros habría eliminado casi toda la muestra.

## Incluidos frente a excluidos

| Variable | Media incluidos | Media excluidos | Diferencia estandarizada |
|---|---|---|---|
| tiene_alcantarillado | 0,90 | 0,54 | −0,85 |
| tiene_gas_natural | 0,79 | 0,48 | −0,69 |
| n_cuartos | 3,30 | 2,85 | −0,44 |
| n_cotizantes | 0,56 | 0,35 | −0,31 |
| prop_subsidiado | 0,47 | 0,58 | 0,23 |
| educ_jefe | 5,86 | 5,48 | −0,14 |
| edad_jefe | 49,99 | 47,78 | −0,12 |
| n_personas | 2,75 | 2,61 | −0,10 |
| prop_dependientes | 0,31 | 0,34 | 0,10 |
| n_ocupados | 1,23 | 1,30 | 0,08 |
| n_menores15 | 0,53 | 0,57 | 0,04 |
| n_mayores65 | 0,36 | 0,35 | −0,01 |

## Conclusión

Los hogares excluidos tienen un perfil más vulnerable: menor acceso a alcantarillado y gas natural, menos cuartos, menos cotizantes a pensión y mayor proporción en régimen subsidiado. Sin embargo, representan el 0,2 % de los hogares de cabecera, por lo que su exclusión no altera de forma relevante la distribución de la muestra. Se reporta como limitación menor.

**Pendiente:** cuando esté `y_hogar.parquet`, agregar la distribución de las clases SES en incluidos y excluidos.