# Reducción de la muestra

# Reducción de la muestra

## Embudo de hogares
| Paso | Hogares | Criterio |
|---|---|---|
| Hogares GEIH (diciembre 2025) | 24.159 | Módulo Datos del hogar y la vivienda |
| Cabeceras | 21.037 | CLASE = 1 (zona urbana) |
| Con cobertura en la base de pobreza del DANE | 19.690 | Se excluyen 9 departamentos sin medición de pobreza (decisión 2) |
| Casos completos | 19.660 | Sin vacíos en ninguno de los 38 predictores |

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
| n_cuartos | 3,32 | 2,67 | −0,64 |
| tiene_alcantarillado | 0,91 | 0,73 | −0,47 |
| n_personas | 2,77 | 2,23 | −0,40 |
| tiene_gas_natural | 0,84 | 0,67 | −0,40 |
| n_menores15 | 0,53 | 0,30 | −0,30 |
| n_cotizantes | 0,55 | 0,40 | −0,22 |
| prop_subsidiado | 0,47 | 0,51 | 0,08 |
| edad_jefe | 50,25 | 49,20 | −0,06 |
| educ_jefe | 5,84 | 5,67 | −0,06 |
| n_mayores65 | 0,36 | 0,33 | −0,05 |
| n_ocupados | 1,23 | 1,20 | −0,04 |
| prop_dependientes | 0,31 | 0,32 | 0,03 |
## Conclusión

Los hogares excluidos tienen un perfil más vulnerable: menor acceso a alcantarillado y gas natural, menos cuartos, menos cotizantes a pensión y mayor proporción en régimen subsidiado.Los 30 excluidos (0,15 % de la población) pero son tan pocos que no cambian la muestra.

**Pendiente:** cuando esté `y_hogar.parquet`, agregar la distribución de las clases SES en incluidos y excluidos.