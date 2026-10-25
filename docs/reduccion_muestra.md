# Reducción de la muestra

## Embudo de hogares

| Paso | Hogares | Criterio |
|---|---|---|
| Hogares GEIH (diciembre 2025) | 24.159 | Módulo Datos del hogar y la vivienda |
| Cabeceras | 21.037 | CLASE = 1 (zona urbana) |
| Con cobertura en la base de pobreza del DANE | 19.690 | Se excluyen 9 departamentos sin medición de pobreza (decisión 2) |
| Casos completos | 19.660 | Sin vacíos en ninguno de los 38 predictores |

Se excluyen 30 hogares de la población (0,15 %). El tamaño final supera ampliamente el mínimo de 5.000 registros.

## Causas de exclusión

| Causa | Hogares |
|---|---|
| Productos financieros (P5222): respondió "no sabe" (8) o no marcó ninguna opción (17) | 25 |
| Estrato (P4030S1A1): "no sabe" | 3 |
| Afiliación a salud: todos los miembros respondieron "no sabe" | 2 |

Antes de este filtro se recodificaron los vacíos que tenían significado, para no perder hogares de forma injustificada: las opciones no marcadas de P5222 (= 0), el combustible de hogares que no cocinan (= 0), el estrato de hogares sin energía (= 7) y los hogares sin ocupados (= 0). Sin esa revisión, la pregunta de productos financieros habría eliminado casi toda la muestra.

## Incluidos frente a excluidos

### Por clase socioeconómica

| Clase | Incluidos | % | Excluidos | % |
|---|---|---|---|---|
| 0 Pobre | 4.253 | 21,6 | 3 | 10,0 |
| 1 Vulnerable | 4.626 | 23,5 | 8 | 26,7 |
| 2 Media | 9.911 | 50,4 | 19 | 63,3 |
| 3 Alta | 870 | 4,4 | 0 | 0,0 |
| **Total** | **19.660** | 100 | **30** | 100 |

### Por características del hogar

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

Los 30 hogares excluidos (0,15 % de la población) son más pequeños y tienen menos cuartos, menos acceso a alcantarillado y gas natural y menos cotizantes a pensión. Sin embargo, por clase socioeconómica no son más pobres: el 10 % es Pobre, frente al 21,6 % de los incluidos. Por su tamaño, la exclusión no altera la distribución de clases de la muestra (las proporciones de los incluidos coinciden con las de la población completa al decimal). Se reporta como limitación menor.