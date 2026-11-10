"""Textos del formulario de la app: qué ve el usuario y qué código recibe el modelo.

Responsable: Juanita. Entrega final.

Este archivo solo tiene datos (diccionarios), sin lógica ni Streamlit, para que:
- la app y las pruebas usen exactamente los mismos códigos que X_hogar (coherencia entrenamiento-uso);
- si cambia un texto o un código, se cambie en un solo lugar.

Fuente de los textos: diccionario de datos GEIH 2025 (DANE),
https://microdatos.dane.gov.co/index.php/catalog/853/data-dictionary
Los códigos 0 y 7 marcados como "código del equipo" vienen de docs/decisiones.md.

Regla: solo se ofrecen los códigos que aparecen en el 80 % de entrenamiento. Un código que el
modelo nunca vio se convertiría en ceros sin avisar (handle_unknown="ignore", decisión 32).
Por eso no está "Cuarto(s) en otro tipo de estructura" (P4000 = 4): 0 hogares en entrenamiento.
"""

# ---------------------------------------------------------------------------
# Clases del target (src/ses/data/target.py)
# ---------------------------------------------------------------------------
CLASES = {0: "Pobre", 1: "Vulnerable", 2: "Media", 3: "Alta"}

# ---------------------------------------------------------------------------
# Variables nominales: {texto que ve el usuario: código que recibe el modelo}
# ---------------------------------------------------------------------------
SEXO_JEFE = {  # P3271
    "Hombre": 1,
    "Mujer": 2,
}

TIPO_VIVIENDA = {  # P4000
    "Casa": 1,
    "Apartamento": 2,
    "Cuarto(s) en inquilinato": 3,
    "Vivienda indígena": 5,
    "Otra vivienda (carpa, vagón, embarcación, cueva, refugio natural)": 6,
}

MATERIAL_PAREDES = {  # P4010 (el 9 es válido, decisión 16)
    "Ladrillo, bloque, material prefabricado, piedra": 1,
    "Madera pulida": 2,
    "Adobe o tapia pisada": 3,
    "Bahareque": 4,
    "Madera burda, tabla, tablón": 5,
    "Guadua": 6,
    "Caña, esterilla u otro material vegetal": 7,
    "Zinc, tela, cartón, latas, desechos, plástico": 8,
    "Sin paredes": 9,
}

MATERIAL_PISOS = {  # P4020
    "Tierra, arena": 1,
    "Cemento, gravilla": 2,
    "Madera burda, tabla, tablón u otro vegetal": 3,
    "Baldosín, ladrillo, vinisol u otros sintéticos": 4,
    "Mármol": 5,
    "Madera pulida": 6,
    "Alfombra o tapete de pared a pared": 7,
}

ESTRATO = {  # P4030S1A1 (decisión 11)
    "Sin estrato": 0,
    "Estrato 1": 1,
    "Estrato 2": 2,
    "Estrato 3": 3,
    "Estrato 4": 4,
    "Estrato 5": 5,
    "Estrato 6": 6,
    "La vivienda no tiene energía eléctrica": 7,  # código del equipo
}

TIPO_SANITARIO = {  # P5020
    "Inodoro conectado a alcantarillado": 1,
    "Inodoro conectado a pozo séptico": 2,
    "Inodoro sin conexión": 3,
    "Letrina": 4,
    "Bajamar": 5,
    "No tiene servicio sanitario": 6,
}

LUGAR_COCINA = {  # P5070
    "En un cuarto usado solo para cocinar": 1,
    "En un cuarto usado también para dormir": 2,
    "En una sala comedor con lavaplatos": 3,
    "En una sala comedor sin lavaplatos": 4,
    "En un patio, corredor, enramada o al aire libre": 5,
    "En ninguna parte, no preparan alimentos": 6,
}

COMBUSTIBLE_COCINA = {  # P5080 (decisión 7)
    "No cocinan en el hogar": 0,  # código del equipo: P5070 = 6
    "Electricidad": 1,
    "Petróleo, gasolina, kerosene, alcohol": 2,
    "Gas natural conectado a red pública": 3,
    "Gas propano en cilindro o pipeta": 4,
    "Leña, madera o carbón de leña": 5,
    "Carbón mineral": 6,
    "Materiales de desecho": 7,
}

TENENCIA_VIVIENDA = {  # P5090 (6 y 7: verificar textos en el cuestionario GEIH 2025)
    "Propia, totalmente pagada": 1,
    "Propia, la están pagando": 2,
    "En arriendo o subarriendo": 3,
    "En usufructo (prestada, sin pagar arriendo)": 4,
    "Posesión sin título (ocupante de hecho)": 5,
    "Propiedad colectiva": 6,
    "Otra": 7,
}

POSICION_OCUP_JEFE = {  # P6430, solo bloque C (decisión 8)
    "El jefe del hogar no está trabajando": 0,  # código del equipo
    "Obrero o empleado de empresa particular": 1,
    "Obrero o empleado del gobierno": 2,
    "Empleado doméstico": 3,
    "Trabajador por cuenta propia": 4,
    "Patrón o empleador": 5,
    "Trabajador familiar sin remuneración": 6,
    "Trabajador sin remuneración en empresas de otros hogares": 7,
    "Jornalero o peón": 8,
}

# educ_jefe es numérica en el Pipeline (ordinal 1-13), pero en el formulario se elige por nombre.
EDUC_JEFE = {  # P3042 (el 9 es válido, decisión 16)
    "Ninguno": 1,
    "Preescolar": 2,
    "Básica primaria (1.º a 5.º)": 3,
    "Básica secundaria (6.º a 9.º)": 4,
    "Media académica (bachillerato clásico)": 5,
    "Media técnica (bachillerato técnico)": 6,
    "Normalista": 7,
    "Técnica profesional": 8,
    "Tecnológica": 9,
    "Universitaria": 10,
    "Especialización": 11,
    "Maestría": 12,
    "Doctorado": 13,
}

# Variable del modelo -> su diccionario (las que se eligen de una lista)
OPCIONES = {
    "sexo_jefe": SEXO_JEFE,
    "educ_jefe": EDUC_JEFE,
    "tipo_vivienda": TIPO_VIVIENDA,
    "material_paredes": MATERIAL_PAREDES,
    "material_pisos": MATERIAL_PISOS,
    "estrato": ESTRATO,
    "tipo_sanitario": TIPO_SANITARIO,
    "lugar_cocina": LUGAR_COCINA,
    "combustible_cocina": COMBUSTIBLE_COCINA,
    "tenencia_vivienda": TENENCIA_VIVIENDA,
    "posicion_ocup_jefe": POSICION_OCUP_JEFE,
}

# Códigos con menos de 30 hogares en entrenamiento: se ofrecen, pero la app avisa que la
# predicción es menos confiable porque el modelo vio muy pocos casos así.
MIN_HOGARES = 30
CODIGOS_POCO_FRECUENTES = {
    "tipo_vivienda": [5, 6],                # 18 y 4 hogares
    "material_paredes": [6, 9],             # 4 y 1
    "material_pisos": [5, 7],               # 24 y 11
    "estrato": [7],                         # 16
    "tipo_sanitario": [4],                  # 9
    "combustible_cocina": [2, 6, 7],        # 4, 3 y 1
    "posicion_ocup_jefe": [8],              # 4
}

# ---------------------------------------------------------------------------
# Preguntas en lenguaje normal (lo que ve el usuario en vez del nombre de la variable)
# ---------------------------------------------------------------------------
PREGUNTAS = {
    # Bloque A: quiénes viven en el hogar
    "n_personas": "¿Cuántas personas viven en el hogar?",
    "edad_jefe": "Edad del jefe o jefa del hogar (años)",
    "sexo_jefe": "Sexo del jefe o jefa del hogar",
    "educ_jefe": "Nivel educativo más alto del jefe o jefa",
    "n_menores15": "¿Cuántas personas tienen menos de 15 años?",
    "n_mayores65": "¿Cuántas personas tienen 65 años o más?",
    # Bloque B: cómo es la vivienda
    "tipo_vivienda": "Tipo de vivienda",
    "material_paredes": "Material principal de las paredes exteriores",
    "material_pisos": "Material principal de los pisos",
    "estrato": "Estrato que aparece en el recibo de la luz",
    "n_cuartos": "¿Cuántos cuartos tiene la vivienda en total?",
    "n_dormitorios": "¿En cuántos de esos cuartos duermen?",
    "tipo_sanitario": "Servicio sanitario del hogar",
    "lugar_cocina": "¿Dónde preparan los alimentos?",
    "combustible_cocina": "¿Con qué cocinan?",
    "tenencia_vivienda": "La vivienda es…",
    "tiene_gas_natural": "Gas natural conectado a red pública",
    "tiene_alcantarillado": "Alcantarillado",
    "tiene_recoleccion_basuras": "Recolección de basuras",
    "tiene_acueducto": "Acueducto",
    # Bloque C: trabajo, salud y productos financieros
    "posicion_ocup_jefe": "¿En qué trabaja el jefe o jefa del hogar?",
    "n_ocupados": "¿Cuántas personas del hogar están trabajando?",
    "n_cotizantes": "¿Cuántas personas del hogar cotizan a pensión?",
    "n_afiliados_salud": "¿Cuántas personas están afiliadas a salud?",
    "n_subsidiados": "¿Cuántas personas están en el régimen subsidiado de salud?",
    "fin_corriente_o_cdt": "Cuenta corriente o CDT",
    "fin_cuenta_ahorros": "Cuenta de ahorros",
    "fin_credito_vivienda_vehiculo": "Crédito de vivienda o de vehículo",
    "fin_prestamo_libre_inversion": "Préstamo de libre inversión",
    "fin_tarjeta_credito": "Tarjeta de crédito",
    "fin_billetera_movil": "Billetera móvil (Nequi, Daviplata…)",
}
# n_afiliados_salud y n_subsidiados no son columnas del modelo: con ellas procesar.py calcula
# prop_afiliados_salud y prop_subsidiado (build_features.py, líneas 141-149).

# ---------------------------------------------------------------------------
# Rangos de las numéricas: (mínimo permitido, máximo visto en entrenamiento, máximo permitido)
# Entre el máximo visto y el permitido la app deja seguir, pero avisa que el modelo
# no vio hogares así (extrapolación).
# ---------------------------------------------------------------------------
RANGOS = {
    "n_personas": (1, 13, 20),
    "edad_jefe": (16, 102, 110),
    "n_menores15": (0, 6, 15),
    "n_mayores65": (0, 4, 10),
    "n_cuartos": (1, 15, 20),      # el máximo real es 41, pero el 99,9 % tiene 15 o menos
    "n_dormitorios": (1, 7, 15),
    "n_ocupados": (0, 7, 20),
    "n_cotizantes": (0, 5, 20),
    "n_afiliados_salud": (0, 13, 20),
    "n_subsidiados": (0, 13, 20),
}

# ---------------------------------------------------------------------------
# Grupos del formulario (orden en pantalla)
# ---------------------------------------------------------------------------
GRUPO_A = ["n_personas", "n_menores15", "n_mayores65", "edad_jefe", "sexo_jefe", "educ_jefe"]
GRUPO_B = ["tipo_vivienda", "material_paredes", "material_pisos", "estrato",
           "n_cuartos", "n_dormitorios", "tipo_sanitario", "lugar_cocina",
           "combustible_cocina", "tenencia_vivienda"]
SERVICIOS = ["tiene_acueducto", "tiene_alcantarillado", "tiene_gas_natural",
             "tiene_recoleccion_basuras"]
GRUPO_C = ["posicion_ocup_jefe", "n_ocupados", "n_cotizantes",
           "n_afiliados_salud", "n_subsidiados"]
FINANCIEROS = ["fin_cuenta_ahorros", "fin_billetera_movil", "fin_tarjeta_credito",
               "fin_prestamo_libre_inversion", "fin_credito_vivienda_vehiculo",
               "fin_corriente_o_cdt"]
