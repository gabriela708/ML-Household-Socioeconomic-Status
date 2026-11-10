"""Modelos candidatos y sus grillas de hiperparámetros.

Responsable: Gaby. Día 3.

Todos se comparan con el bloque C (las 32 variables permitidas), para que la
diferencia entre modelos se deba al modelo y no a la información disponible.
La ablación A -> B -> C (scripts/run_ablacion.py).

Desbalance: class_weight="balanced" en las logísticas y el PPOM; priors uniformes en
Naive Bayes, LDA y QDA; KNN no admite pesos por clase (se reporta como limitación).

Uso:
    from ses.models.candidates import CANDIDATOS, construir_busqueda
    busqueda = construir_busqueda("knn", cv)
    busqueda.fit(X, y)
"""

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ses.evaluation.metrics import METRICA_PRINCIPAL, SCORERS
from ses.models.ordinal import PPOM
from ses.pipeline import construir_pipeline

BLOQUE_COMPARACION = "C"
GRILLA_C = [0.01, 0.1, 1, 10]
GRILLA_K = [5, 15, 31, 51]
GRILLA_PCA = ["passthrough", PCA(n_components=10), PCA(n_components=20)]
GRILLA_ALPHA_GAMMA = [0.1, 1, 10]     # penalización de las desviaciones del PPOM
PRIORS_UNIFORMES = [0.25, 0.25, 0.25, 0.25]   # desbalance: igual que Naive Bayes


CANDIDATOS = {
    "lr_multinomial": {
        "modelo": LogisticRegression(penalty=None, class_weight="balanced", max_iter=3000),
        "grilla": {},
        "opciones": {},
    },
    "lr_l2": {
        "modelo": LogisticRegression(penalty="l2", class_weight="balanced", max_iter=3000),
        "grilla": {"modelo__C": GRILLA_C},
        "opciones": {},
    },
    "lr_l1": {
        "modelo": LogisticRegression(penalty="l1", solver="saga", class_weight="balanced",
                                     max_iter=5000, tol=1e-3),
        "grilla": {"modelo__C": GRILLA_C},
        "opciones": {},
    },
    "naive_bayes": {
        "modelo": GaussianNB(priors=[0.25, 0.25, 0.25, 0.25]),
        "grilla": {"modelo__var_smoothing": [1e-9, 1e-6, 1e-3]},
        "opciones": {},
    },
    "knn": {
        "modelo": Pipeline([("reduccion", "passthrough"), ("knn", KNeighborsClassifier())]),
        "grilla": {
            "modelo__reduccion": GRILLA_PCA,
            "modelo__knn__n_neighbors": GRILLA_K,
            "modelo__knn__p": [1, 2],          # 1 = Manhattan, 2 = euclidiana
        },
        "opciones": {},
    },
    # Entrega final: LDA y QDA.
    # LDA: "svd" no estima la covarianza completa (aguanta colinealidad); "lsqr" permite
    # encogerla (shrinkage), útil con dummies muy correlacionadas.
    "lda": {
        "modelo": LinearDiscriminantAnalysis(priors=PRIORS_UNIFORMES),
        "grilla": [{"modelo__solver": ["svd"]},
                   {"modelo__solver": ["lsqr"], "modelo__shrinkage": ["auto", 0.1, 0.5]}],
        "opciones": {},
    },
    # QDA: una covarianza por clase; con dummies que son constantes dentro de una clase
    # (sobre todo en Alta) la covarianza es singular, por eso siempre con reg_param > 0.
    "qda": {
        "modelo": QuadraticDiscriminantAnalysis(priors=PRIORS_UNIFORMES),
        "grilla": {"modelo__reg_param": [0.01, 0.1, 0.5, 0.9]},
        "opciones": {},
    },
    # Entrega final: PPOM con autofit dentro de cada fold.
    # alpha = 0 en β, igual que el POM; solo se ajusta alpha_gamma.
    "ppom": {
        "modelo": PPOM(alpha=0.0, class_weight="balanced"),
        "grilla": {"modelo__alpha_gamma": GRILLA_ALPHA_GAMMA},
        "opciones": {},
    },
}


def construir_busqueda(nombre, cv, bloque=BLOQUE_COMPARACION, n_jobs=-1):
    """GridSearchCV del candidato `nombre` sobre los folds fijos `cv`.

    Calcula todas las métricas de SCORERS y elige el mejor hiperparámetro por la
    métrica principal (macro-F1). Sin grilla, evalúa el modelo tal cual.
    """
    c = CANDIDATOS[nombre]
    pipe = construir_pipeline(c["modelo"], bloque=bloque, **c["opciones"])
    return GridSearchCV(
        pipe,
        param_grid=c["grilla"] or [{}],
        scoring=SCORERS,
        refit=METRICA_PRINCIPAL,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )

