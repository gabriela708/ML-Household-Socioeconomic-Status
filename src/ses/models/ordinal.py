"""Regresión logística ordinal de odds proporcionales (POM).

Responsable: Silva. Día 4.

Dos usos:
1. Predicción (ablación y comparación de modelos): la clase `POM`, compatible con scikit-learn,
   entra en `construir_pipeline()` igual que cualquier otro modelo.

       from ses.models.ordinal import POM
       from ses.pipeline import cargar_entrenamiento, construir_pipeline
       X, y, cv = cargar_entrenamiento()
       pipe = construir_pipeline(POM(class_weight="balanced"), bloque="C")
       cross_validate(pipe, X, y, cv=cv, scoring=SCORERS)

2. Inferencia (Tablas 2–4 del informe): `preparar_X_inferencia()` + `tablas_pom()`, con
   statsmodels, sobre el 80 % de entrenamiento completo.

Modelo (igual al del paper y al OrderedModel de statsmodels):
    P(ses <= j | x) = F(c_j - x'β),  F = logística,  j = 0, 1, 2
- Un solo β por variable para los 3 cortes (supuesto de odds proporcionales).
- β > 0 → mayores odds de estar en una clase más alta. OR = exp(β).
- Sin constante: los cortes c_0 < c_1 < c_2 cumplen ese papel.

El PPOM no se implementa: queda como siguiente paso (decisiones.md).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


# ---------------------------------------------------------------------------
# 1. POM para predicción (compatible con scikit-learn)
# ---------------------------------------------------------------------------
class POM(BaseEstimator, ClassifierMixin):
    """POM por máxima verosimilitud.

    Parámetros
    ----------
    alpha : float
        Penalización L2 sobre β (0 = POM sin regularizar, como en el paper).
    class_weight : None o "balanced"
        "balanced" pondera cada hogar por n / (K · n_clase), igual que en la logística
        de Juanita, para que la comparación sea justa con Alta = 4,4 %.
    max_iter : int
        Iteraciones máximas de L-BFGS.
    """

    def __init__(self, alpha=0.0, class_weight=None, max_iter=2000):
        self.alpha = alpha
        self.class_weight = class_weight
        self.max_iter = max_iter

    # -- cortes: c_0 = t_0 ; c_j = c_{j-1} + exp(t_j)  (así siempre quedan en orden) --
    @staticmethod
    def _cortes(t):
        return np.cumsum(np.r_[t[0], np.exp(t[1:])])

    def _perdida(self, params, X, k, w):
        """-log-verosimilitud ponderada + penalización, y su gradiente."""
        p = X.shape[1]
        beta, t = params[:p], params[p:]
        c = np.r_[-np.inf, self._cortes(t), np.inf]
        eta = X @ beta

        F_sup, F_inf = expit(c[k + 1] - eta), expit(c[k] - eta)
        prob = np.clip(F_sup - F_inf, 1e-12, None)
        f_sup, f_inf = F_sup * (1 - F_sup), F_inf * (1 - F_inf)

        perdida = -np.sum(w * np.log(prob)) + 0.5 * self.alpha * beta @ beta

        # gradiente respecto a β
        g_eta = w * (f_sup - f_inf) / prob
        g_beta = X.T @ g_eta + self.alpha * beta

        # gradiente respecto a cada corte c_j y luego a los parámetros t
        K = len(self.classes_)
        g_c = np.zeros(K - 1)
        for j in range(K - 1):
            g_c[j] = -np.sum(w[k == j] * f_sup[k == j] / prob[k == j]) \
                     + np.sum(w[k == j + 1] * f_inf[k == j + 1] / prob[k == j + 1])
        g_t = np.array([g_c.sum()] + [np.exp(t[m]) * g_c[m:].sum() for m in range(1, K - 1)])

        return perdida, np.r_[g_beta, g_t]

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, k = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        n, p = X.shape

        if self.class_weight == "balanced":
            w = (n / (K * np.bincount(k)))[k]
        else:
            w = np.ones(n)

        # inicio: β = 0 y cortes en el logit de las proporciones acumuladas
        acum = np.cumsum(np.bincount(k))[:-1] / n
        c0 = np.log(acum / (1 - acum))
        t0 = np.r_[c0[0], np.log(np.diff(c0))]

        res = minimize(self._perdida, np.r_[np.zeros(p), t0], args=(X, k, w),
                       jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = res.x[:p]
        self.cortes_ = self._cortes(res.x[p:])
        self.convergio_ = bool(res.success)
        self.n_features_in_ = p
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X)
        eta = X @ self.coef_
        acum = expit(self.cortes_[None, :] - eta[:, None])          # P(ses <= j)
        acum = np.c_[np.zeros(len(X)), acum, np.ones(len(X))]
        return np.diff(acum, axis=1)                                 # P(ses = j)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# 2. POM para inferencia (Tablas 2–4), con statsmodels
# ---------------------------------------------------------------------------
def preparar_X_inferencia(X, min_frecuencia=30):
    """Preprocesa los 32 predictores para statsmodels, con nombres de columna.

    - Mismo preprocesamiento de Juanita (bloque C), ajustado con todo el 80 %.
    - Categorías con < min_frecuencia hogares → "infrequent_sklearn" (evita errores
      estándar enormes).
    - Quita lugar_cocina = 6 ("no cocina"): es idéntica a la categoría de referencia de
      combustible_cocina (0 = no cocina) → colinealidad perfecta.
    """
    from ses.pipeline import BLOQUE_C, construir_preprocesador

    prep = construir_preprocesador(BLOQUE_C)
    prep.set_params(nom__onehot__min_frequency=min_frecuencia)
    Xp = pd.DataFrame(prep.fit_transform(X), columns=prep.get_feature_names_out(), index=X.index)
    return Xp.drop(columns="nom__lugar_cocina_6.0", errors="ignore")


def tablas_pom(Xp, y):
    """Ajusta el POM con statsmodels y devuelve (tabla2, tabla3, tabla4).

    tabla2: coeficientes, OR e IC 95 %.
    tabla3: cortes, log-verosimilitud, AIC y razón de verosimilitud frente al nulo.
    tabla4: supuesto de odds proporcionales con 3 logísticas binarias (ses > 0, > 1, > 2).
    Tarda unos minutos (derivadas numéricas).
    """
    import statsmodels.api as sm
    from scipy.stats import chi2
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    res = OrderedModel(y, Xp, distr="logit").fit(method="bfgs", maxiter=5000, disp=False)
    k = Xp.shape[1]
    coef, ic = res.params.iloc[:k], res.conf_int().iloc[:k]

    tabla2 = pd.DataFrame({
        "coef": coef, "ee": res.bse.iloc[:k], "p": res.pvalues.iloc[:k],
        "OR": np.exp(coef), "OR_IC95_inf": np.exp(ic[0]), "OR_IC95_sup": np.exp(ic[1]),
    })

    cortes = res.model.transform_threshold_params(res.params)[1:-1]
    n = np.bincount(y)
    ll_nulo = (n * np.log(n / n.sum())).sum()
    lr = 2 * (res.llf - ll_nulo)
    tabla3 = pd.Series({
        "corte Pobre | Vulnerable": cortes[0], "corte Vulnerable | Media": cortes[1],
        "corte Media | Alta": cortes[2], "log-verosimilitud": res.llf,
        "log-verosimilitud nulo": ll_nulo, "AIC": res.aic,
        "razón de verosimilitud (LR)": lr, "gl": k, "p LR": chi2.sf(lr, k),
        "convergió": res.mle_retvals["converged"],
    }, name="valor")

    Xc = sm.add_constant(Xp)
    tabla4 = pd.DataFrame({"POM": coef})
    fuera_ic = pd.Series(False, index=coef.index)
    for j, nombre in enumerate(["corte 1 (≥ Vulnerable)", "corte 2 (≥ Media)", "corte 3 (Alta)"]):
        logit = sm.Logit((y > j).astype(int), Xc).fit(disp=False, maxiter=500)
        tabla4[nombre] = logit.params.drop("const")
        ic_j = logit.conf_int().drop("const")
        fuera_ic |= (coef < ic_j[0]) | (coef > ic_j[1])
    cols = tabla4.columns[1:]
    tabla4["cambia_signo"] = (tabla4[cols] > 0).nunique(axis=1) > 1
    tabla4["revisar"] = tabla4["cambia_signo"] | fuera_ic

    return tabla2, tabla3, tabla4
