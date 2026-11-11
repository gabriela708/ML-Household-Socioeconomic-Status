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

PPOM (entrega final, Gaby): clase `PPOM` y función `tablas_ppom()`, al final de este archivo.
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
    tabla4["no_estimable"] = (tabla4[cols].abs() > 10).any(axis=1)
    tabla4["revisar"] = (tabla4["cambia_signo"] | fuera_ic) & ~tabla4["no_estimable"]

    return tabla2, tabla3, tabla4


# ---------------------------------------------------------------------------
# 3. PPOM: odds proporcionales parciales (entrega final, Gaby)
# ---------------------------------------------------------------------------
class PPOM(BaseEstimator, ClassifierMixin):
    """Odds proporcionales parciales con selección automática, como AUTOFIT del paper.

    Modelo (ecuación 8 del paper, PPOM sin restricciones):
        P(ses <= j | x) = F(c_j - x'β - z'γ_j),   γ_0 = 0
    - z son solo las variables que NO cumplen el supuesto de odds proporcionales;
      el resto conserva un único β para todos los cortes (igual que en el POM).
    - Si ninguna variable se relaja, el modelo es exactamente el POM.

    Ajustes para que el modelo no se rompa:
    1. AUTOFIT dentro de cada fold: en `fit` se comparan el POM y las K-1 logísticas
       binarias (ses > j) SOLO con los datos de entrenamiento del fold. Se relaja la
       variable si su β del POM queda fuera del IC (nivel `z`) de alguna logística
       binaria; es el mismo criterio de la tabla 4 de Silva. Así la selección de
       variables no usa el fold de validación (sin fuga).
    2. Nunca se relajan variables no estimables (|coef| > 10 en alguna logística
       binaria, típico de la separación en Alta) ni columnas con menos de
       `min_casos` hogares distintos de cero.
    3. Penalización L2 `alpha_gamma` sobre las desviaciones γ: si una desviación no
       tiene soporte en los datos, se encoge hacia 0 (hacia el POM).
    4. Si las probabilidades acumuladas se cruzan para un hogar, se fuerzan a ser
       monótonas antes de calcular P(ses = j); `tasa_cruces(X)` reporta cuántos.

    Parámetros
    ----------
    alpha : float
        Penalización L2 sobre β (0 = como el POM del proyecto, decisión 36).
    alpha_gamma : float
        Penalización L2 sobre las desviaciones γ por corte (se ajusta en la CV).
    z : float
        Valor crítico del IC del autofit (1,96 = 95 %).
    min_casos : int
        Mínimo de hogares con valor distinto de cero para poder relajar una columna.
    class_weight : None o "balanced"
        Igual que en el POM.
    """

    def __init__(self, alpha=0.0, alpha_gamma=1.0, z=1.96, min_casos=30,
                 class_weight=None, max_iter=3000):
        self.alpha = alpha
        self.alpha_gamma = alpha_gamma
        self.z = z
        self.min_casos = min_casos
        self.class_weight = class_weight
        self.max_iter = max_iter

    # ---------------- autofit: qué variables violan el supuesto ----------------
    def _autofit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        pom = POM(alpha=self.alpha, max_iter=self.max_iter).fit(X, y)  # sin pesos, como la tabla 4
        beta_ref = pom.coef_
        p = X.shape[1]
        relajar = np.zeros(p, dtype=bool)
        estimable = (np.count_nonzero(X, axis=0) >= self.min_casos) & (X.std(axis=0) > 0)
        k = np.searchsorted(self.classes_, y)
        Xc = np.c_[np.ones(len(X)), X]
        for j in range(len(self.classes_) - 1):
            yj = (k > j).astype(int)
            lr = LogisticRegression(C=1e6, max_iter=5000).fit(X, yj)       # ≈ sin penalizar
            b = lr.coef_.ravel()
            pr = lr.predict_proba(X)[:, 1]
            H = Xc.T @ (Xc * (pr * (1 - pr))[:, None])
            ee = np.sqrt(np.clip(np.diag(np.linalg.pinv(H))[1:], 0, None))
            estimable &= (np.abs(b) <= 10) & (ee > 0)
            relajar |= np.abs(beta_ref - b) > self.z * ee
        return relajar & estimable, beta_ref

    # ---------------- verosimilitud ----------------
    def _desempacar(self, params, p, r, K):
        beta = params[:p]
        gamma = np.zeros((r, K - 1))
        gamma[:, 1:] = params[p:p + r * (K - 2)].reshape(r, K - 2)
        t = params[p + r * (K - 2):]
        return beta, gamma, t

    def _perdida(self, params, X, Z, k, w):
        n, p = X.shape
        r, K = Z.shape[1], len(self.classes_)
        beta, gamma, t = self._desempacar(params, p, r, K)
        c = POM._cortes(t)
        U = c[None, :] - (X @ beta)[:, None] - Z @ gamma            # n × (K-1)
        F = expit(U)
        Fe = np.c_[np.zeros(n), F, np.ones(n)]                       # bordes F(-inf)=0, F(inf)=1
        fila = np.arange(n)
        prob = np.clip(Fe[fila, k + 1] - Fe[fila, k], 1e-12, None)
        f = F * (1 - F)

        perdida = (-np.sum(w * np.log(prob)) + 0.5 * self.alpha * beta @ beta
                   + 0.5 * self.alpha_gamma * np.sum(gamma ** 2))

        # G[i, j] = d(pérdida)/dU[i, j]
        G = np.zeros((n, K - 1))
        sup = k < K - 1
        G[fila[sup], k[sup]] -= w[sup] * f[fila[sup], k[sup]] / prob[sup]
        inf_ = k > 0
        G[fila[inf_], k[inf_] - 1] += w[inf_] * f[fila[inf_], k[inf_] - 1] / prob[inf_]

        g_beta = -X.T @ G.sum(axis=1) + self.alpha * beta
        g_gamma = (-Z.T @ G[:, 1:] + self.alpha_gamma * gamma[:, 1:]).ravel()
        g_c = G.sum(axis=0)
        g_t = np.array([g_c.sum()] + [np.exp(t[m]) * g_c[m:].sum() for m in range(1, K - 1)])
        return perdida, np.r_[g_beta, g_gamma, g_t]

    def fit(self, X, y):
        nombres = list(X.columns) if hasattr(X, "columns") else None
        X, y = check_X_y(X, y)
        self.classes_, k = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        n, p = X.shape

        self.relajadas_, beta_ref = self._autofit(X, y)
        Z = X[:, self.relajadas_]
        r = Z.shape[1]

        w = (n / (K * np.bincount(k)))[k] if self.class_weight == "balanced" else np.ones(n)

        acum = np.cumsum(np.bincount(k))[:-1] / n
        c0 = np.log(acum / (1 - acum))
        t0 = np.r_[c0[0], np.log(np.diff(c0))]
        x0 = np.r_[beta_ref, np.zeros(r * (K - 2)), t0]

        res = minimize(self._perdida, x0, args=(X, Z, k, w), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_, self.gamma_, t = self._desempacar(res.x, p, r, K)
        self.cortes_ = POM._cortes(t)
        self.convergio_ = bool(res.success)
        self.n_relajadas_ = int(r)
        self.n_features_in_ = p
        self.nombres_ = nombres
        self.log_verosimilitud_ = -self._perdida(res.x, X, Z, k, np.ones(n))[0] \
            if self.alpha == 0 and self.alpha_gamma == 0 else np.nan
        return self

    def _acumuladas(self, X):
        check_is_fitted(self, "coef_")
        X = check_array(X)
        Z = X[:, self.relajadas_]
        return expit(self.cortes_[None, :] - (X @ self.coef_)[:, None] - Z @ self.gamma_)

    def tasa_cruces(self, X):
        """Proporción de hogares cuyas probabilidades acumuladas se cruzan."""
        acum = self._acumuladas(X)
        return float(np.mean(np.any(np.diff(acum, axis=1) < 0, axis=1)))

    def predict_proba(self, X):
        acum = np.maximum.accumulate(self._acumuladas(X), axis=1)   # fuerza P(≤0) ≤ P(≤1) ≤ ...
        acum = np.c_[np.zeros(len(acum)), acum, np.ones(len(acum))]
        return np.diff(acum, axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def tablas_ppom(Xp, y, z=1.96):
    """Comparación POM vs. PPOM como en el paper (razón de verosimilitud y AIC).

    Usa el 80 % de entrenamiento completo, sin pesos ni penalización (MLE, igual que
    statsmodels en tablas_pom). Xp: salida de preparar_X_inferencia (con nombres).

    Devuelve (tabla5, tabla6):
    tabla5: variables relajadas con su coeficiente en cada corte (β + γ_j).
    tabla6: log-verosimilitud, parámetros, AIC de POM y PPOM y prueba LR.
    """
    from scipy.stats import chi2

    y = np.asarray(y)
    pom = POM(alpha=0.0).fit(Xp.to_numpy(), y)
    ppom = PPOM(alpha=0.0, alpha_gamma=0.0, z=z).fit(Xp, y)

    Xn = Xp.to_numpy()
    K = len(pom.classes_)
    k = np.searchsorted(pom.classes_, y)
    t = np.r_[pom.cortes_[0], np.log(np.diff(pom.cortes_))]
    ll_pom = -pom._perdida(np.r_[pom.coef_, t], Xn, k, np.ones(len(y)))[0]
    ll_ppom = ppom.log_verosimilitud_

    k_pom = Xn.shape[1] + (K - 1)
    k_ppom = k_pom + ppom.n_relajadas_ * (K - 2)
    lr = 2 * (ll_ppom - ll_pom)
    gl = k_ppom - k_pom

    nombres = np.array(Xp.columns)[ppom.relajadas_]
    beta_r = ppom.coef_[ppom.relajadas_]
    tabla5 = pd.DataFrame(
        {f"corte {j + 1}": beta_r + ppom.gamma_[:, j] for j in range(K - 1)}, index=nombres)
    tabla5.insert(0, "POM", pom.coef_[ppom.relajadas_])

    tabla6 = pd.DataFrame({
        "log-verosimilitud": [ll_pom, ll_ppom],
        "parámetros": [k_pom, k_ppom],
        "AIC": [2 * k_pom - 2 * ll_pom, 2 * k_ppom - 2 * ll_ppom],
        "convergió": [pom.convergio_, ppom.convergio_],
    }, index=["POM", "PPOM"])
    tabla6.loc["PPOM", "LR vs. POM"] = lr
    tabla6.loc["PPOM", "gl"] = gl
    tabla6.loc["PPOM", "p LR"] = chi2.sf(lr, gl) if gl > 0 else np.nan
    tabla6.loc["PPOM", "tasa de cruces"] = ppom.tasa_cruces(Xn)
    return tabla5, tabla6


def tabla_or_ppom(Xp, y, z=1.96, paso=1e-5):
    """Odds ratios del PPOM con errores estándar e IC 95 %.

    Ajusta el PPOM por máxima verosimilitud (sin pesos ni penalización) con el 80 %
    completo. Los errores estándar salen de la inversa del Hessiano de la
    -log-verosimilitud, calculado por diferencias finitas del gradiente analítico.

    - Variables proporcionales: una sola fila (corte = "todos"), igual que en el POM.
    - Variables relajadas: una fila por corte (β + γ_j).
    Las numéricas están estandarizadas: el OR es por cada desviación estándar.
    """
    from scipy.stats import norm

    y = np.asarray(y)
    m = PPOM(alpha=0.0, alpha_gamma=0.0, z=z).fit(Xp, y)
    X = Xp.to_numpy(dtype=float)
    Z = X[:, m.relajadas_]
    k = np.searchsorted(m.classes_, y)
    K, (n, p), r = len(m.classes_), X.shape, Z.shape[1]
    w = np.ones(n)

    t = np.r_[m.cortes_[0], np.log(np.diff(m.cortes_))]
    theta = np.r_[m.coef_, m.gamma_[:, 1:].ravel(), t]
    grad = lambda th: m._perdida(th, X, Z, k, w)[1]
    H = np.empty((theta.size, theta.size))
    for i in range(theta.size):
        e = np.zeros(theta.size)
        e[i] = paso
        H[:, i] = (grad(theta + e) - grad(theta - e)) / (2 * paso)
    cov = np.linalg.pinv((H + H.T) / 2)

    nombres = np.array(Xp.columns)
    idx_r = {j: i for i, j in enumerate(np.flatnonzero(m.relajadas_))}
    filas = []
    for j in range(p):
        if j not in idx_r:
            cortes = [("todos", m.coef_[j], cov[j, j])]
        else:
            cortes = []
            for c in range(K - 1):
                if c == 0:
                    coef, var = m.coef_[j], cov[j, j]
                else:
                    g = p + idx_r[j] * (K - 2) + (c - 1)
                    coef = m.coef_[j] + m.gamma_[idx_r[j], c]
                    var = cov[j, j] + cov[g, g] + 2 * cov[j, g]
                cortes.append((f"corte {c + 1}", coef, var))
        for corte, coef, var in cortes:
            ee = np.sqrt(max(var, 0))
            filas.append({"variable": nombres[j], "corte": corte, "coef": coef, "ee": ee,
                          "p": 2 * norm.sf(abs(coef / ee)) if ee > 0 else np.nan,
                          "OR": np.exp(coef), "OR_IC95_inf": np.exp(coef - 1.96 * ee),
                          "OR_IC95_sup": np.exp(coef + 1.96 * ee)})
    return pd.DataFrame(filas)