"""Sesión 6 · Experimentos: diseño y lectura de un A/B con el correo de Hillstrom.

Uso:
    python -m growth_lab.experimentos

Decisión 008:
- Métrica principal: **gasto** por cliente en 2 semanas; la **visita** es la métrica guía.
- Efecto mínimo a detectar (MDE): gasto +50% (80% de potencia, α = 5% bilateral).
- Lectura clásica (IC, p-valor con corrección de Holm por dos e-mails) **y** bayesiana.

Pasos:
1. ¿El sorteo salió bien? (SRM y balance de variables previas)
2. Potencia: cuánta gente hacía falta y qué efecto mínimo se puede ver con la que hay.
3. Efecto de cada e-mail vs. sin e-mail (Welch y bootstrap) con Holm.
4. CUPED y ajuste por regresión: cuánto ruido quitan las variables previas.
5. Mirar antes de tiempo: simulación A/A de *peeking* y límites de O'Brien-Fleming.
6. Bayesiano: P(e-mail mejor), mejora esperada y pérdida esperada.
7. Economía: gasto incremental por 1.000 correos y costo máximo por correo.

Decisión 009: margen supuesto de 30% (`MARGEN`, sensibilidad 20–40%); los experimentos se leen de forma
secuencial con límites de O'Brien-Fleming calibrados; recomendación: correo de hombres para todos
mientras el uplift (sesión 7) dice a quién personalizar.

Hillstrom es de uso libre: todas las salidas se publican en data/processed/ (prefijo exp_).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from growth_lab import db

CONTROL = "No E-Mail"
TRATAMIENTOS = ["Mens E-Mail", "Womens E-Mail"]
METRICAS = ["spend", "visit", "conversion"]  # la primera es la principal
ALFA, POTENCIA, MDE_RELATIVO = 0.05, 0.80, 0.50
MARGEN = 0.30  # supuesto: margen bruto sobre el gasto, para el costo de equilibrio por correo
SEMILLA = 0
COVARIABLES_NUM = ["recency", "history", "mens", "womens", "newbie"]
COVARIABLES_CAT = ["zip_code", "channel"]


def cargar(con=None) -> pd.DataFrame:
    con = con or db.conectar(db.PROCESSED)
    return con.sql("SELECT * FROM hillstrom").df()


# ---------- 1. ¿Salió bien el sorteo? ----------

def srm(n: pd.Series) -> float:
    """Sample Ratio Mismatch: p-valor chi² de que los grupos tengan el tamaño esperado (iguales)."""
    return float(stats.chisquare(n.to_numpy()).pvalue)


def balance(d: pd.DataFrame) -> pd.DataFrame:
    """Diferencia estandarizada (SMD) de cada variable previa entre cada brazo y el control. |SMD| < 0,1 = balanceado."""
    x = pd.get_dummies(d[COVARIABLES_NUM + COVARIABLES_CAT], columns=COVARIABLES_CAT, dtype=float)
    c = x[d.segment == CONTROL]
    filas = []
    for t in TRATAMIENTOS:
        tr = x[d.segment == t]
        for col in x.columns:
            sd = np.sqrt((tr[col].var() + c[col].var()) / 2)
            filas.append({"brazo": t, "variable": col, "media_tratado": tr[col].mean(), "media_control": c[col].mean(),
                          "smd": (tr[col].mean() - c[col].mean()) / sd if sd > 0 else 0.0})
    return pd.DataFrame(filas)


# ---------- 2. Potencia ----------

Z = lambda: stats.norm.ppf(1 - ALFA / 2) + stats.norm.ppf(POTENCIA)


def n_por_grupo_media(sd: float, efecto_abs: float) -> int:
    return int(np.ceil(2 * Z() ** 2 * sd ** 2 / efecto_abs ** 2))


def n_por_grupo_proporcion(p: float, lift: float) -> int:
    p2 = p * (1 + lift)
    return int(np.ceil(Z() ** 2 * (p * (1 - p) + p2 * (1 - p2)) / (p2 - p) ** 2))


def mde_media(sd: float, n: int) -> float:
    """Efecto absoluto mínimo detectable con n por grupo."""
    return float(Z() * sd * np.sqrt(2 / n))


def potencia(d: pd.DataFrame) -> pd.DataFrame:
    c = d[d.segment == CONTROL]
    n = int(d.segment.value_counts().min())
    filas = []
    for m in METRICAS:
        media, sd = c[m].mean(), c[m].std()
        for lift in (0.1, 0.2, 0.5, 1.0):
            req = n_por_grupo_proporcion(media, lift) if m != "spend" else n_por_grupo_media(sd, media * lift)
            filas.append({"metrica": m, "base_control": media, "mde_relativo": lift, "n_por_grupo_necesario": req,
                          "n_por_grupo_disponible": n, "alcanza": req <= n})
        filas.append({"metrica": m, "base_control": media, "mde_relativo": mde_media(sd, n) / media,
                      "n_por_grupo_necesario": n, "n_por_grupo_disponible": n, "alcanza": True, "es_mde_alcanzable": True})
    return pd.DataFrame(filas)


# ---------- 3. Efectos ----------

def efecto(y1: np.ndarray, y0: np.ndarray) -> dict[str, float]:
    """Diferencia de medias con Welch (sirve también para proporciones con n grande)."""
    dif = y1.mean() - y0.mean()
    ee = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    z = stats.norm.ppf(1 - ALFA / 2)
    return {"control": y0.mean(), "tratado": y1.mean(), "dif": dif, "ic_inf": dif - z * ee, "ic_sup": dif + z * ee,
            "lift": dif / y0.mean(), "ee": ee, "p_valor": float(2 * stats.norm.sf(abs(dif / ee)))}


def ic_bootstrap(y1, y0, n_boot: int = 2000, semilla: int = SEMILLA) -> tuple[float, float]:
    rng = np.random.default_rng(semilla)
    d = [rng.choice(y1, len(y1)).mean() - rng.choice(y0, len(y0)).mean() for _ in range(n_boot)]
    return float(np.quantile(d, ALFA / 2)), float(np.quantile(d, 1 - ALFA / 2))


def holm(p: list[float]) -> list[float]:
    """p-valores ajustados por Holm (controla el error de familia con varias comparaciones)."""
    p = np.asarray(p)
    orden = np.argsort(p)
    m, ajust, acum = len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(orden):
        acum = max(acum, min(1.0, (m - k) * p[i]))
        ajust[i] = acum
    return ajust.tolist()


def efectos(d: pd.DataFrame) -> pd.DataFrame:
    y0 = {m: d.loc[d.segment == CONTROL, m].to_numpy(float) for m in METRICAS}
    filas = []
    for m in METRICAS:
        for t in TRATAMIENTOS:
            y1 = d.loc[d.segment == t, m].to_numpy(float)
            fila = {"metrica": m, "brazo": t, **efecto(y1, y0[m])}
            if m == "spend":
                fila["ic_boot_inf"], fila["ic_boot_sup"] = ic_bootstrap(y1, y0[m])
            filas.append(fila)
    # ¿Cuál de los dos correos es mejor? (comparación directa, informativa)
    for m in METRICAS:
        y_m = d.loc[d.segment == TRATAMIENTOS[0], m].to_numpy(float)
        y_w = d.loc[d.segment == TRATAMIENTOS[1], m].to_numpy(float)
        filas.append({"metrica": m, "brazo": "Mens vs Womens", **efecto(y_m, y_w)})
    t = pd.DataFrame(filas)
    t["p_holm"] = np.nan
    for m in METRICAS:
        i = (t.metrica == m) & t.brazo.isin(TRATAMIENTOS)
        t.loc[i, "p_holm"] = holm(t.loc[i, "p_valor"].tolist())
    t["principal"] = (t.metrica == METRICAS[0]) & t.brazo.isin(TRATAMIENTOS)
    return t


# ---------- 4. CUPED y ajuste por regresión ----------

def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Y − θ(X − media(X)), θ = cov(Y, X)/var(X). Misma media, menos varianza si X predice Y."""
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


def ajuste_regresion(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Generaliza CUPED a varias variables previas: resta la parte de Y que explican (MCO, con intercepto)."""
    xc = np.column_stack([np.ones(len(x)), x - x.mean(axis=0)])
    beta = np.linalg.lstsq(xc, y, rcond=None)[0]
    return y - xc[:, 1:] @ beta[1:]


def reduccion_varianza(d: pd.DataFrame) -> pd.DataFrame:
    """Para cada métrica y brazo: ancho del IC sin ajustar, con CUPED (history) y con regresión (todas las previas)."""
    x_todas = pd.get_dummies(d[COVARIABLES_NUM + COVARIABLES_CAT], columns=COVARIABLES_CAT, drop_first=True, dtype=float).to_numpy()
    filas = []
    for m in METRICAS:
        y = d[m].to_numpy(float)
        variantes = {"sin_ajuste": y, "cuped_history": cuped(y, d.history.to_numpy(float)),
                     "regresion_previas": ajuste_regresion(y, x_todas)}
        for t in TRATAMIENTOS:
            i1, i0 = (d.segment == t).to_numpy(), (d.segment == CONTROL).to_numpy()
            base = None
            for nombre, v in variantes.items():
                e = efecto(v[i1], v[i0])
                base = base or e["ee"]
                filas.append({"metrica": m, "brazo": t, "metodo": nombre, "dif": e["dif"], "ee": e["ee"],
                              "ic_inf": e["ic_inf"], "ic_sup": e["ic_sup"], "p_valor": e["p_valor"],
                              "reduccion_varianza": 1 - (e["ee"] / base) ** 2,
                              "muestra_equivalente": (base / e["ee"]) ** 2})
    return pd.DataFrame(filas)


# ---------- 5. Mirar antes de tiempo ----------

def constante_obrien_fleming(miradas: int, alfa: float = ALFA, sims: int = 200_000, semilla: int = SEMILLA) -> float:
    """Constante C tal que P(algún |Z_k| > C / sqrt(t_k)) = α sin efecto real (movimiento browniano, por simulación).

    La forma z_{α/2}/sqrt(t) es solo una aproximación: con 20 miradas deja ~7% de falsos positivos.
    """
    rng = np.random.default_rng(semilla)
    t = np.arange(1, miradas + 1) / miradas
    incrementos = rng.standard_normal((sims, miradas)) * np.sqrt(np.diff(np.concatenate([[0], t])))
    z = np.abs(np.cumsum(incrementos, axis=1)) / np.sqrt(t)
    maximo = (z * np.sqrt(t)).max(axis=1)  # max_k |Z_k|·sqrt(t_k) > C  ⇔  algún |Z_k| > C/sqrt(t_k)
    return float(np.quantile(maximo, 1 - alfa))


def limites_obrien_fleming(miradas: int, alfa: float = ALFA) -> np.ndarray:
    """Límites de O'Brien-Fleming: z_k = C / sqrt(t_k). Muy exigentes al principio, cerca de 1,96 al final."""
    t = np.arange(1, miradas + 1) / miradas
    return constante_obrien_fleming(miradas, alfa) / np.sqrt(t)


def simular_peeking(y: np.ndarray, miradas: int = 20, reps: int = 2000, semilla: int = SEMILLA) -> dict[str, float]:
    """A/A: parte al azar el grupo control en dos 'brazos' sin efecto real y mira `miradas` veces mientras llegan datos.

    Devuelve la tasa de falsos positivos si se para al primer p < 0,05 vs. con límites de O'Brien-Fleming.
    """
    rng = np.random.default_rng(semilla)
    lim = limites_obrien_fleming(miradas)
    n = len(y) // 2
    cortes = np.linspace(n / miradas, n, miradas).astype(int)
    fp_ingenuo = fp_obf = fp_final = 0
    for _ in range(reps):
        p = rng.permutation(y)
        a, b = p[:n], p[n:2 * n]
        ca, cb, ca2, cb2 = np.cumsum(a), np.cumsum(b), np.cumsum(a ** 2), np.cumsum(b ** 2)
        k = cortes - 1
        ma, mb = ca[k] / cortes, cb[k] / cortes
        va = (ca2[k] - cortes * ma ** 2) / (cortes - 1)
        vb = (cb2[k] - cortes * mb ** 2) / (cortes - 1)
        z = np.abs(ma - mb) / np.sqrt((va + vb) / cortes + 1e-12)
        fp_ingenuo += np.any(z > stats.norm.ppf(1 - ALFA / 2))
        fp_obf += np.any(z > lim)
        fp_final += z[-1] > stats.norm.ppf(1 - ALFA / 2)
    return {"miradas": miradas, "reps": reps, "fp_mirando_siempre": fp_ingenuo / reps,
            "fp_obrien_fleming": fp_obf / reps, "fp_una_sola_mirada": fp_final / reps}


def secuencial_real(d: pd.DataFrame, metrica: str = "spend", miradas: int = 20, semilla: int = SEMILLA) -> pd.DataFrame:
    """Simula que los clientes llegan en orden aleatorio y muestra z en cada mirada contra el límite O'Brien-Fleming."""
    rng = np.random.default_rng(semilla)
    lim = limites_obrien_fleming(miradas)
    y0 = rng.permutation(d.loc[d.segment == CONTROL, metrica].to_numpy(float))
    filas = []
    for t in TRATAMIENTOS:
        y1 = rng.permutation(d.loc[d.segment == t, metrica].to_numpy(float))
        n = min(len(y0), len(y1))
        for k, corte in enumerate(np.linspace(n / miradas, n, miradas).astype(int)):
            e = efecto(y1[:corte], y0[:corte])
            filas.append({"brazo": t, "mirada": k + 1, "n_por_grupo": corte, "z": e["dif"] / e["ee"],
                          "limite_obf": lim[k], "limite_fijo": stats.norm.ppf(1 - ALFA / 2)})
    return pd.DataFrame(filas)


# ---------- 6. Bayesiano ----------

def bayes_proporcion(x1, n1, x0, n0, draws: int = 200_000, semilla: int = SEMILLA) -> dict[str, float]:
    """Beta(1,1) a priori → Beta a posteriori; P(B > A), mejora esperada y pérdida esperada de elegir B."""
    rng = np.random.default_rng(semilla)
    a = rng.beta(1 + x0, 1 + n0 - x0, draws)
    b = rng.beta(1 + x1, 1 + n1 - x1, draws)
    return {"p_mejor": float((b > a).mean()), "lift_esperado": float(np.mean(b / a - 1)),
            "lift_p05": float(np.quantile(b / a - 1, 0.05)), "lift_p95": float(np.quantile(b / a - 1, 0.95)),
            "perdida_esperada": float(np.mean(np.maximum(a - b, 0)))}


def bayes_media(y1, y0, draws: int = 4000, semilla: int = SEMILLA) -> dict[str, float]:
    """Bootstrap bayesiano (pesos Dirichlet): distribución a posteriori de la diferencia de medias sin suponer normalidad."""
    rng = np.random.default_rng(semilla)
    m1 = np.array([np.dot(rng.dirichlet(np.ones(len(y1))), y1) for _ in range(draws)])
    m0 = np.array([np.dot(rng.dirichlet(np.ones(len(y0))), y0) for _ in range(draws)])
    dif = m1 - m0
    return {"p_mejor": float((dif > 0).mean()), "lift_esperado": float(np.mean(m1 / m0 - 1)),
            "lift_p05": float(np.quantile(m1 / m0 - 1, 0.05)), "lift_p95": float(np.quantile(m1 / m0 - 1, 0.95)),
            "perdida_esperada": float(np.mean(np.maximum(-dif, 0)))}


def bayes(d: pd.DataFrame) -> pd.DataFrame:
    c = d[d.segment == CONTROL]
    filas = []
    for t in TRATAMIENTOS:
        tr = d[d.segment == t]
        for m in ("visit", "conversion"):
            filas.append({"metrica": m, "brazo": t, **bayes_proporcion(tr[m].sum(), len(tr), c[m].sum(), len(c))})
        filas.append({"metrica": "spend", "brazo": t, **bayes_media(tr.spend.to_numpy(float), c.spend.to_numpy(float))})
    return pd.DataFrame(filas)


# ---------- 7. Economía ----------

def economia(ef: pd.DataFrame, margen: float = MARGEN) -> pd.DataFrame:
    g = ef[ef.metrica == "spend"].copy()
    g["gasto_incremental_por_1000"] = g.dif * 1000
    g["ic_inf_por_1000"], g["ic_sup_por_1000"] = g.ic_inf * 1000, g.ic_sup * 1000
    g["costo_max_por_correo"] = g.dif * margen  # sobre esto, el correo pierde plata
    g["costo_max_ic_inf"] = g.ic_inf * margen  # versión pesimista (límite inferior del IC)
    for m in (0.2, 0.4):
        g[f"costo_max_margen_{int(m * 100)}"] = g.dif * m
    g["margen_supuesto"] = margen
    g = g[g.brazo.isin(TRATAMIENTOS)]
    return g[["brazo", "gasto_incremental_por_1000", "ic_inf_por_1000", "ic_sup_por_1000", "costo_max_por_correo",
              "costo_max_ic_inf", "costo_max_margen_20", "costo_max_margen_40", "margen_supuesto"]]


def generar() -> dict[str, pd.DataFrame]:
    d = cargar()
    n = d.segment.value_counts()
    ef = efectos(d)
    salidas = {
        "exp_balance": balance(d).assign(srm_p_valor=srm(n)),
        "exp_potencia": potencia(d),
        "exp_efectos": ef,
        "exp_cuped": reduccion_varianza(d),
        "exp_peeking": pd.DataFrame([simular_peeking(d.loc[d.segment == CONTROL, "spend"].to_numpy(float))]),
        "exp_secuencial": secuencial_real(d),
        "exp_bayes": bayes(d),
        "exp_economia": economia(ef),
    }
    for nombre, df in salidas.items():
        df.to_parquet(db.PROCESSED / f"{nombre}.parquet", compression="zstd", index=False)
    return salidas


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    s = generar()
    b = s["exp_balance"]
    print(f"SRM p = {b.srm_p_valor.iloc[0]:.3f} · |SMD| máximo = {b.smd.abs().max():.3f}")
    for k in ("exp_potencia", "exp_efectos", "exp_cuped", "exp_peeking", "exp_bayes", "exp_economia"):
        print(k); print(s[k].round(4).to_string(index=False)); print()
