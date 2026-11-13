"""Sesión 4 · Lead scoring y calidad del lead (embudo de Olist) + rentabilidad por canal.

Uso:
    python -m growth_lab.leads_modelos 40   # búsqueda de modelos con Optuna (ver ese módulo)
    python -m growth_lab.leads              # score de los leads con el campeón + rentabilidad por canal

Decisión 004:
- Población: leads de 2018 (los contratos del dataset empiezan en dic-2017; en 2017 casi nadie
  aparece como cerrado por cobertura, no por calidad). Los leads de 2017 solo sirven como historia.
- Validación fuera de tiempo: entreno con ene–mar 2018 y pruebo con abr–may 2018.
- Objetivo: P(cierre). Calidad del lead = P(cierre) × GMV por contrato de su canal (sesión 2).
- Costos: CPL de equilibrio (= ingreso por lead) + índice de costo por conversión del benchmark
  de atribución (búsqueda pagada = 1). Orgánica y directo son costo fijo: sin ROI de pauta.

Decisión 005: campeón = regresión logística (empata con LightGBM en la prueba y es la más simple).
Las variables de historia se mantienen y se monitorea su PSI (`leads_estabilidad`).

Variables: solo lo que se sabe cuando llega el lead. Las columnas de `closed_deals` (segmento,
tipo de lead…) solo existen si el lead cerró: usarlas sería fuga de información (leakage).
Las tasas "previas" usan únicamente los cierres ya ocurridos antes de la fecha del lead.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from growth_lab import db, embudo

DESDE = "2018-01-01"
CORTE = "2018-04-01"  # entreno antes, pruebo desde aquí
TOP_LANDINGS = 40
SUAVIZADO = 20  # leads "imaginarios" con la tasa global al calcular tasas previas
SEMILLA = 0

CATEGORICAS = ["canal", "landing", "dia_semana"]
NUMERICAS = ["landing_leads_previos_log", "landing_tasa_previa", "canal_tasa_previa", "landing_edad_dias", "landing_nueva"]
VARIABLES = CATEGORICAS + NUMERICAS

# Benchmark de atribución -> nombres de canal de Olist
MAPA_BENCHMARK = {"direct": "direct_traffic"}
COSTO_FIJO = ("organic_search", "direct_traffic")


def poblacion(con: duckdb.DuckDBPyConnection, desde: str = DESDE) -> pd.DataFrame:
    """Leads desde `desde` con su historia previa, calculada sin mirar el futuro."""
    sql = f"""
    WITH h AS (
        SELECT m.mql_id, m.landing_page_id, coalesce(m.origin, 'sin_dato') AS canal,
               m.first_contact_date AS f, d.won_date AS w
        FROM marketing_qualified_leads m LEFT JOIN closed_deals d USING (mql_id)
    ),
    p AS (
        SELECT h.*, h.w IS NOT NULL AS cerro FROM h WHERE h.f >= TIMESTAMP '{desde}'
    )
    SELECT p.mql_id, p.canal, p.landing_page_id, p.f AS fecha, p.cerro,
        (SELECT count(*) FROM h WHERE h.landing_page_id = p.landing_page_id AND h.f < p.f) AS landing_leads_previos,
        (SELECT count(*) FROM h WHERE h.landing_page_id = p.landing_page_id AND h.f < p.f AND h.w < p.f) AS landing_cierres_conocidos,
        (SELECT min(h.f) FROM h WHERE h.landing_page_id = p.landing_page_id AND h.f < p.f) AS landing_primer_lead,
        (SELECT count(*) FROM h WHERE h.canal = p.canal AND h.f < p.f) AS canal_leads_previos,
        (SELECT count(*) FROM h WHERE h.canal = p.canal AND h.f < p.f AND h.w < p.f) AS canal_cierres_conocidos,
        (SELECT count(*) FROM h WHERE h.f < p.f) AS global_leads_previos,
        (SELECT count(*) FROM h WHERE h.f < p.f AND h.w < p.f) AS global_cierres_conocidos
    FROM p
    ORDER BY p.mql_id
    """
    d = con.sql(sql).df()
    d["dia_semana"] = d.fecha.dt.day_name()
    tasa_global = d.global_cierres_conocidos / d.global_leads_previos.clip(lower=1)
    d["landing_tasa_previa"] = (d.landing_cierres_conocidos + SUAVIZADO * tasa_global) / (d.landing_leads_previos + SUAVIZADO)
    d["canal_tasa_previa"] = (d.canal_cierres_conocidos + SUAVIZADO * tasa_global) / (d.canal_leads_previos + SUAVIZADO)
    d["landing_leads_previos_log"] = np.log1p(d.landing_leads_previos)
    d["landing_edad_dias"] = (d.fecha - d.landing_primer_lead).dt.days.fillna(0)
    d["landing_nueva"] = (d.landing_leads_previos == 0).astype(int)
    return d


def preparar(d: pd.DataFrame, landings: list[str]) -> pd.DataFrame:
    """Landing pages poco frecuentes en el entrenamiento -> 'otra'. Categóricas como `category`."""
    x = d.copy()
    x["landing"] = np.where(x.landing_page_id.isin(landings), x.landing_page_id, "otra")
    for c in CATEGORICAS:
        x[c] = pd.Categorical(x[c].astype(str))
    return x


def particion(d: pd.DataFrame, corte: str = CORTE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entreno < corte <= prueba. Las landings frecuentes se eligen solo con el entrenamiento."""
    entreno = d[d.fecha < pd.Timestamp(corte)]
    landings = entreno.landing_page_id.value_counts().head(TOP_LANDINGS).index.tolist()
    x = preparar(d, landings)
    # Mismas categorías en ambas muestras (los modelos con categóricas nativas lo exigen).
    for c in CATEGORICAS:
        x[c] = pd.Categorical(x[c].astype(str), categories=sorted(x[c].astype(str).unique()))
    return x[x.fecha < pd.Timestamp(corte)].copy(), x[x.fecha >= pd.Timestamp(corte)].copy()


# ---------- Métricas ----------

def ic_auc(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, semilla: int = SEMILLA) -> tuple[float, float]:
    """IC 90% del AUC remuestreando leads."""
    rng = np.random.default_rng(semilla)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.quantile(aucs, 0.05)), float(np.quantile(aucs, 0.95))


def ic_diferencia_auc(y: np.ndarray, p1: np.ndarray, p0: np.ndarray, n_boot: int = 1000,
                      semilla: int = SEMILLA) -> tuple[float, float, float]:
    """AUC(p1) − AUC(p0) con IC 90% por bootstrap pareado (mismos leads en ambos)."""
    rng = np.random.default_rng(semilla)
    dif = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            dif.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
    return float(roc_auc_score(y, p1) - roc_auc_score(y, p0)), float(np.quantile(dif, 0.05)), float(np.quantile(dif, 0.95))


def deciles(y: np.ndarray, p: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Decil 1 = los leads con mayor score. Lift = tasa del decil / tasa global."""
    orden = np.argsort(-p, kind="stable")
    grupos = np.array_split(orden, n)
    tasa_global = y.mean()
    filas, acum = [], 0
    for k, g in enumerate(grupos, 1):
        acum += y[g].sum()
        filas.append({"decil": k, "leads": len(g), "cierres": int(y[g].sum()), "tasa_cierre": y[g].mean(),
                      "p_media": p[g].mean(), "lift": y[g].mean() / tasa_global, "captura_acumulada": acum / y.sum()})
    return pd.DataFrame(filas)


def psi(esperado: pd.Series, real: pd.Series, n: int = 10) -> float:
    """Population Stability Index con cortes por deciles del entrenamiento (> 0,25 = cambio fuerte)."""
    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, n + 1)))
    if len(cortes) < 3:  # variable casi constante (p. ej. binaria): usar sus valores
        cortes = np.unique(np.concatenate([[-np.inf], np.unique(esperado), [np.inf]]))
    cortes[0], cortes[-1] = -np.inf, np.inf
    e = np.histogram(esperado, cortes)[0] / len(esperado)
    r = np.histogram(real, cortes)[0] / len(real)
    e, r = np.clip(e, 1e-4, None), np.clip(r, 1e-4, None)
    return float(np.sum((r - e) * np.log(r / e)))


def estabilidad(entreno: pd.DataFrame, prueba: pd.DataFrame) -> pd.DataFrame:
    filas = [{"variable": c, "media_entreno": entreno[c].mean(), "media_prueba": prueba[c].mean(),
              "psi": psi(entreno[c], prueba[c])} for c in NUMERICAS]
    t = pd.DataFrame(filas)
    t["alerta"] = np.select([t.psi > 0.25, t.psi > 0.10], ["fuerte", "moderada"], "estable")
    return t


def coeficientes_logistica(modelo) -> pd.DataFrame:
    """Coeficientes del campeón logístico, como un scorecard: cuánto suma o resta cada atributo (log-odds)."""
    pre, lr = modelo[0], modelo[-1]
    return (pd.DataFrame({"atributo": pre.get_feature_names_out(), "coeficiente": lr.coef_[0]})
            .assign(variable=lambda d: d.atributo.str.split("__").str[1].str.split("_").str[0])
            .sort_values("coeficiente", ascending=False, ignore_index=True))


# ---------- Rentabilidad por canal ----------

def indice_costo_benchmark(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """Costo por unidad de conversión atribuida (Shapley verdadero), relativo a búsqueda pagada = 1."""
    canales = ["paid_search", "social", "display", "email", "organic_search", "direct", "referral"]
    credito = ", ".join(f"sum(gt_shapley_{c}) AS {c}" for c in canales)
    cred = con.sql(f"SELECT {credito} FROM attr_journeys WHERE converted").df().iloc[0]
    costo = con.sql("SELECT channel, sum(cost) AS costo FROM attr_touchpoints GROUP BY 1").df().set_index("channel").costo
    por_conv = costo / cred
    indice = por_conv / por_conv["paid_search"]
    return indice.rename(lambda c: MAPA_BENCHMARK.get(c, c))


def rentabilidad(con: duckdb.DuckDBPyConnection, comision: float = embudo.COMISION) -> pd.DataFrame:
    """Por canal: valor por lead, CPL de equilibrio y valor por unidad de costo (índice del benchmark)."""
    canal = embudo.embudo_canal(con, comision=comision, n_boot=500)
    t = canal[["canal", "leads", "cerrados_maduros", "gmv_por_lead", "gmv_por_lead_p05", "gmv_por_lead_p95",
               "ingreso_por_lead"]].rename(columns={"ingreso_por_lead": "cpl_equilibrio"})
    t["indice_costo"] = t.canal.map(indice_costo_benchmark(con))
    t["tipo_costo"] = np.where(t.canal.isin(COSTO_FIJO), "fijo", np.where(t.indice_costo.notna(), "por_lead", "sin_dato"))
    pauta = t.tipo_costo == "por_lead"
    # Cuánto GMV trae cada "unidad de costo" frente a búsqueda pagada (= 1).
    t.loc[pauta, "valor_relativo_por_costo"] = t.gmv_por_lead / t.indice_costo
    base = t.loc[t.canal == "paid_search", "valor_relativo_por_costo"].iloc[0]
    t["valor_relativo_por_costo"] = t.valor_relativo_por_costo / base
    return t.sort_values("leads", ascending=False, ignore_index=True)


def generar(campeon: str | None = None) -> dict[str, pd.DataFrame]:
    """Score de todos los leads de 2018 con el campeón (entrenado con ene–mar) + rentabilidad por canal."""
    from growth_lab import leads_modelos as lm

    con = db.conectar(db.PROCESSED)
    entreno, prueba = particion(poblacion(con))
    campeon = campeon or lm.CAMPEON
    modelo = lm.ajustar_final(campeon, entreno)
    todos = pd.concat([entreno, prueba])
    todos["p_cierre"] = modelo.predict_proba(todos[VARIABLES])[:, 1]
    gmv_contrato = embudo.embudo_canal(con, n_boot=10).set_index("canal").gmv_por_contrato
    todos["calidad_gmv_esperado"] = todos.p_cierre * todos.canal.astype(str).map(gmv_contrato)
    todos["muestra"] = np.where(todos.fecha < pd.Timestamp(CORTE), "entreno", "prueba")
    score = todos[["mql_id", "fecha", "canal", "landing", "p_cierre", "calidad_gmv_esperado", "cerro", "muestra"]].copy()
    for c in ("canal", "landing"):
        score[c] = score[c].astype(str)
    salidas = {"leads_score": score.assign(modelo=campeon), "canales_rentabilidad": rentabilidad(con),
               "leads_estabilidad": estabilidad(entreno, prueba)}
    if campeon == "logistica":
        salidas["leads_coeficientes"] = coeficientes_logistica(modelo)
    for nombre, df in salidas.items():
        df.to_parquet(db.PROCESSED / f"{nombre}.parquet", compression="zstd", index=False)
    return salidas


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    s = generar()
    print(s["canales_rentabilidad"].round(3).to_string(index=False))
    print(s["leads_score"].groupby("muestra").p_cierre.describe().round(3))
    print(s["leads_estabilidad"].round(3).to_string(index=False))
    if "leads_coeficientes" in s:
        c = s["leads_coeficientes"]
        print(pd.concat([c.head(6), c.tail(6)]).round(3).to_string(index=False))
