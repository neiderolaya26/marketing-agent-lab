"""Sesión 3 · Segmentación de clientes: RFM y clustering con validación fuera de tiempo (Complete Journey).

Uso:
    python -m growth_lab.segmentacion

Diseño:
- **Calibración** (ene–sep 2017): con estas compras se calculan las variables y los segmentos.
- **Validación** (oct–dic 2017): ¿el segmento predice el gasto y la actividad futura?
  Una segmentación útil para el presupuesto separa hogares que *luego* valen distinto.

Salidas:
- `data/interim/segmentos_hogar.parquet`   una fila por hogar (NO se publica: licencia de dunnhumby)
- `data/processed/segmentos_perfil.parquet` perfil agregado por segmento (≥ 20 hogares por fila)
- `data/processed/segmentos_comparacion.parquet` RFM vs. K-means vs. GMM por k

Elegida (decisión 003): K-means sobre RFM con k = 4 → Núcleo, Regulares, Enfriándose, Dormidos.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from growth_lab import db

INICIO = "2017-01-01"
CORTE = "2017-10-01"  # fin de la calibración, inicio de la validación
FIN = "2018-01-01"
MIN_HOGARES_PUBLICABLE = 20
SEMILLA = 0

FRESCOS = ("PRODUCE", "MEAT", "MEAT-PCKGD", "SEAFOOD", "SEAFOOD-PCKGD", "DELI", "PASTRY")

VARIABLES_RFM = ["recencia_dias", "frecuencia", "monto"]
VARIABLES_COMPORTAMIENTO = ["ticket_medio", "pct_desc_tienda", "pct_cupon", "pct_frescos", "pct_drug_gm", "pct_semanas_activas"]


def variables_hogar(con: duckdb.DuckDBPyConnection, inicio: str = INICIO, corte: str = CORTE, fin: str = FIN) -> pd.DataFrame:
    """Variables de calibración y resultado de validación por hogar."""
    frescos = ", ".join(f"'{d}'" for d in FRESCOS)
    sql = f"""
    WITH t AS (
        SELECT t.household_id, t.basket_id, t.transaction_timestamp AS ts, t.week,
               t.sales_value, t.retail_disc, t.coupon_disc, p.department
        FROM cj_transactions t LEFT JOIN cj_products p USING (product_id)
        WHERE t.transaction_timestamp >= TIMESTAMP '{inicio}' AND t.transaction_timestamp < TIMESTAMP '{fin}'
    ),
    cal AS (
        SELECT household_id,
               date_diff('day', max(ts), TIMESTAMP '{corte}') AS recencia_dias,
               count(DISTINCT basket_id) AS frecuencia,
               sum(sales_value) AS monto,
               sum(sales_value) / count(DISTINCT basket_id) AS ticket_medio,
               sum(retail_disc) / nullif(sum(sales_value) + sum(retail_disc), 0) AS pct_desc_tienda,
               sum(coupon_disc) / nullif(sum(sales_value) + sum(coupon_disc), 0) AS pct_cupon,
               sum(sales_value) FILTER (department IN ({frescos})) / nullif(sum(sales_value), 0) AS pct_frescos,
               sum(sales_value) FILTER (department = 'DRUG GM') / nullif(sum(sales_value), 0) AS pct_drug_gm,
               count(DISTINCT week) / (date_diff('day', TIMESTAMP '{inicio}', TIMESTAMP '{corte}') / 7.0) AS pct_semanas_activas
        FROM t WHERE ts < TIMESTAMP '{corte}'
        GROUP BY 1
    ),
    val AS (
        SELECT household_id, sum(sales_value) AS ventas_futuras, count(DISTINCT basket_id) AS canastas_futuras
        FROM t WHERE ts >= TIMESTAMP '{corte}' GROUP BY 1
    )
    SELECT cal.*, coalesce(val.ventas_futuras, 0) AS ventas_futuras,
           coalesce(val.canastas_futuras, 0) AS canastas_futuras,
           val.household_id IS NOT NULL AS activo_futuro
    FROM cal LEFT JOIN val USING (household_id)
    ORDER BY household_id
    """
    d = con.sql(sql).df()
    return d.fillna({c: 0.0 for c in ["pct_desc_tienda", "pct_cupon", "pct_frescos", "pct_drug_gm"]})


# ---------- RFM clásico ----------

def puntajes_rfm(d: pd.DataFrame) -> pd.DataFrame:
    """Quintiles 1–5 (5 = mejor). La recencia se invierte: menos días = mejor."""
    r = pd.qcut(d.recencia_dias.rank(method="first", ascending=False), 5, labels=False) + 1
    f = pd.qcut(d.frecuencia.rank(method="first"), 5, labels=False) + 1
    m = pd.qcut(d.monto.rank(method="first"), 5, labels=False) + 1
    return pd.DataFrame({"R": r.astype(int), "F": f.astype(int), "M": m.astype(int)}, index=d.index)


def segmento_rfm(R: int, F: int, M: int) -> str:
    """Mapa clásico de segmentos RFM (Recencia y Frecuencia mandan; Monto desempata)."""
    if R >= 4 and F >= 4:
        return "Campeones"
    if F >= 4:
        return "Leales en riesgo" if R <= 2 else "Leales"
    if R >= 4 and F <= 2:
        return "Recientes de bajo uso"
    if R <= 2 and F <= 2:
        return "Hibernando"
    if R <= 2:
        return "En riesgo"
    return "Potenciales" if M >= 3 else "Ocasionales"


# ---------- Clustering ----------

def matriz(d: pd.DataFrame, variables: list[str]) -> np.ndarray:
    """log1p a las variables de escala (R, F, M, ticket) y estandarización."""
    x = d[variables].astype(float).copy()
    for c in ("recencia_dias", "frecuencia", "monto", "ticket_medio"):
        if c in x:
            x[c] = np.log1p(x[c])
    return StandardScaler().fit_transform(x)


def eta2(etiquetas: pd.Series, y: pd.Series) -> float:
    """Proporción de la varianza de `y` explicada por el segmento (R² de un ANOVA de una vía)."""
    media = y.mean()
    entre = y.groupby(etiquetas).apply(lambda g: len(g) * (g.mean() - media) ** 2).sum()
    total = ((y - media) ** 2).sum()
    return float(entre / total) if total else float("nan")


def comparar(d: pd.DataFrame, ks: range = range(3, 9)) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Compara RFM y clustering por cuánto explican del gasto futuro (eta²) y separación (silhouette)."""
    filas, etiquetas = [], {}
    y_ventas, y_activo = np.log1p(d.ventas_futuras), d.activo_futuro.astype(float)

    rfm = puntajes_rfm(d)
    etiquetas["rfm"] = pd.Series([segmento_rfm(*v) for v in rfm.itertuples(index=False)], index=d.index)
    filas.append({"metodo": "rfm", "variables": "rfm", "k": etiquetas["rfm"].nunique(), "silhouette": np.nan,
                  "bic": np.nan, "eta2_ventas_futuras": eta2(etiquetas["rfm"], y_ventas),
                  "eta2_activo_futuro": eta2(etiquetas["rfm"], y_activo)})

    for nombre, variables in (("rfm", VARIABLES_RFM), ("rfm+comportamiento", VARIABLES_RFM + VARIABLES_COMPORTAMIENTO)):
        x = matriz(d, variables)
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(x)
            gm = GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=SEMILLA).fit(x)
            for metodo, lab, bic in (("kmeans", km.labels_, np.nan), ("gmm", gm.predict(x), gm.bic(x))):
                s = pd.Series(lab, index=d.index)
                etiquetas[f"{metodo}|{nombre}|{k}"] = s
                filas.append({"metodo": metodo, "variables": nombre, "k": k,
                              "silhouette": silhouette_score(x, lab, random_state=SEMILLA) if s.nunique() > 1 else np.nan,
                              "bic": bic, "eta2_ventas_futuras": eta2(s, y_ventas), "eta2_activo_futuro": eta2(s, y_activo)})
    return pd.DataFrame(filas), etiquetas


NOMBRES_K4 = ["Núcleo", "Regulares", "Enfriándose", "Dormidos"]


def nombrar(d: pd.DataFrame, etiquetas: pd.Series, nombres: list[str] = NOMBRES_K4) -> pd.Series:
    """Nombra los clusters por monto mediano descendente, así el nombre no depende del número que asigne K-means."""
    orden = d.groupby(etiquetas.values).monto.median().sort_values(ascending=False).index
    if len(orden) != len(nombres):
        raise ValueError(f"{len(orden)} clusters y {len(nombres)} nombres")
    return etiquetas.map(dict(zip(orden, nombres)))


def perfil(d: pd.DataFrame, segmento: pd.Series, demografia: pd.DataFrame | None = None) -> pd.DataFrame:
    """Perfil agregado por segmento (publicable si cada fila tiene ≥ MIN_HOGARES_PUBLICABLE hogares)."""
    x = d.assign(segmento=segmento.values)
    agg = x.groupby("segmento").agg(
        hogares=("household_id", "size"),
        **{c: (c, "median") for c in VARIABLES_RFM + VARIABLES_COMPORTAMIENTO},
        ventas_futuras_media=("ventas_futuras", "mean"),
        pct_activo_futuro=("activo_futuro", "mean"),
    )
    agg["pct_hogares"] = agg.hogares / agg.hogares.sum()
    agg["pct_ventas_futuras"] = x.groupby("segmento").ventas_futuras.sum() / x.ventas_futuras.sum()
    if demografia is not None:
        dem = x.merge(demografia, on="household_id", how="left")
        agg["pct_con_demografia"] = dem.groupby("segmento").age.apply(lambda s: s.notna().mean())
        # Solo con ≥ MIN_HOGARES_PUBLICABLE hogares con demografía: con menos, el % no es publicable ni estable.
        agg["pct_con_hijos"] = dem.groupby("segmento").kids_count.apply(
            lambda s: (s.dropna().astype(str) != "0").mean() if s.notna().sum() >= MIN_HOGARES_PUBLICABLE else np.nan)
    return agg.reset_index().sort_values("ventas_futuras_media", ascending=False, ignore_index=True)


def generar(metodo: str = "kmeans", k: int | None = 4, variables: str = "rfm") -> dict[str, pd.DataFrame]:
    """Calcula todo y guarda las salidas con la segmentación elegida.

    Decisión 003: K-means sobre RFM (log + estandarizado) con k = 4; el comportamiento solo describe.
    """
    con = db.conectar()
    d = variables_hogar(con)
    comp, etiquetas = comparar(d)
    clave = "rfm" if metodo == "rfm" else f"{metodo}|{variables}|{k}"
    seg = etiquetas[clave]
    if metodo != "rfm" and k == len(NOMBRES_K4):
        seg = nombrar(d, seg)
    demo = con.sql("SELECT household_id, age, kids_count FROM cj_demographics").df()
    perf = perfil(d, seg.astype(str), demo)
    assert perf.hogares.min() >= MIN_HOGARES_PUBLICABLE, "hay segmentos demasiado chicos para publicar"

    hogares = d.assign(segmento=seg.astype(str).values)
    hogares.to_parquet(db.INTERIM / "segmentos_hogar.parquet", compression="zstd", index=False)
    perf.assign(metodo=clave).to_parquet(db.PROCESSED / "segmentos_perfil.parquet", compression="zstd", index=False)
    comp.to_parquet(db.PROCESSED / "segmentos_comparacion.parquet", compression="zstd", index=False)
    return {"hogares": hogares, "perfil": perf, "comparacion": comp}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    s = generar()
    print(s["comparacion"].round(3).to_string(index=False))
    print(s["perfil"].round(3).to_string(index=False))
