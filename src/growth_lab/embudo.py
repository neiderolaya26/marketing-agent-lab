"""Sesión 2 · Inteligencia de mercado: embudo de captación y cohortes de compradores (Olist).

Uso:
    python -m growth_lab.embudo            # ventana de 90 días
    python -m growth_lab.embudo 180

Escribe en data/processed/:
- `embudo_canal.parquet`         valor de un lead por canal = P(cierre) × GMV esperado del contrato
- `cohortes_compradores.parquet` recompra acumulada por mes de primera compra
- `kpis_mensuales.parquet`       pedidos, GMV, AOV, nuevos vs. recurrentes, entregas tarde, reseña

Solo usa tablas publicadas en data/processed, así que se reproduce sin descargar nada.

Supuestos (decisión 002):
- Un pedido cuenta para el GMV si no está cancelado ni indisponible.
- GMV = suma de `price` de los ítems (sin flete), en BRL.
- Un contrato está **maduro** si `won_date + ventana <= CORTE`; solo los maduros entran
  al valor esperado, para no castigar a los canales que cerraron tarde.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd

from growth_lab import db

CORTE = "2018-08-29"  # último pedido de Olist
VENTANA_DIAS = 90
ESTADOS_EXCLUIDOS = ("canceled", "unavailable")

_VENTAS = f"""
    SELECT i.seller_id, o.order_id, o.order_purchase_timestamp AS fecha, i.price
    FROM order_items i JOIN orders o USING (order_id)
    WHERE o.order_status NOT IN {ESTADOS_EXCLUIDOS}
"""


def _sql_leads(ventana_dias: int, corte: str) -> str:
    """Una fila por lead: canal, si cerró, si el contrato está maduro, pedidos y GMV en la ventana."""
    return f"""
    WITH ventas AS ({_VENTAS}),
    contratos AS (
        SELECT d.mql_id, d.seller_id, d.won_date,
               d.won_date + INTERVAL {int(ventana_dias)} DAY <= TIMESTAMP '{corte}' AS maduro
        FROM closed_deals d
    ),
    valor AS (
        SELECT c.mql_id, c.maduro,
               count(DISTINCT v.order_id) AS pedidos,
               coalesce(sum(v.price), 0) AS gmv
        FROM contratos c
        LEFT JOIN ventas v
          ON v.seller_id = c.seller_id
         AND v.fecha >= c.won_date
         AND v.fecha < c.won_date + INTERVAL {int(ventana_dias)} DAY
        GROUP BY 1, 2
    )
    SELECT m.mql_id, coalesce(m.origin, 'sin_dato') AS canal,
           v.mql_id IS NOT NULL AS cerro, coalesce(v.maduro, false) AS maduro,
           coalesce(v.pedidos, 0) AS pedidos, coalesce(v.gmv, 0) AS gmv
    FROM marketing_qualified_leads m
    LEFT JOIN valor v USING (mql_id)
    """


def leads(con: duckdb.DuckDBPyConnection, ventana_dias: int = VENTANA_DIAS, corte: str = CORTE) -> pd.DataFrame:
    return con.sql(_sql_leads(ventana_dias, corte)).df()


def embudo_canal(
    con: duckdb.DuckDBPyConnection,
    ventana_dias: int = VENTANA_DIAS,
    corte: str = CORTE,
    n_boot: int = 2000,
    semilla: int = 0,
) -> pd.DataFrame:
    """Embudo lead → contrato → vendedor que vende, y valor esperado de un lead por canal con IC 90% (bootstrap)."""
    sql = f"""
    WITH l AS ({_sql_leads(ventana_dias, corte)})
    SELECT canal,
           count(*) AS leads,
           count(*) FILTER (cerro) AS cerrados,
           count(*) FILTER (maduro) AS cerrados_maduros,
           count(*) FILTER (maduro AND pedidos > 0) AS maduros_que_venden,
           coalesce(sum(gmv) FILTER (maduro), 0) AS gmv_maduros
    FROM l GROUP BY 1
    """
    t = con.sql(sql).df()
    t["tasa_cierre"] = t.cerrados / t.leads
    t["tasa_venta_si_cierra"] = (t.maduros_que_venden / t.cerrados_maduros).where(t.cerrados_maduros > 0)
    t["gmv_por_contrato"] = (t.gmv_maduros / t.cerrados_maduros).where(t.cerrados_maduros > 0)
    # Propensión × valor: lo que vale un lead de este canal en la ventana.
    t["gmv_por_lead"] = t.tasa_cierre * t.gmv_por_contrato
    ic = {c: ic_gmv_por_lead(g, n_boot, semilla) for c, g in leads(con, ventana_dias, corte).groupby("canal")}
    t["gmv_por_lead_p05"] = t.canal.map(lambda c: ic[c][0])
    t["gmv_por_lead_p95"] = t.canal.map(lambda c: ic[c][1])
    t["ventana_dias"] = ventana_dias
    return t.sort_values("leads", ascending=False, ignore_index=True)


def gmv_por_lead(cerro: np.ndarray, maduro: np.ndarray, gmv: np.ndarray) -> float:
    """P(cierre) × GMV medio de los contratos maduros (NaN si no hay maduros)."""
    if maduro.sum() == 0:
        return float("nan")
    return float(cerro.mean() * gmv[maduro].mean())


def ic_gmv_por_lead(g: pd.DataFrame, n_boot: int = 2000, semilla: int = 0) -> tuple[float, float]:
    """IC 90% del GMV por lead remuestreando leads del canal."""
    rng = np.random.default_rng(semilla)
    cerro, maduro, gmv = g.cerro.to_numpy(), g.maduro.to_numpy(), g.gmv.to_numpy(dtype=float)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    est = np.array([gmv_por_lead(cerro[i], maduro[i], gmv[i]) for i in idx])
    est = est[~np.isnan(est)]
    if len(est) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(est, 0.05)), float(np.quantile(est, 0.95)))


def cohortes_compradores(con: duckdb.DuckDBPyConnection, meses: int = 12, min_compradores: int = 100) -> pd.DataFrame:
    """Por mes de primera compra: compradores y % que volvió a comprar dentro de k meses (acumulado).

    Se omiten las cohortes con menos de `min_compradores` (2016 casi no tiene pedidos).
    """
    sql = f"""
    WITH p AS (
        SELECT c.customer_unique_id AS comprador, o.order_id, o.order_purchase_timestamp AS fecha
        FROM orders o JOIN customers c USING (customer_id)
        WHERE o.order_status NOT IN {ESTADOS_EXCLUIDOS}
    ),
    primera AS (SELECT comprador, min(fecha) AS f0 FROM p GROUP BY 1),
    segunda AS (
        SELECT p.comprador, min(p.fecha) AS f1
        FROM p JOIN primera USING (comprador)
        WHERE p.fecha > primera.f0 + INTERVAL 1 DAY  -- ignora pedidos partidos el mismo día
        GROUP BY 1
    )
    SELECT date_trunc('month', f0)::DATE AS cohorte,
           date_diff('month', date_trunc('month', f0), date_trunc('month', f1)) AS mes_recompra
    FROM primera LEFT JOIN segunda USING (comprador)
    """
    d = con.sql(sql).df()
    filas = []
    fin = pd.Timestamp(CORTE).to_period("M")
    for cohorte, g in d.groupby("cohorte"):
        n = len(g)
        if n < min_compradores:
            continue
        edad = (fin - pd.Timestamp(cohorte).to_period("M")).n  # meses observables
        for k in range(1, meses + 1):
            if k > edad:
                break
            filas.append({"cohorte": cohorte, "compradores": n, "mes": k,
                          "pct_recompra_acum": float((g.mes_recompra <= k).sum() / n)})
    return pd.DataFrame(filas)


def kpis_mensuales(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = f"""
    WITH p AS (
        SELECT o.order_id, c.customer_unique_id AS comprador,
               date_trunc('month', o.order_purchase_timestamp)::DATE AS mes,
               o.order_delivered_customer_date > o.order_estimated_delivery_date AS tarde,
               o.order_status = 'delivered' AS entregado
        FROM orders o JOIN customers c USING (customer_id)
        WHERE o.order_status NOT IN {ESTADOS_EXCLUIDOS}
    ),
    primera AS (SELECT comprador, min(mes) AS mes0 FROM p GROUP BY 1),
    gmv AS (SELECT order_id, sum(price) AS gmv FROM order_items GROUP BY 1),
    resena AS (SELECT order_id, avg(review_score) AS nota FROM order_reviews GROUP BY 1)
    SELECT p.mes,
           count(*) AS pedidos,
           count(DISTINCT p.comprador) AS compradores,
           count(DISTINCT p.comprador) FILTER (p.mes = primera.mes0) AS compradores_nuevos,
           sum(gmv.gmv) AS gmv,
           sum(gmv.gmv) / count(*) AS aov,
           avg(p.tarde::INT) FILTER (p.entregado) AS pct_entrega_tarde,
           avg(resena.nota) AS resena_media
    FROM p JOIN primera USING (comprador)
    LEFT JOIN gmv USING (order_id)
    LEFT JOIN resena USING (order_id)
    GROUP BY 1 ORDER BY 1
    """
    t = con.sql(sql).df()
    t["pct_nuevos"] = t.compradores_nuevos / t.compradores
    return t


def generar(ventana_dias: int = VENTANA_DIAS) -> dict[str, pd.DataFrame]:
    con = db.conectar(db.PROCESSED)
    salidas = {
        "embudo_canal": embudo_canal(con, ventana_dias),
        "cohortes_compradores": cohortes_compradores(con),
        "kpis_mensuales": kpis_mensuales(con),
    }
    for nombre, df in salidas.items():
        df.to_parquet(db.PROCESSED / f"{nombre}.parquet", compression="zstd", index=False)
    return salidas


if __name__ == "__main__":
    ventana = int(sys.argv[1]) if len(sys.argv) > 1 else VENTANA_DIAS
    s = generar(ventana)
    pd.set_option("display.width", 160)
    cols = ["canal", "leads", "tasa_cierre", "cerrados_maduros", "tasa_venta_si_cierra", "gmv_por_contrato",
            "gmv_por_lead", "gmv_por_lead_p05", "gmv_por_lead_p95"]
    print(s["embudo_canal"][cols].round(3).to_string(index=False))
    print(f"\nCohortes: {s['cohortes_compradores'].cohorte.nunique()} · meses KPI: {len(s['kpis_mensuales'])}")
