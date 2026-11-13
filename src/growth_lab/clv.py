"""Sesión 5 · Valor del cliente (CLV) y churn parcial (Complete Journey).

Uso:
    python -m growth_lab.clv

Decisión 006:
- Horizonte de 13 semanas: calibro con ene–sep 2017 y valido con oct–dic 2017.
- Churn = el gasto de oct–dic cae a menos de la mitad del de jul–sep (se va de a poco).
- Compito: BG/NBD + Gamma-Gamma (implementados aquí con scipy) vs. LightGBM vs. "el próximo
  trimestre = el último" (línea base ingenua).

BG/NBD (Fader, Hardie y Lee, 2005): cada hogar compra a un ritmo λ (Gamma(r, α) entre hogares) y
después de cada compra puede "morir" (dejar de ser cliente) con probabilidad p (Beta(a, b)).
Solo usa x (compras repetidas), t_x (semana de la última compra) y T (antigüedad).
Gamma-Gamma (Fader y Hardie, 2013): el gasto medio por compra de cada hogar sigue una Gamma
con parámetros (p, q, v); supone que el gasto no depende de la frecuencia.

Decisión 007: CLV con LightGBM (mejor que el ingenuo por poco, no significativo); churn con el
clasificador LightGBM; la retención se prioriza por hogar: gasto en riesgo = P(churn) × gasto.

Salidas (agregadas, por la licencia): data/processed/clv_metricas, clv_deciles, clv_churn,
clv_parametros, clv_por_segmento. Por hogar: data/interim/clv_hogar.parquet (no se publica).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.metrics import roc_auc_score

from growth_lab import db

INICIO, CORTE, FIN = "2017-01-01", "2017-10-01", "2018-01-01"
CORTE_ENTRENO_ML = "2017-07-01"  # el ML aprende de ene–jun → jul–sep
SEMANAS = 13
UMBRAL_CHURN = 0.5
SEMILLA = 0


# ---------- Datos ----------

def compras_diarias(con, inicio: str = INICIO, fin: str = FIN) -> pd.DataFrame:
    """Una fila por hogar y día con compra (varias canastas el mismo día = una compra)."""
    return con.sql(f"""
        SELECT household_id, CAST(transaction_timestamp AS DATE) AS dia, sum(sales_value) AS gasto,
               sum(retail_disc) AS desc_tienda, sum(coupon_disc) AS desc_cupon
        FROM cj_transactions
        WHERE transaction_timestamp >= TIMESTAMP '{inicio}' AND transaction_timestamp < TIMESTAMP '{fin}'
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df().assign(dia=lambda d: pd.to_datetime(d.dia))


def resumen_rfm(compras: pd.DataFrame, inicio: str, corte: str) -> pd.DataFrame:
    """x, t_x, T (en semanas) y gasto medio de las compras repetidas, para el período [inicio, corte)."""
    c = compras[(compras.dia >= pd.Timestamp(inicio)) & (compras.dia < pd.Timestamp(corte))]
    g = c.groupby("household_id")
    primera, ultima = g.dia.min(), g.dia.max()
    fin = pd.Timestamp(corte)
    r = pd.DataFrame({
        "x": g.size() - 1,
        "t_x": (ultima - primera).dt.days / 7,
        "T": (fin - primera).dt.days / 7,
        "gasto_total": g.gasto.sum(),
    })
    # Gasto medio de las compras repetidas (sin la primera), como pide Gamma-Gamma.
    repetidas = c[c.dia > c.household_id.map(primera)]
    r["gasto_medio_rep"] = repetidas.groupby("household_id").gasto.mean().reindex(r.index)
    return r


def gasto_ventana(compras: pd.DataFrame, desde: str, hasta: str) -> pd.Series:
    c = compras[(compras.dia >= pd.Timestamp(desde)) & (compras.dia < pd.Timestamp(hasta))]
    return c.groupby("household_id").gasto.sum()


# ---------- BG/NBD ----------

def _bgnbd_ll(params: np.ndarray, x, t_x, T) -> np.ndarray:
    r, alpha, a, b = np.exp(params)
    a1 = special.gammaln(r + x) - special.gammaln(r) + r * np.log(alpha)
    a2 = special.gammaln(a + b) + special.gammaln(b + x) - special.gammaln(b) - special.gammaln(a + b + x)
    a3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        a4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-12)) - (r + x) * np.log(alpha + t_x), -np.inf)
    return a1 + a2 + np.logaddexp(a3, a4)


def ajustar_bgnbd(x, t_x, T) -> dict[str, float]:
    x, t_x, T = map(np.asarray, (x, t_x, T))
    res = optimize.minimize(lambda p: -_bgnbd_ll(p, x, t_x, T).sum(), x0=np.log([1.0, 1.0, 1.0, 1.0]),
                            method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
    r, alpha, a, b = np.exp(res.x)
    return {"r": r, "alpha": alpha, "a": a, "b": b, "log_verosimilitud": -res.fun}


def _denominador(p: dict, x, t_x, T):
    r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        termino = np.where(x > 0, (a / np.maximum(b + x - 1, 1e-12)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return 1 + termino


def prob_vivo(p: dict, x, t_x, T) -> np.ndarray:
    """P(el hogar sigue siendo cliente | su historia)."""
    x, t_x, T = map(np.asarray, (x, t_x, T))
    return 1 / _denominador(p, x, t_x, T)


def compras_esperadas(p: dict, t: float, x, t_x, T) -> np.ndarray:
    """E[compras en las próximas t semanas | x, t_x, T] (Fader, Hardie y Lee, 2005, ec. 10)."""
    x, t_x, T = map(np.asarray, (x, t_x, T))
    r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
    z = t / (alpha + T + t)
    hyp = special.hyp2f1(r + x, b + x, a + b + x - 1, z)
    num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
    return num / _denominador(p, x, t_x, T)


# ---------- Gamma-Gamma ----------

def _gg_ll(params, x, m):
    p, q, v = np.exp(params)
    return (special.gammaln(p * x + q) - special.gammaln(p * x) - special.gammaln(q) + q * np.log(v)
            + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(x * m + v))


def ajustar_gamma_gamma(x, m) -> dict[str, float]:
    x, m = np.asarray(x, float), np.asarray(m, float)
    ok = (x > 0) & (m > 0)
    res = optimize.minimize(lambda pr: -_gg_ll(pr, x[ok], m[ok]).sum(), x0=np.log([1.0, 1.0, 1.0]),
                            method="Nelder-Mead", options={"maxiter": 20000})
    p, q, v = np.exp(res.x)
    return {"p": p, "q": q, "v": v}


def gasto_medio_esperado(g: dict, x, m) -> np.ndarray:
    """E[gasto por compra | x, m]; con x = 0 se usa la media poblacional p·v/(q−1)."""
    x, m = np.asarray(x, float), np.nan_to_num(np.asarray(m, float))
    p, q, v = g["p"], g["q"], g["v"]
    return np.where(x > 0, p * (v + x * m) / (p * x + q - 1), p * v / (q - 1))


# ---------- LightGBM ----------

def variables_ml(compras: pd.DataFrame, inicio: str, corte: str) -> pd.DataFrame:
    """Variables por semana (tasas), comparables entre ventanas de distinto largo."""
    r = resumen_rfm(compras, inicio, corte)
    semanas = (pd.Timestamp(corte) - pd.Timestamp(inicio)).days / 7
    c = compras[(compras.dia >= pd.Timestamp(inicio)) & (compras.dia < pd.Timestamp(corte))]
    hace = lambda s: (pd.Timestamp(corte) - pd.Timedelta(weeks=s)).strftime("%Y-%m-%d")
    g13 = gasto_ventana(compras, hace(13), corte)
    g13_previo = gasto_ventana(compras, hace(26), hace(13))
    g4 = gasto_ventana(compras, hace(4), corte)
    v = pd.DataFrame(index=r.index)
    v["recencia_sem"] = r["T"] - r.t_x
    v["antiguedad_sem"] = r["T"]
    v["compras_por_sem"] = (r.x + 1) / semanas
    v["gasto_por_sem"] = r.gasto_total / semanas
    v["ticket"] = r.gasto_total / (r.x + 1)
    v["gasto_13s"] = g13.reindex(r.index).fillna(0)
    v["gasto_4s"] = g4.reindex(r.index).fillna(0)
    v["tendencia_13s"] = (v.gasto_13s + 1) / (g13_previo.reindex(r.index).fillna(0) + 1)
    v["pct_desc_tienda"] = c.groupby("household_id").desc_tienda.sum() / (c.groupby("household_id").gasto.sum() + 1e-9)
    v["pct_cupon"] = c.groupby("household_id").desc_cupon.sum() / (c.groupby("household_id").gasto.sum() + 1e-9)
    v["sd_intervalo_dias"] = c.groupby("household_id").dia.apply(lambda s: s.diff().dt.days.std()).reindex(r.index).fillna(60)
    return v


def modelo_lightgbm():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, n_estimators=400, learning_rate=0.03,
                         num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=SEMILLA, verbose=-1)


# ---------- Evaluación ----------

def metricas_valor(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    orden = np.argsort(-pred, kind="stable")
    top = orden[: int(len(y) * 0.2)]
    return {
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
        "spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
        "sesgo_total": float(pred.sum() / y.sum() - 1),
        "captura_top_20": float(y[top].sum() / y.sum()),
    }


def ic_diferencia_mae(y, p1, p0, n_boot: int = 1000, semilla: int = SEMILLA) -> tuple[float, float, float]:
    """MAE(p1) − MAE(p0) con IC 90% por bootstrap pareado (negativo = p1 es mejor)."""
    rng = np.random.default_rng(semilla)
    e1, e0 = np.abs(y - p1), np.abs(y - p0)
    dif = [np.mean(e1[i] - e0[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.mean(e1 - e0)), float(np.quantile(dif, 0.05)), float(np.quantile(dif, 0.95))


def deciles_valor(y, pred, n: int = 10) -> pd.DataFrame:
    orden = np.argsort(-pred, kind="stable")
    filas = []
    for k, g in enumerate(np.array_split(orden, n), 1):
        filas.append({"decil": k, "hogares": len(g), "pred_media": pred[g].mean(), "real_media": y[g].mean()})
    return pd.DataFrame(filas)


def generar() -> dict[str, pd.DataFrame]:
    con = db.conectar()
    compras = compras_diarias(con)

    # Calibración (ene–sep) y lo que pasó después (oct–dic).
    rfm = resumen_rfm(compras, INICIO, CORTE)
    real = gasto_ventana(compras, CORTE, FIN).reindex(rfm.index).fillna(0).to_numpy()
    q3 = gasto_ventana(compras, "2017-07-01", CORTE).reindex(rfm.index).fillna(0).to_numpy()

    bg = ajustar_bgnbd(rfm.x, rfm.t_x, rfm["T"])
    gg = ajustar_gamma_gamma(rfm.x, rfm.gasto_medio_rep)
    n_compras = compras_esperadas(bg, SEMANAS, rfm.x, rfm.t_x, rfm["T"])
    ticket = gasto_medio_esperado(gg, rfm.x, rfm.gasto_medio_rep)
    pred_bg = n_compras * ticket
    p_vivo = prob_vivo(bg, rfm.x, rfm.t_x, rfm["T"])

    # LightGBM: aprende ene–jun → jul–sep y predice con ene–sep → oct–dic.
    x_tr = variables_ml(compras, INICIO, CORTE_ENTRENO_ML)
    y_tr = gasto_ventana(compras, CORTE_ENTRENO_ML, CORTE).reindex(x_tr.index).fillna(0)
    x_te = variables_ml(compras, INICIO, CORTE).reindex(rfm.index)
    lgbm = modelo_lightgbm().fit(x_tr, y_tr)
    pred_ml = np.clip(lgbm.predict(x_te), 0, None)

    preds = {"ingenuo_ultimo_trimestre": q3, "bgnbd_gamma_gamma": pred_bg, "lightgbm": pred_ml}
    met = []
    for nombre, p in preds.items():
        fila = {"modelo": nombre, **metricas_valor(real, p)}
        fila["dif_mae_vs_ingenuo"], fila["dif_p05"], fila["dif_p95"] = ic_diferencia_mae(real, p, q3)
        met.append(fila)
    met = pd.DataFrame(met)
    dec = pd.concat([deciles_valor(real, p).assign(modelo=n) for n, p in preds.items()], ignore_index=True)

    # Churn parcial: el gasto de oct–dic cae a menos de la mitad del de jul–sep (solo hogares con gasto en jul–sep).
    con_q3 = q3 > 0
    churn = (real < UMBRAL_CHURN * q3)[con_q3].astype(int)
    from lightgbm import LGBMClassifier

    q2 = gasto_ventana(compras, "2017-04-01", CORTE_ENTRENO_ML).reindex(x_tr.index).fillna(0).to_numpy()
    y_churn_tr = (y_tr.to_numpy() < UMBRAL_CHURN * q2)[q2 > 0].astype(int)
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEMILLA, verbose=-1)
    clf.fit(x_tr[q2 > 0], y_churn_tr)
    scores = {
        "bgnbd_prob_no_vivo": 1 - p_vivo[con_q3],
        "bgnbd_caida_esperada": 1 - pred_bg[con_q3] / q3[con_q3],
        "lightgbm_caida_esperada": 1 - pred_ml[con_q3] / q3[con_q3],
        "lightgbm_clasificador": clf.predict_proba(x_te[con_q3])[:, 1],
    }
    churn_met = pd.DataFrame([{"score": n, "auc": roc_auc_score(churn, s), "tasa_churn": churn.mean(),
                               "hogares": int(con_q3.sum())} for n, s in scores.items()])

    hogares = pd.DataFrame({"household_id": rfm.index, "x": rfm.x.values, "t_x": rfm.t_x.values, "T": rfm["T"].values,
                            "gasto_real_13s": real, "gasto_jul_sep": q3, "pred_ingenuo": q3, "pred_bgnbd": pred_bg,
                            "pred_lightgbm": pred_ml, "prob_vivo": p_vivo, "compras_esperadas": n_compras,
                            "ticket_esperado": ticket})
    hogares["churn_parcial"] = np.where(q3 > 0, (real < UMBRAL_CHURN * q3).astype(float), np.nan)
    hogares["prob_churn"] = np.nan
    hogares.loc[con_q3, "prob_churn"] = scores["lightgbm_clasificador"]
    # Decisión 007: priorizar por hogar, no por segmento. Gasto en riesgo = P(churn) × la mitad del gasto de jul–sep.
    hogares["gasto_en_riesgo"] = hogares.prob_churn.fillna(0) * hogares.gasto_jul_sep * (1 - UMBRAL_CHURN)
    importancia = pd.DataFrame({"variable": x_tr.columns, "ganancia": clf.booster_.feature_importance("gain")})
    importancia["pct"] = importancia.ganancia / importancia.ganancia.sum()
    importancia = importancia.sort_values("pct", ascending=False, ignore_index=True)

    # Por segmento (sesión 3), si existe.
    seg_path = db.INTERIM / "segmentos_hogar.parquet"
    por_seg = pd.DataFrame()
    if seg_path.exists():
        seg = pd.read_parquet(seg_path)[["household_id", "segmento"]]
        h = hogares.merge(seg, on="household_id", how="left")
        por_seg = h.groupby("segmento").agg(
            hogares=("household_id", "size"), clv_13s_bgnbd=("pred_bgnbd", "mean"), clv_13s_lightgbm=("pred_lightgbm", "mean"),
            gasto_real_13s=("gasto_real_13s", "mean"), prob_vivo=("prob_vivo", "mean"),
            tasa_churn_parcial=("churn_parcial", "mean"), prob_churn_media=("prob_churn", "mean"),
        ).reset_index()
        # Gasto en riesgo: lo que se perdería en 13 semanas si los que caen dejan la mitad.
        por_seg["gasto_en_riesgo_total"] = por_seg.segmento.map(h.groupby("segmento").gasto_en_riesgo.sum())
        # ¿De qué segmentos son los 20% de hogares con más gasto en riesgo?
        top = h.nlargest(int(len(h) * 0.2), "gasto_en_riesgo")
        por_seg["hogares_en_top20_riesgo"] = por_seg.segmento.map(top.segmento.value_counts()).fillna(0).astype(int)
        por_seg["pct_riesgo_en_top20"] = por_seg.segmento.map(top.groupby("segmento").gasto_en_riesgo.sum()).fillna(0) / h.gasto_en_riesgo.sum()

    params = pd.DataFrame([{"modelo": "bgnbd", **bg}, {"modelo": "gamma_gamma", **gg}])
    corr_fm = float(rfm.loc[rfm.x > 0, ["x", "gasto_medio_rep"]].corr().iloc[0, 1])
    params["corr_frecuencia_gasto"] = corr_fm  # Gamma-Gamma supone ~0

    hogares.to_parquet(db.INTERIM / "clv_hogar.parquet", compression="zstd", index=False)
    salidas = {"clv_metricas": met, "clv_deciles": dec, "clv_churn": churn_met, "clv_parametros": params,
               "clv_por_segmento": por_seg, "clv_churn_importancia": importancia}
    for nombre, df in salidas.items():
        df.to_parquet(db.PROCESSED / f"{nombre}.parquet", compression="zstd", index=False)
    salidas["hogares"] = hogares
    return salidas


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    s = generar()
    for k in ("clv_parametros", "clv_metricas", "clv_churn", "clv_por_segmento", "clv_churn_importancia"):
        print(k); print(s[k].round(4).to_string(index=False)); print()
