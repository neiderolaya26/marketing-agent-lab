"""Sesión 8 · Efecto de campañas que NO fueron al azar (Complete Journey).

Uso:
    python -m growth_lab.campanas

Decisión 011:
- Campañas tipo A 8 (may–jun), 13 (ago–sep) y 18 (oct–dic), ~1.100 hogares cada una.
- Ventanas: "antes" = las 5 semanas previas al inicio (con 8 se superponía con la campaña tipo A
  anterior); "durante" = lo que dura la campaña. Resultado: gasto semanal promedio del hogar.
- Método principal: DiD doblemente robusto (Sant'Anna y Zhao, 2020): pondera a los no tratados por
  propensión para que se parezcan a los tratados y además ajusta el cambio con una regresión.
  Contraste: DML (modelo parcialmente lineal con LightGBM y cross-fitting). Referencias: ingenuo y DiD simple.
- Placebo: el mismo cálculo corrido 5 semanas antes (el "durante" falso es la ventana previa). Si da
  distinto de 0, el efecto no está identificado y se reporta así.

Salidas agregadas en data/processed/ (prefijo cmp_): efectos, estudio de eventos, soporte común y economía.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from growth_lab import clv, db

CAMPANAS = ["8", "13", "18"]
SEMANAS_PREVIAS = 5
INICIO_HISTORIA = "2017-01-01"
RECORTE = (0.02, 0.98)  # soporte común: se descartan hogares con propensión extrema
MARGEN = 0.30
SEMILLA = 0
N_BOOT = 300


# ---------- Datos ----------

def campanas(con) -> pd.DataFrame:
    return con.sql("SELECT * FROM cj_campaign_descriptions").df().assign(
        start_date=lambda d: pd.to_datetime(d.start_date), end_date=lambda d: pd.to_datetime(d.end_date))


def gasto_semanal(compras: pd.DataFrame, desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.Series:
    """Gasto promedio por semana de cada hogar en [desde, hasta)."""
    c = compras[(compras.dia >= desde) & (compras.dia < hasta)]
    semanas = (hasta - desde).days / 7
    return c.groupby("household_id").gasto.sum() / semanas


def variables_previas(compras: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Historia del hogar antes de `corte`: gasto, frecuencia, recencia, regularidad, descuentos y tendencia."""
    c = compras[(compras.dia >= pd.Timestamp(INICIO_HISTORIA)) & (compras.dia < corte)]
    g = c.groupby("household_id")
    semanas = (corte - pd.Timestamp(INICIO_HISTORIA)).days / 7
    hace = lambda s: corte - pd.Timedelta(weeks=s)
    x = pd.DataFrame({
        "gasto_sem": g.gasto.sum() / semanas,
        "dias_compra_sem": g.size() / semanas,
        "recencia_dias": (corte - g.dia.max()).dt.days,
        "pct_desc_tienda": g.desc_tienda.sum() / (g.gasto.sum() + 1e-9),
        "pct_cupon": g.desc_cupon.sum() / (g.gasto.sum() + 1e-9),
    })
    x["gasto_ult_5s"] = gasto_semanal(compras, hace(5), corte).reindex(x.index).fillna(0)
    x["gasto_5_10s"] = gasto_semanal(compras, hace(10), hace(5)).reindex(x.index).fillna(0)
    x["tendencia"] = x.gasto_ult_5s - x.gasto_5_10s
    x["log_gasto_sem"] = np.log1p(x.gasto_sem)
    return x


def panel(con, compras, campana: str, desplazamiento_semanas: int = 0) -> pd.DataFrame:
    """Una fila por hogar: tratado, gasto antes y durante, variables previas.

    Con `desplazamiento_semanas` > 0 se arma el placebo: todo se corre hacia atrás.
    """
    d = campanas(con).set_index("campaign_id").loc[campana]
    corrimiento = pd.Timedelta(weeks=desplazamiento_semanas)
    inicio = d.start_date - corrimiento
    fin = d.end_date + pd.Timedelta(days=1) - corrimiento
    if desplazamiento_semanas:  # el "durante" falso dura lo mismo que la ventana previa real
        fin = inicio + pd.Timedelta(weeks=SEMANAS_PREVIAS)
    antes_desde = inicio - pd.Timedelta(weeks=SEMANAS_PREVIAS)
    x = variables_previas(compras, antes_desde)  # historia hasta ANTES de la ventana previa
    x = x[x.gasto_sem > 0]
    tratados = set(con.sql(f"SELECT household_id FROM cj_campaigns WHERE campaign_id = '{campana}'").df().household_id)
    p = x.copy()
    p["tratado"] = p.index.isin(tratados).astype(int)
    p["antes"] = gasto_semanal(compras, antes_desde, inicio).reindex(p.index).fillna(0)
    p["durante"] = gasto_semanal(compras, inicio, fin).reindex(p.index).fillna(0)
    p["cambio"] = p.durante - p.antes
    p.attrs.update(campana=campana, inicio=inicio, fin=fin, antes_desde=antes_desde, placebo=desplazamiento_semanas > 0)
    return p


COVARIABLES = ["log_gasto_sem", "dias_compra_sem", "recencia_dias", "pct_desc_tienda", "pct_cupon",
               "gasto_ult_5s", "gasto_5_10s", "tendencia"]


# ---------- Estimadores ----------

def propension(p: pd.DataFrame) -> np.ndarray:
    x = StandardScaler().fit_transform(p[COVARIABLES])
    return LogisticRegression(max_iter=2000, C=1.0).fit(x, p.tratado).predict_proba(x)[:, 1]


def ingenuo(p: pd.DataFrame, w=None) -> float:
    return float(p.loc[p.tratado == 1, "durante"].mean() - p.loc[p.tratado == 0, "durante"].mean())


def did_simple(p: pd.DataFrame) -> float:
    return float(p.loc[p.tratado == 1, "cambio"].mean() - p.loc[p.tratado == 0, "cambio"].mean())


def did_doble_robusto(p: pd.DataFrame, ps: np.ndarray) -> float:
    """ATT = media en tratados de (Δ − m(X)) − media ponderada en controles de (Δ − m(X)), pesos ps/(1 − ps).

    m(X) = regresión lineal del cambio en los controles. Es consistente si acierta el modelo de propensión
    O el de resultado (por eso "doblemente robusto").
    """
    t = p.tratado.to_numpy() == 1
    x = StandardScaler().fit_transform(p[COVARIABLES])
    m = LinearRegression().fit(x[~t], p.cambio.to_numpy()[~t]).predict(x)
    r = p.cambio.to_numpy() - m
    w = ps[~t] / (1 - ps[~t])
    return float(r[t].mean() - np.sum(w * r[~t]) / np.sum(w))


def dml(p: pd.DataFrame, k: int = 5) -> tuple[float, float]:
    """Modelo parcialmente lineal: θ de regresar (Y − E[Y|X]) sobre (D − E[D|X]), con cross-fitting.

    Y = gasto semanal durante la campaña; X incluye el gasto previo. Devuelve (θ, error estándar).
    """
    x = p[COVARIABLES + ["antes"]].to_numpy(float)
    y, d = p.durante.to_numpy(float), p.tratado.to_numpy(float)
    ry, rd = np.empty(len(y)), np.empty(len(y))
    for tr, te in KFold(k, shuffle=True, random_state=SEMILLA).split(x):
        my = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                           random_state=SEMILLA, verbose=-1).fit(x[tr], y[tr])
        md = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                            random_state=SEMILLA, verbose=-1).fit(x[tr], d[tr])
        ry[te] = y[te] - my.predict(x[te])
        rd[te] = d[te] - md.predict_proba(x[te])[:, 1]
    theta = np.sum(rd * ry) / np.sum(rd ** 2)
    psi = (ry - theta * rd) * rd
    ee = np.sqrt(np.mean(psi ** 2) / np.mean(rd ** 2) ** 2 / len(y))
    return float(theta), float(ee)


def estimar(p: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Todos los estimadores para un panel, con IC 95% (bootstrap por hogar; DML analítico)."""
    ps = propension(p)
    dentro = (ps > RECORTE[0]) & (ps < RECORTE[1])
    q, ps_q = p[dentro], ps[dentro]
    puntos = {"ingenuo": ingenuo(q), "did_simple": did_simple(q), "did_doble_robusto": did_doble_robusto(q, ps_q)}
    rng = np.random.default_rng(SEMILLA)
    boot = {k: [] for k in puntos}
    for _ in range(n_boot):
        i = rng.integers(0, len(q), len(q))
        b, psb = q.iloc[i], ps_q[i]
        boot["ingenuo"].append(ingenuo(b))
        boot["did_simple"].append(did_simple(b))
        boot["did_doble_robusto"].append(did_doble_robusto(b, psb))
    filas = [{"metodo": k, "efecto": v, "ic_inf": float(np.quantile(boot[k], 0.025)),
              "ic_sup": float(np.quantile(boot[k], 0.975))} for k, v in puntos.items()]
    theta, ee = dml(q)
    filas.append({"metodo": "dml", "efecto": theta, "ic_inf": theta - 1.96 * ee, "ic_sup": theta + 1.96 * ee})
    t = pd.DataFrame(filas)
    base = q.loc[q.tratado == 1, "antes"].mean()
    t["efecto_pct"] = t.efecto / base
    t["gasto_semanal_tratados_antes"] = base
    t["hogares"], t["tratados"] = len(q), int(q.tratado.sum())
    t["recortados_por_soporte"] = int((~dentro).sum())
    t["tratados_recortados"] = int(((~dentro) & (p.tratado == 1)).sum())
    return t


# ---------- Estudio de eventos ----------

def estudio_eventos(con, compras, campana: str, semanas_antes: int = 15) -> pd.DataFrame:
    """Brecha semanal tratados − controles ponderados (pesos de propensión), relativa a la semana −1."""
    p = panel(con, compras, campana)
    ps = propension(p)
    dentro = (ps > RECORTE[0]) & (ps < RECORTE[1])
    p, ps = p[dentro], ps[dentro]
    w = np.where(p.tratado == 1, 1.0, ps / (1 - ps))
    inicio, fin = p.attrs["inicio"], p.attrs["fin"]
    semanas_despues = int(np.ceil((fin - inicio).days / 7))
    filas = []
    for k in range(-semanas_antes, semanas_despues):
        desde = inicio + pd.Timedelta(weeks=k)
        g = gasto_semanal(compras, desde, desde + pd.Timedelta(weeks=1)).reindex(p.index).fillna(0).to_numpy()
        t = p.tratado.to_numpy() == 1
        filas.append({"campana": campana, "semana": k,
                      "brecha": g[t].mean() - np.average(g[~t], weights=w[~t])})
    e = pd.DataFrame(filas)
    # Relativa al promedio de las semanas −5 a −1 (la ventana "antes" del DiD); una sola semana es muy ruidosa.
    e["brecha_relativa"] = e.brecha - e.loc[e.semana.between(-SEMANAS_PREVIAS, -1), "brecha"].mean()
    return e


# ---------- Economía ----------

def economia(con, compras, p: pd.DataFrame, efecto: float, margen: float = MARGEN) -> dict[str, float]:
    """Margen incremental de la campaña vs. el descuento de cupones que se llevaron los tratados."""
    inicio, fin = p.attrs["inicio"], p.attrs["fin"]
    semanas = (fin - inicio).days / 7
    tratados = p.index[p.tratado == 1]
    c = compras[(compras.dia >= inicio) & (compras.dia < fin) & compras.household_id.isin(tratados)]
    canjes = con.sql(f"SELECT count(*) AS n FROM cj_coupon_redemptions WHERE campaign_id = '{p.attrs['campana']}'").df().n[0]
    incremental = efecto * semanas * len(tratados)
    descuento = float(c.desc_cupon.abs().sum())
    # Efecto mínimo (USD por hogar y semana) para que el margen incremental pague los cupones.
    equilibrio = descuento / (margen * semanas * len(tratados))
    return {"semanas": semanas, "tratados": len(tratados), "gasto_incremental_total": incremental,
            "margen_incremental": incremental * margen, "descuento_cupones_total": descuento,
            "efecto_equilibrio_semanal": equilibrio, "canjes": int(canjes), "tasa_canje_por_hogar": canjes / len(tratados),
            "hogares_control_necesarios": tamano_control(p, equilibrio)}


def tamano_control(p: pd.DataFrame, efecto: float, fraccion_control: float = 0.15, alfa: float = 0.05,
                   potencia: float = 0.80) -> int:
    """Hogares de control sorteados para detectar `efecto` en el cambio de gasto semanal (80% de potencia).

    Con un grupo de control al azar se compara el cambio (durante − antes) de tratados y control, como CUPED.
    Con k = control/tratados, Var(dif) = σ²(1/n_t + 1/n_c) y n_c = k·n_t, así que
    n_control = (z_α/2 + z_β)² σ² (1 + k) / δ²  (y la campaña completa necesita n_control / k tratados).
    """
    from scipy import stats

    sd = p.loc[p.tratado == 1, "cambio"].std()
    z = stats.norm.ppf(1 - alfa / 2) + stats.norm.ppf(potencia)
    k = fraccion_control / (1 - fraccion_control)
    return int(np.ceil(z ** 2 * sd ** 2 * (1 + k) / efecto ** 2))


def combinar(efectos: pd.DataFrame) -> pd.DataFrame:
    """Promedio de las campañas ponderado por precisión (inverso de la varianza), por método y ventana.

    El error estándar de cada campaña sale del ancho de su IC 95% (≈ 2 × 1,96 × EE).
    """
    filas = []
    for (metodo, ventana), g in efectos.groupby(["metodo", "ventana"]):
        ee = (g.ic_sup - g.ic_inf) / (2 * 1.96)
        w = 1 / ee ** 2
        est = float(np.sum(w * g.efecto) / np.sum(w))
        ee_c = float(np.sqrt(1 / np.sum(w)))
        base = float(np.average(g.gasto_semanal_tratados_antes, weights=g.tratados))
        filas.append({"campana": "combinado", "ventana": ventana, "metodo": metodo, "efecto": est,
                      "ic_inf": est - 1.96 * ee_c, "ic_sup": est + 1.96 * ee_c, "efecto_pct": est / base,
                      "gasto_semanal_tratados_antes": base, "hogares": int(g.hogares.sum()), "tratados": int(g.tratados.sum())})
    return pd.DataFrame(filas)


def generar() -> dict[str, pd.DataFrame]:
    con = db.conectar()
    compras = clv.compras_diarias(con, "2016-06-01", "2018-01-01")
    efectos, eventos, econ = [], [], []
    for c in CAMPANAS:
        for placebo in (False, True):
            p = panel(con, compras, c, SEMANAS_PREVIAS if placebo else 0)
            e = estimar(p).assign(campana=c, ventana="placebo" if placebo else "real",
                                  inicio=p.attrs["inicio"], fin=p.attrs["fin"])
            efectos.append(e)
            if not placebo:
                principal = e.set_index("metodo").loc["did_doble_robusto", "efecto"]
                econ.append({"campana": c, **economia(con, compras, p, principal)})
        eventos.append(estudio_eventos(con, compras, c))
    efectos = pd.concat(efectos, ignore_index=True)
    efectos = pd.concat([efectos, combinar(efectos)], ignore_index=True)
    salidas = {"cmp_efectos": efectos,
               "cmp_evento": pd.concat(eventos, ignore_index=True),
               "cmp_economia": pd.DataFrame(econ)}
    for nombre, df in salidas.items():
        df.to_parquet(db.PROCESSED / f"{nombre}.parquet", compression="zstd", index=False)
    return salidas


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    s = generar()
    cols = ["campana", "ventana", "metodo", "efecto", "ic_inf", "ic_sup", "efecto_pct", "hogares", "tratados",
            "recortados_por_soporte", "tratados_recortados"]
    print(s["cmp_efectos"][cols].round(3).to_string(index=False))
    print(s["cmp_economia"].round(2).to_string(index=False))
