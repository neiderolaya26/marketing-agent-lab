"""Sesión 7 · Uplift: a quién contactar para cambiarle la decisión (Hillstrom + Lenta).

Uso:
    python -m growth_lab.uplift

Idea: el experimento de la sesión 6 da el efecto PROMEDIO. El uplift estima el efecto de cada
cliente, τ(x) = E[Y | x, contactado] − E[Y | x, no contactado], para contactar solo a quien le cambia.

Modelos (base LightGBM):
- **S-learner:** un solo modelo con el tratamiento como variable; τ = f(x, 1) − f(x, 0).
- **T-learner:** un modelo para tratados y otro para control; τ = m1(x) − m0(x).
- **X-learner:** parte del T, imputa el efecto individual de cada cliente y lo modela; combina
  ambos lados con la probabilidad de tratamiento (útil si los grupos son de distinto tamaño, como Lenta 75/25).
- **Resultado transformado:** z = y·(t − p)/(p(1 − p)) tiene esperanza τ(x); se modela z directamente.
- **Propensión (contraste):** ordenar por P(compra | contactado). Es lo que se hace sin uplift: elige a
  los que más compran, incluidos los que comprarían igual.

Evaluación (conserva el sorteo). Hillstrom: cross-fitting de 5 pliegues, cada cliente se predice con
modelos que no lo vieron y se evalúa con los 64.000. Lenta: prueba del 30% (687 mil clientes).
- **Curva Qini:** compras (o gasto) EXTRA acumuladas si se contacta al top X% según el modelo.
- **Política personalizada (Hillstrom, 3 brazos):** a cada cliente el correo con mayor uplift, o ninguno;
  su valor se estima sin sesgo con IPW porque la asignación fue aleatoria con probabilidad conocida.

Salidas en data/processed/ (prefijo upl_). De Lenta solo se publican curvas y métricas agregadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from growth_lab import db

SEMILLA = 0
PRUEBA = 0.3
# Decisión 010: política = correo de hombres a todos (el uplift no encuentra diferencias en el gasto);
# modelo de referencia cuando haya presupuesto parcial y señal medible = X-learner.
MODELO_REFERENCIA = "x_learner"
POLITICA_RECOMENDADA = "todos_hombres"
CONTROL_H, TRATAMIENTOS_H = "No E-Mail", ["Mens E-Mail", "Womens E-Mail"]
MARGEN = 0.30  # mismo supuesto que la sesión 6
COSTO_CORREO = 0.01  # USD por correo (supuesto)
FRACCIONES = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def _base(n: int):
    """LightGBM regresor para cualquier y (0/1 o gasto); hojas grandes porque el efecto es una señal débil."""
    return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=max(50, n // 400),
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                         random_state=SEMILLA, verbose=-1)


# ---------- Learners ----------

def s_learner(x, t, y, x_nuevo):
    m = _base(len(y)).fit(np.column_stack([x, t]), y)
    return m.predict(np.column_stack([x_nuevo, np.ones(len(x_nuevo))])) - m.predict(np.column_stack([x_nuevo, np.zeros(len(x_nuevo))]))


def t_learner(x, t, y, x_nuevo):
    m1 = _base(int(t.sum())).fit(x[t == 1], y[t == 1])
    m0 = _base(int((t == 0).sum())).fit(x[t == 0], y[t == 0])
    return m1.predict(x_nuevo) - m0.predict(x_nuevo)


def x_learner(x, t, y, x_nuevo):
    m1 = _base(int(t.sum())).fit(x[t == 1], y[t == 1])
    m0 = _base(int((t == 0).sum())).fit(x[t == 0], y[t == 0])
    d1 = y[t == 1] - m0.predict(x[t == 1])  # efecto imputado de los tratados
    d0 = m1.predict(x[t == 0]) - y[t == 0]  # efecto imputado de los controles
    tau1 = _base(len(d1)).fit(x[t == 1], d1)
    tau0 = _base(len(d0)).fit(x[t == 0], d0)
    e = t.mean()  # P(tratamiento): constante en un experimento
    return e * tau0.predict(x_nuevo) + (1 - e) * tau1.predict(x_nuevo)


def resultado_transformado(x, t, y, x_nuevo):
    p = t.mean()
    z = y * (t - p) / (p * (1 - p))
    return _base(len(y)).fit(x, z).predict(x_nuevo)


def propension(x, t, y, x_nuevo):
    """No es uplift: P(compra | contactado). Sirve para mostrar por qué no basta con 'a quién le va a gustar'."""
    return _base(int(t.sum())).fit(x[t == 1], y[t == 1]).predict(x_nuevo)


LEARNERS = {"s_learner": s_learner, "t_learner": t_learner, "x_learner": x_learner,
            "resultado_transformado": resultado_transformado, "propension": propension}


# ---------- Evaluación ----------

def curva_qini(score: np.ndarray, t: np.ndarray, y: np.ndarray, fracciones=FRACCIONES) -> pd.DataFrame:
    """Ganancia incremental acumulada si se contacta al top f según el score (escalada a toda la muestra).

    En el top f: (media tratados − media control) × número de clientes del top. Al 100% = efecto promedio × N.
    """
    orden = np.argsort(-score, kind="stable")
    t, y = t[orden], y[orden]
    n = len(y)
    filas = [{"fraccion": 0.0, "ganancia": 0.0}]
    for f in fracciones:
        k = max(int(round(f * n)), 2)
        tt, yy = t[:k], y[:k]
        if tt.sum() == 0 or (1 - tt).sum() == 0:
            filas.append({"fraccion": f, "ganancia": np.nan})
            continue
        efecto = yy[tt == 1].mean() - yy[tt == 0].mean()
        filas.append({"fraccion": f, "ganancia": efecto * k})
    return pd.DataFrame(filas)


def coef_qini(curva: pd.DataFrame, n: int | None = None) -> float:
    """Área entre la curva del modelo y la recta del azar, por cliente (unidades del resultado).

    0 = ordena como al azar; > 0 = el ranking concentra el efecto arriba. No se divide por la ganancia
    total porque, cuando el efecto promedio es cercano a 0, esa razón explota.
    """
    total = curva.ganancia.iloc[-1]
    azar = curva.fraccion * total
    area = np.trapezoid((curva.ganancia - azar).to_numpy(float), curva.fraccion.to_numpy(float))
    return float(area / n) if n else float(area)


def ic_qini(score, t, y, n_boot: int = 200, semilla: int = SEMILLA) -> tuple[float, float]:
    rng = np.random.default_rng(semilla)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(coef_qini(curva_qini(score[i], t[i], y[i]), len(y)))
    return float(np.nanquantile(vals, 0.05)), float(np.nanquantile(vals, 0.95))


def evaluar(nombre_datos: str, x_tr, t_tr, y_tr, x_te, t_te, y_te, resultado: str, n_boot: int = 200,
            scores: dict[str, np.ndarray] | None = None):
    curvas, metricas = [], []
    for nombre, learner in LEARNERS.items():
        score = scores[nombre] if scores is not None else learner(x_tr, t_tr, y_tr, x_te)
        c = curva_qini(score, t_te, y_te).assign(datos=nombre_datos, modelo=nombre, resultado=resultado)
        lo, hi = ic_qini(score, t_te, y_te, n_boot)
        top30 = c.loc[np.isclose(c.fraccion, 0.3), "ganancia"].iloc[0]
        metricas.append({"datos": nombre_datos, "resultado": resultado, "modelo": nombre,
                         "qini": coef_qini(c, len(y_te)), "qini_p05": lo, "qini_p95": hi,
                         "ganancia_top30": top30, "ganancia_top30_azar": 0.3 * c.ganancia.iloc[-1],
                         "n": len(y_te)})
        curvas.append(c)
    return pd.concat(curvas, ignore_index=True), pd.DataFrame(metricas)


# ---------- Datos ----------

def hillstrom(con=None) -> pd.DataFrame:
    con = con or db.conectar(db.PROCESSED)
    d = con.sql("SELECT * FROM hillstrom").df()
    return d


def matriz_hillstrom(d: pd.DataFrame) -> np.ndarray:
    x = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                       columns=["zip_code", "channel"], dtype=float)
    return x.to_numpy(float)


def lenta(con=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    con = con or db.conectar()
    d = con.sql("SELECT * EXCLUDE (response_sms) FROM lenta").df()
    t = (d.pop("group") == "test").to_numpy(int)
    y = d.pop("response_att").to_numpy(float)
    d["gender"] = d.gender.map({"Ж": 0.0, "М": 1.0})  # Ж = mujer, М = hombre; el resto queda NaN
    return d.to_numpy(float), t, y


# ---------- Hillstrom: comparación de modelos y política personalizada ----------

K_FOLDS = 5


def pliegues(d: pd.DataFrame) -> np.ndarray:
    """Pliegue de cada cliente (estratificado por brazo) para el cross-fitting."""
    from sklearn.model_selection import StratifiedKFold

    f = np.empty(len(d), int)
    for k, (_, i) in enumerate(StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEMILLA).split(d, d.segment)):
        f[i] = k
    return f


def uplift_fuera_de_pliegue(d: pd.DataFrame, brazo: str, resultado: str, modelo: str, f: np.ndarray) -> np.ndarray:
    """Uplift de `brazo` vs. control para TODOS los clientes, cada uno predicho con modelos que no lo vieron."""
    x, y = matriz_hillstrom(d), d[resultado].to_numpy(float)
    pareja = d.segment.isin([brazo, CONTROL_H]).to_numpy()
    t = (d.segment == brazo).to_numpy(int)
    out = np.empty(len(d))
    for k in range(K_FOLDS):
        tr, te = pareja & (f != k), f == k
        out[te] = LEARNERS[modelo](x[tr], t[tr], y[tr], x[te])
    return out


def evaluar_hillstrom(n_boot: int = 200):
    """Qini de cada modelo con los 64.000 clientes (uplift fuera de pliegue)."""
    d = hillstrom()
    f = pliegues(d)
    curvas, metricas = [], []
    for brazo in TRATAMIENTOS_H:
        pareja = d.segment.isin([brazo, CONTROL_H]).to_numpy()
        b = d[pareja]
        for resultado in ("spend", "visit"):
            scores = {m: uplift_fuera_de_pliegue(d, brazo, resultado, m, f)[pareja] for m in LEARNERS}
            c, m = evaluar(f"hillstrom · {brazo}", None, None, None, None, (b.segment == brazo).to_numpy(int),
                           b[resultado].to_numpy(float), resultado, n_boot, scores=scores)
            curvas.append(c)
            metricas.append(m)
    return pd.concat(curvas, ignore_index=True), pd.concat(metricas, ignore_index=True)


def valor_politica(asignado: np.ndarray, recomendado: np.ndarray, y: np.ndarray, p: float = 1 / 3) -> float:
    """IPW: valor esperado por cliente si se siguiera la política. Solo cuentan los clientes cuyo brazo
    sorteado coincide con el recomendado, repesados por 1/P(brazo)."""
    return float(np.mean(y * (asignado == recomendado) / p))


def politicas_hillstrom(modelo: str = "t_learner", resultado: str = "spend", n_boot: int = 500) -> pd.DataFrame:
    """Compara políticas de envío con los 64.000 clientes (uplift fuera de pliegue + IPW, IC 90% bootstrap)."""
    d = hillstrom()
    f = pliegues(d)
    u = np.column_stack([uplift_fuera_de_pliegue(d, b, resultado, modelo, f) for b in TRATAMIENTOS_H])
    umbral = COSTO_CORREO / MARGEN  # gasto incremental mínimo para que el correo pague su costo
    mejor = np.where(u.max(axis=1) > umbral, np.array(TRATAMIENTOS_H)[u.argmax(axis=1)], CONTROL_H)
    segun_historial = np.where(d.mens.to_numpy() >= d.womens.to_numpy(), TRATAMIENTOS_H[0], TRATAMIENTOS_H[1])
    politicas = {
        "nadie": np.full(len(d), CONTROL_H),
        "todos_hombres": np.full(len(d), TRATAMIENTOS_H[0]),
        "todos_mujeres": np.full(len(d), TRATAMIENTOS_H[1]),
        "segun_historial": segun_historial,
        f"personalizada_{modelo}": mejor,
    }
    asignado, y = d.segment.to_numpy(), d[resultado].to_numpy(float)
    rng = np.random.default_rng(SEMILLA)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    boot = {n: np.array([valor_politica(asignado[i], p[i], y[i]) for i in idx]) for n, p in politicas.items()}
    filas = []
    for nombre, pol in politicas.items():
        v = valor_politica(asignado, pol, y)
        pct_envio = float((pol != CONTROL_H).mean())
        inc = boot[nombre] - boot["nadie"]
        vs_h = boot[nombre] - boot["todos_hombres"]
        filas.append({"politica": nombre, "valor_por_cliente": v,
                      "ic_inf": float(np.quantile(boot[nombre], 0.05)), "ic_sup": float(np.quantile(boot[nombre], 0.95)),
                      "incremental_vs_nadie": v - valor_politica(asignado, politicas["nadie"], y),
                      "inc_ic_inf": float(np.quantile(inc, 0.05)), "inc_ic_sup": float(np.quantile(inc, 0.95)),
                      "vs_todos_hombres": v - valor_politica(asignado, politicas["todos_hombres"], y),
                      "vs_h_ic_inf": float(np.quantile(vs_h, 0.05)), "vs_h_ic_sup": float(np.quantile(vs_h, 0.95)),
                      "pct_con_correo": pct_envio, "pct_hombres": float((pol == TRATAMIENTOS_H[0]).mean()),
                      "margen_neto_por_cliente": v * MARGEN - pct_envio * COSTO_CORREO})
    return pd.DataFrame(filas)


# ---------- Presupuesto ----------

def presupuesto(curvas: pd.DataFrame, datos: str, modelo: str, resultado: str) -> pd.DataFrame:
    """Con presupuesto para contactar al X%: ganancia incremental con el modelo vs. al azar."""
    c = curvas[(curvas.datos == datos) & (curvas.modelo == modelo) & (curvas.resultado == resultado)].copy()
    total = c.ganancia.iloc[-1]
    c["ganancia_al_azar"] = c.fraccion * total
    c["pct_de_la_ganancia_total"] = c.ganancia / total
    return c


def generar(n_boot: int = 200) -> dict[str, pd.DataFrame]:
    curvas_h, met_h = evaluar_hillstrom(n_boot)
    x, t, y = lenta()
    x_tr, x_te, t_tr, t_te, y_tr, y_te = train_test_split(x, t, y, test_size=PRUEBA, random_state=SEMILLA, stratify=t)
    curvas_l, met_l = evaluar("lenta", x_tr, t_tr, y_tr, x_te, t_te, y_te, "response_att", n_boot=100)
    salidas = {
        "upl_qini": pd.concat([curvas_h, curvas_l], ignore_index=True),
        "upl_metricas": pd.concat([met_h, met_l], ignore_index=True),
        "upl_politicas": politicas_hillstrom(),
    }
    for nombre, df in salidas.items():
        df.to_parquet(db.PROCESSED / f"{nombre}.parquet", compression="zstd", index=False)
    return salidas


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    s = generar()
    print(s["upl_metricas"].round(4).to_string(index=False))
    print(s["upl_politicas"].round(4).to_string(index=False))
