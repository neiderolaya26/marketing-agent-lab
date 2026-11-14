"""Sesión 9 · Atribución multitoque contra la verdad conocida (benchmark sintético, CC BY 4.0).

Uso:
    python -m growth_lab.atribucion

En datos reales no se sabe qué canal "causó" una venta. Este benchmark (500 mil recorridos) se generó
con un proceso conocido y trae, para cada recorrido que convirtió, la atribución Shapley verdadera.
Así se puede medir cuánto se equivoca cada método.

Métodos:
- Reglas por recorrido: último clic, primer clic, lineal, decaimiento temporal (vida media 7 días) y en U
  (40% primero, 40% último, 20% repartido en el medio).
- **Markov** (primer orden): efecto de quitar cada canal de la cadena de transiciones.
- **Shapley con datos** (Zhao et al., 2018): valor de un conjunto de canales = conversiones de los
  recorridos cuyos canales están contenidos en él; Shapley exacto sobre los 7 canales.

Métrica: participación de cada canal en las conversiones (cada recorrido convertido reparte 1), y el
error absoluto medio (MAE, puntos porcentuales) contra la verdad. Todo se publica (prefijo atr_).
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from growth_lab import db

CANALES = ["paid_search", "social", "display", "email", "organic_search", "direct", "referral"]
VENTANA_H, VIDA_MEDIA_H = 720.0, 168.0  # del proceso generador (ground_truth.json)
SEMILLA = 0
METODO_ELEGIDO = "shapley_modelo"  # decisión 013: atribución por modelo de conversión (causa, no presencia)
REGLAS =["ultimo_clic", "primer_clic", "lineal", "decaimiento_temporal", "en_u"]


# ---------- Datos ----------

def cargar(con=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = con or db.conectar(db.PROCESSED)
    j = con.sql("SELECT * FROM attr_journeys ORDER BY journey_id").df()
    t = con.sql("SELECT journey_id, step, channel, t_hours, cost FROM attr_touchpoints ORDER BY journey_id, step").df()
    return j, t


def verdad_por_recorrido(j: pd.DataFrame) -> pd.DataFrame:
    """Shapley verdadero de cada recorrido convertido, normalizado a 1 (misma convención que los métodos)."""
    g = j.loc[j.converted.astype(bool), ["journey_id"] + [f"gt_shapley_{c}" for c in CANALES]].set_index("journey_id")
    g.columns = CANALES
    return g.div(g.sum(axis=1), axis=0)


# ---------- Reglas por recorrido ----------

def pesos_regla(t: pd.DataFrame, regla: str) -> np.ndarray:
    """Peso de cada toque dentro de su recorrido (luego se normaliza a 1 por recorrido)."""
    n = t.groupby("journey_id").step.transform("size").to_numpy()
    pos = t.step.to_numpy()
    if regla == "ultimo_clic":
        return (pos == n - 1).astype(float)
    if regla == "primer_clic":
        return (pos == 0).astype(float)
    if regla == "lineal":
        return np.ones(len(t))
    if regla == "decaimiento_temporal":
        return 0.5 ** ((VENTANA_H - t.t_hours.to_numpy()) / VIDA_MEDIA_H)
    if regla == "en_u":
        medio = np.where(n > 2, 0.2 / np.maximum(n - 2, 1), 0.0)
        extremos = np.where(n == 1, 1.0, np.where(n == 2, 0.5, 0.4))
        return np.where((pos == 0) | (pos == n - 1), extremos, medio)
    raise ValueError(regla)


def credito_regla(t: pd.DataFrame, convertidos: pd.Index, regla: str) -> pd.DataFrame:
    """Crédito por canal de cada recorrido convertido (filas suman 1)."""
    tt = t[t.journey_id.isin(convertidos)].copy()
    tt["w"] = pesos_regla(tt, regla)
    tt["w"] = tt.w / tt.groupby("journey_id").w.transform("sum")
    c = tt.pivot_table(index="journey_id", columns="channel", values="w", aggfunc="sum", fill_value=0.0)
    return c.reindex(columns=CANALES, fill_value=0.0).reindex(convertidos, fill_value=0.0)


# ---------- Markov ----------

def caminos(t: pd.DataFrame, j: pd.DataFrame) -> pd.DataFrame:
    """Recorridos únicos (secuencia de canales) con sus conversiones y no conversiones."""
    seq = t.groupby("journey_id").channel.agg(" > ".join)
    d = pd.DataFrame({"camino": seq}).join(j.set_index("journey_id").converted)
    g = d.groupby("camino").converted.agg(conv="sum", total="size").reset_index()
    g["no_conv"] = g.total - g.conv
    return g


def prob_conversion_markov(c: pd.DataFrame, quitar: str | None = None) -> float:
    """P(conversión) de la cadena de Markov de primer orden; `quitar` manda ese canal a 'no conversión'."""
    estados = ["inicio"] + CANALES
    idx = {s: i for i, s in enumerate(estados)}
    k = len(estados)
    trans = np.zeros((k, k))
    a_conv, a_null = np.zeros(k), np.zeros(k)
    for camino, conv, no_conv in c[["camino", "conv", "no_conv"]].itertuples(index=False):
        pasos = ["inicio"] + camino.split(" > ")
        peso = conv + no_conv
        for a, b in zip(pasos[:-1], pasos[1:]):
            trans[idx[a], idx[b]] += peso
        a_conv[idx[pasos[-1]]] += conv
        a_null[idx[pasos[-1]]] += no_conv
    if quitar is not None:
        q = idx[quitar]
        a_null += trans[:, q]  # quien iba a pasar por el canal quitado, no convierte
        trans[:, q] = 0.0
        trans[q, :], a_conv[q], a_null[q] = 0.0, 0.0, 0.0
    total = trans.sum(axis=1) + a_conv + a_null
    total[total == 0] = 1.0
    q_mat, r = trans / total[:, None], a_conv / total
    p = np.linalg.solve(np.eye(k) - q_mat, r)  # probabilidad de absorber en conversión desde cada estado
    return float(p[0])


def participacion_markov(c: pd.DataFrame) -> pd.Series:
    base = prob_conversion_markov(c)
    efecto = pd.Series({ch: 1 - prob_conversion_markov(c, ch) / base for ch in CANALES})
    return efecto / efecto.sum()


# ---------- Shapley con datos ----------

def participacion_shapley(t: pd.DataFrame, j: pd.DataFrame) -> pd.Series:
    """Zhao et al. (2018): v(S) = conversiones de recorridos cuyo conjunto de canales ⊆ S."""
    bit = {ch: 1 << i for i, ch in enumerate(CANALES)}
    mascara = t.assign(b=t.channel.map(bit)).groupby("journey_id").b.agg(lambda s: int(np.bitwise_or.reduce(s.unique())))
    conv = j.set_index("journey_id").converted.reindex(mascara.index)
    conv_por_conjunto = conv.groupby(mascara.values).sum()
    n = len(CANALES)
    v = np.zeros(1 << n)
    for s in range(1 << n):  # v(S) = Σ_{T ⊆ S} conversiones(T)
        v[s] = sum(c for t_mask, c in conv_por_conjunto.items() if t_mask & ~s == 0)
    phi = {}
    for i, ch in enumerate(CANALES):
        otros = [k for k in range(n) if k != i]
        total = 0.0
        for r in range(n):
            peso = factorial(r) * factorial(n - r - 1) / factorial(n)
            for comb in combinations(otros, r):
                s = sum(1 << k for k in comb)
                total += peso * (v[s | (1 << i)] - v[s])
        phi[ch] = total
    phi = pd.Series(phi)
    return phi / phi.sum()


# ---------- Shapley sobre un modelo de conversión (atribución "basada en datos") ----------

def exposicion(t: pd.DataFrame, ids: pd.Index) -> np.ndarray:
    """Exposición de cada recorrido a cada canal, con decaimiento (vida media 7 días al cierre de la ventana)."""
    w = t.assign(w=0.5 ** ((VENTANA_H - t.t_hours) / VIDA_MEDIA_H))
    x = w.pivot_table(index="journey_id", columns="channel", values="w", aggfunc="sum", fill_value=0.0)
    return x.reindex(index=ids, columns=CANALES, fill_value=0.0).to_numpy(float)


def _modelo_conversion():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    # log(1 + exposición): rendimientos decrecientes de ver muchas veces el mismo canal.
    return make_pipeline(FunctionTransformer(np.log1p), LogisticRegression(max_iter=2000))


def shapley_sobre_modelo(modelo, x: np.ndarray, lote: int = 4000) -> np.ndarray:
    """Shapley exacto de P(conversión) entre los 7 canales para cada fila de x (canal ausente = exposición 0).

    v(S) = P(conv | solo los canales de S). Se normaliza a 1 por recorrido, igual que la verdad.
    """
    n = len(CANALES)
    mascaras = np.array([[(m >> k) & 1 for k in range(n)] for m in range(1 << n)], dtype=float)
    pares = []
    for i in range(n):
        sin_i = [m for m in range(1 << n) if not (m >> i) & 1]
        pesos = np.array([factorial(bin(m).count("1")) * factorial(n - bin(m).count("1") - 1) / factorial(n) for m in sin_i])
        pares.append((np.array(sin_i), np.array(sin_i) | (1 << i), pesos))
    salida = np.zeros_like(x)
    for a in range(0, len(x), lote):
        xb = x[a:a + lote]
        xm = (xb[:, None, :] * mascaras[None, :, :]).reshape(-1, n)
        v = modelo.predict_proba(xm)[:, 1].reshape(len(xb), 1 << n)
        for i, (s, s_i, w) in enumerate(pares):
            salida[a:a + lote, i] = (v[:, s_i] - v[:, s]) @ w
    return salida / salida.sum(axis=1, keepdims=True)


def credito_modelo(t: pd.DataFrame, j: pd.DataFrame, convertidos: pd.Index) -> pd.DataFrame:
    """Cross-fitting de 2 mitades: el modelo que atribuye un recorrido nunca lo vio al entrenar."""
    ids = pd.Index(j.journey_id)
    x = exposicion(t, ids)
    y = j.converted.to_numpy(int)
    mitad = np.random.default_rng(SEMILLA).integers(0, 2, len(ids))
    conv = j.converted.astype(bool).to_numpy()
    credito = np.zeros((len(ids), len(CANALES)))
    for k in (0, 1):
        m = _modelo_conversion().fit(x[mitad != k], y[mitad != k])
        filas = (mitad == k) & conv
        credito[filas] = shapley_sobre_modelo(m, x[filas])
    c = pd.DataFrame(credito, index=ids, columns=CANALES)
    return c.loc[convertidos]


# ---------- Evaluación ----------

def comparar(n_boot: int = 100) -> dict[str, pd.DataFrame]:
    j, t = cargar()
    verdad = verdad_por_recorrido(j)
    convertidos = verdad.index
    participacion = {"verdad": verdad.mean()}
    creditos = {}
    for regla in REGLAS:
        creditos[regla] = credito_regla(t, convertidos, regla)
        participacion[regla] = creditos[regla].mean()
    cam = caminos(t, j)
    participacion["markov"] = participacion_markov(cam)
    participacion["shapley_datos"] = participacion_shapley(t, j)
    creditos["shapley_modelo"] = credito_modelo(t, j, convertidos)
    participacion["shapley_modelo"] = creditos["shapley_modelo"].mean()
    part = pd.DataFrame(participacion).loc[CANALES]

    # MAE contra la verdad, con IC por bootstrap de recorridos convertidos (reglas) o de caminos (Markov).
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for metodo in part.columns.drop("verdad"):
        mae = float((part[metodo] - part.verdad).abs().mean() * 100)
        boot = []
        if metodo in creditos:
            c, v = creditos[metodo].to_numpy(), verdad.to_numpy()
            for _ in range(n_boot):
                i = rng.integers(0, len(c), len(c))
                boot.append(np.abs(c[i].mean(axis=0) - v[i].mean(axis=0)).mean() * 100)
        elif metodo == "markov":
            p = cam.total / cam.total.sum()
            for _ in range(n_boot // 4):
                m = rng.multinomial(int(cam.total.sum()), p)
                tasa = np.divide(cam.conv, cam.total, out=np.zeros(len(cam)), where=cam.total > 0)
                conv_b = rng.binomial(m, tasa)
                cb = cam.assign(conv=conv_b, no_conv=m - conv_b)
                boot.append(float((participacion_markov(cb) - part.verdad).abs().mean() * 100))
        dif = (part[metodo] - part.verdad) * 100
        filas.append({"metodo": metodo, "mae_pp": mae,
                      "mae_p05": float(np.quantile(boot, 0.05)) if boot else np.nan,
                      "mae_p95": float(np.quantile(boot, 0.95)) if boot else np.nan,
                      "canal_mas_sobreatribuido": dif.idxmax(), "sobre_pp": float(dif.max()),
                      "canal_mas_subatribuido": dif.idxmin(), "sub_pp": float(dif.min())})
    error = pd.DataFrame(filas).sort_values("mae_pp", ignore_index=True)

    # Costo por conversión atribuida, por canal y método (solo canales con costo por toque).
    costo = t.groupby("channel").cost.sum().reindex(CANALES)
    n_conv = len(convertidos)
    cpc = pd.DataFrame({m: costo / (part[m] * n_conv) for m in part.columns}).loc[CANALES]
    cpc = cpc[costo > 0]
    salidas = {
        "atr_participacion": part.reset_index(names="canal"),
        "atr_error": error,
        "atr_costo_por_conversion": cpc.reset_index(names="canal"),
    }
    for nombre, df in salidas.items():
        df.to_parquet(db.PROCESSED / f"{nombre}.parquet", compression="zstd", index=False)
    return salidas


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    s = comparar()
    for k, v in s.items():
        print(k); print(v.round(4).to_string(index=False)); print()
