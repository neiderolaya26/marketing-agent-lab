"""Sesión 10 · Marketing Mix Modeling con verdad conocida (generador sintético propio).

Uso:
    python -m growth_lab.mmm

No hay un MMM público con efectos verdaderos publicados (el benchmark de Heusch 2026 no libera datos),
así que se generan 156 semanas de ventas y gasto en 5 canales con un proceso conocido:

    ventas_t = base + tendencia + estacionalidad + demanda oculta_t
               + Σ_c β_c · (1 − exp(−adstock_c,t / λ_c)) + ruido_t
    adstock_c,t = (1 − θ_c) · Σ_k θ_c^k · gasto_c,t−k        (arrastre geométrico, en unidades de gasto)

Dos escenarios (decisión 014): **limpio** (el gasto no mira la demanda) y **endógeno moderado** (se gasta
más cuando se espera vender más, incluida una demanda que el modelo no ve).

MMM: para cada (θ, λ) el modelo es lineal en β y en la base, así que β y la base se resuelven por
mínimos cuadrados con β ≥ 0 y el optimizador externo solo busca θ y λ (estimación MAP). Tres versiones:
- ``plano``: sin priors.
- ``debil``: priors débiles (θ ~ Beta(2, 3), λ log-normal alrededor del gasto medio y ROI ~ N+(0, 5)).
  El prior va sobre el ROI y no sobre β (como Google Meridian): si no, una saturación casi plana se
  confunde con la base y el canal se lleva ventas que no son suyas.
- ``debil_calibrado``: lo anterior + el ROI del e-mail medido con un experimento (decisión 013).
- ``con_control``: lo anterior + un indicador ruidoso de la demanda (tipo Google Trends de la categoría),
  que es la forma estándar de atacar el gasto endógeno.
La incertidumbre sale de un bootstrap por bloques de residuos (8 semanas).

Presupuesto: mismo total del último año, cada canal entre −30% y +30% de su gasto actual. Se compara lo que
promete el modelo contra lo que de verdad pasaría (se evalúa con los parámetros verdaderos).
Todo se publica (prefijo mmm_): son datos sintéticos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

from growth_lab import db

CANALES = ["tv", "paid_search", "social", "display", "email"]
SEMANAS = 156
SEMILLA = 7
GASTO_MEDIO = np.array([60.0, 40.0, 30.0, 25.0, 3.0])  # miles USD por semana
VERDAD = {  # parámetros verdaderos del generador
    "theta": np.array([0.60, 0.10, 0.30, 0.20, 0.00]),
    "lam": np.array([60.0, 50.0, 40.0, 50.0, 3.0]),
    "beta": np.array([380.0, 360.0, 170.0, 75.0, 38.0]),
}
ENDOGENEIDAD = {"limpio": 0.0, "endogeno": 0.25}  # cuánto sigue el gasto a la demanda esperada (log)
THETA_MAX = 0.9
LIMITE = 0.30  # decisión 014: cada canal ±30% frente a hoy
ERROR_EXPERIMENTO = 0.15  # error estándar relativo del ROI del e-mail medido con prueba
ESCALA_PRIOR_ROI = 5.0  # prior débil: ROI en ventas entre 0 y ~10 es plausible
VERSIONES = ["plano", "debil", "debil_calibrado", "con_control"]
VERSION_ELEGIDA = "con_control"
RUIDO_CONTROL = 20.0  # error del indicador de demanda (desv. estándar, miles USD de ventas)
MARGEN = 0.30  # mismo margen de experimentos.MARGEN: ROI en ventas de equilibrio = 1 / 0,30


# ---------- Generador ----------

def adstock(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Arrastre geométrico normalizado: columnas = canales; conserva el gasto total (salvo la cola)."""
    a = np.zeros_like(x, dtype=float)
    acum = np.zeros(x.shape[1])
    for t in range(len(x)):
        acum = theta * acum + (1 - theta) * x[t]
        a[t] = acum
    return a


def saturacion(a: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return 1 - np.exp(-a / lam)


def contribuciones(gasto: np.ndarray, theta, lam, beta) -> np.ndarray:
    return beta * saturacion(adstock(gasto, theta), lam)


def base_estacional(n: int = SEMANAS) -> np.ndarray:
    t = np.arange(n)
    return 1500 + 1.5 * t + 160 * np.sin(2 * np.pi * t / 52) + 60 * np.cos(4 * np.pi * t / 52)


def generar(escenario: str, semilla: int = SEMILLA, n: int = SEMANAS) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    base = base_estacional(n)
    demanda = np.zeros(n)  # demanda oculta AR(1): promociones, competencia, clima (el modelo no la ve)
    for t in range(1, n):
        demanda[t] = 0.7 * demanda[t - 1] + rng.normal(0, 40)
    esperado = (base - base.mean()) + demanda
    z = esperado / esperado.std()
    g = ENDOGENEIDAD[escenario]
    ruido_gasto = rng.normal(0, 0.35, (n, len(CANALES)))
    gasto = GASTO_MEDIO * np.exp(g * z[:, None] + ruido_gasto - 0.5 * 0.35**2)
    gasto[rng.random(n) < 0.25, 0] = 0.0  # TV va por oleadas: 1 de cada 4 semanas apagada
    contrib = contribuciones(gasto, **VERDAD)
    ventas = base + demanda + contrib.sum(axis=1) + rng.normal(0, 50, n)
    df = pd.DataFrame(gasto, columns=[f"gasto_{c}" for c in CANALES])
    df.insert(0, "semana", np.arange(n))
    df.insert(0, "escenario", escenario)
    df["ventas"] = ventas
    for i, c in enumerate(CANALES):
        df[f"verdad_aporte_{c}"] = contrib[:, i]
    df["verdad_base"] = base + demanda
    df["indice_demanda"] = demanda + rng.normal(0, RUIDO_CONTROL, n)  # lo que se ve de la demanda oculta
    return df


def roi_verdadero(gasto: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return roi_y_marginal(gasto, VERDAD["theta"], VERDAD["lam"], VERDAD["beta"])


def roi_y_marginal(gasto, theta, lam, beta) -> tuple[np.ndarray, np.ndarray]:
    """ROI promedio (aporte / gasto) y ROI marginal (lo que trae el siguiente 1% de gasto)."""
    c0 = contribuciones(gasto, theta, lam, beta).sum(axis=0)
    roi = c0 / gasto.sum(axis=0)
    marg = np.empty(len(CANALES))
    for i in range(len(CANALES)):
        g = gasto.copy()
        g[:, i] *= 1.01
        marg[i] = (contribuciones(g, theta, lam, beta).sum(axis=0)[i] - c0[i]) / (0.01 * gasto[:, i].sum())
    return roi, marg


# ---------- Modelo ----------

def _base_x(n: int) -> np.ndarray:
    t = np.arange(n) / 52
    cols = [np.ones(n), t]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
    return np.column_stack(cols)


class MMM:
    """MMM con β y base perfilados por mínimos cuadrados (β ≥ 0) y θ, λ por MAP."""

    def __init__(self, version: str = VERSION_ELEGIDA, roi_email: float | None = None):
        assert version in VERSIONES
        self.version, self.roi_email = version, roi_email

    # Parámetros externos: z = [logit(θ/THETA_MAX) × 5, log(λ / gasto medio) × 5]
    def _desempacar(self, z):
        k = len(CANALES)
        theta = THETA_MAX / (1 + np.exp(-z[:k]))
        lam = self.escala * np.exp(z[k:])
        return theta, lam

    def _resolver(self, z, y):
        theta, lam = self._desempacar(z)
        s = saturacion(adstock(self.gasto, theta), lam)
        k, nb = len(CANALES), self.xb.shape[1]
        x = np.hstack([s, self.xb])
        filas_x, filas_y = [x], [y]
        if self.version != "plano":  # prior débil sobre el ROI: ROI_c = β_c · media(s_c) / media(gasto_c) ~ N+(0, 5)
            r = np.zeros((k, k + nb))
            r[np.arange(k), np.arange(k)] = self.sigma * s.mean(axis=0) / self.gasto.mean(axis=0) / ESCALA_PRIOR_ROI
            filas_x.append(r)
            filas_y.append(np.zeros(k))
        if self.version in ("debil_calibrado", "con_control") and self.roi_email is not None:
            # El experimento mide ROI_email = β_e · Σ s_e / Σ gasto_e con error estándar relativo conocido.
            i = CANALES.index("email")
            fila = np.zeros(k + nb)
            fila[i] = s[:, i].sum() / self.gasto[:, i].sum()
            w = self.sigma / (ERROR_EXPERIMENTO * self.roi_email)
            filas_x.append(fila[None, :] * w)
            filas_y.append(np.array([self.roi_email * w]))
        xa, ya = np.vstack(filas_x), np.concatenate(filas_y)
        lb = np.r_[np.zeros(k), np.full(nb, -np.inf)]
        sol = lsq_linear(xa, ya, bounds=(lb, np.full(k + nb, np.inf)), method="bvls")
        return sol.x, float(((xa @ sol.x - ya) ** 2).sum())

    def _objetivo(self, z, y):
        _, rss = self._resolver(z, y)
        nlp = 0.5 * rss / self.sigma**2
        if self.version != "plano":
            theta, _ = self._desempacar(z)
            u = np.clip(theta, 1e-6, 1 - 1e-6)
            nlp -= ((2 - 1) * np.log(u) + (3 - 1) * np.log(1 - u)).sum()  # Beta(2, 3)
            nlp += 0.5 * float(((z[len(CANALES):] / 0.7) ** 2).sum())  # log(λ / gasto medio) ~ N(0, 0,7)
        return nlp

    def _ajustar_z(self, y, inicios):
        cotas = [(-6, 6)] * len(CANALES) + [(np.log(0.1), np.log(10))] * len(CANALES)
        mejor = None
        for z0 in inicios:
            r = minimize(self._objetivo, z0, args=(y,), method="L-BFGS-B", bounds=cotas)
            if mejor is None or r.fun < mejor.fun:
                mejor = r
        return mejor.x

    def ajustar(self, gasto: np.ndarray, ventas: np.ndarray, control: np.ndarray | None = None,
                n_inicios: int = 6, semilla: int = 0) -> MMM:
        self.gasto, self.xb = gasto, _base_x(len(gasto))
        if self.version == "con_control":
            assert control is not None, "con_control necesita el indicador de demanda"
            self.xb = np.column_stack([self.xb, control])
        self.escala = gasto.mean(axis=0)
        rng = np.random.default_rng(semilla)
        inicios = [np.zeros(2 * len(CANALES))] + [rng.normal(0, 1, 2 * len(CANALES)) for _ in range(n_inicios - 1)]
        self.sigma = ventas.std() * 0.1
        for _ in range(2):  # σ del ruido: se re-estima con los residuos
            self.z = self._ajustar_z(ventas, inicios)
            coef, _ = self._resolver(self.z, ventas)
            res = ventas - np.hstack([saturacion(adstock(gasto, self._desempacar(self.z)[0]),
                                                 self._desempacar(self.z)[1]), self.xb]) @ coef
            self.sigma = float(res.std())
            inicios = [self.z]
        self._guardar(ventas)
        return self

    def _guardar(self, ventas):
        coef, _ = self._resolver(self.z, ventas)
        self.theta, self.lam = self._desempacar(self.z)
        self.beta, self.coef_base = coef[:len(CANALES)], coef[len(CANALES):]
        self.ajuste = np.hstack([saturacion(adstock(self.gasto, self.theta), self.lam), self.xb]) @ coef
        self.residuos = ventas - self.ajuste

    def bootstrap(self, n: int = 100, bloque: int = 8, semilla: int = 0) -> np.ndarray:
        """ROI por canal en n réplicas: residuos remuestreados por bloques sobre el ajuste."""
        rng = np.random.default_rng(semilla)
        nt = len(self.residuos)
        copia = MMM(self.version, self.roi_email)
        copia.gasto, copia.xb, copia.escala, copia.sigma = self.gasto, self.xb, self.escala, self.sigma
        rois = []
        for _ in range(n):
            if self.roi_email is not None:  # el experimento también tiene error: se remuestrea
                copia.roi_email = self.roi_email * (1 + rng.normal(0, ERROR_EXPERIMENTO))
            inicios = rng.integers(0, nt - bloque + 1, nt // bloque + 1)
            e = np.concatenate([self.residuos[i:i + bloque] for i in inicios])[:nt]
            yb = self.ajuste + e
            copia.z = copia._ajustar_z(yb, [self.z])
            copia._guardar(yb)
            rois.append(roi_y_marginal(self.gasto, copia.theta, copia.lam, copia.beta)[0])
        return np.array(rois)


# ---------- Presupuesto ----------

def ventas_medios(gasto: np.ndarray, mult: np.ndarray, theta, lam, beta, semanas: int = 52) -> float:
    """Aporte total de medios en las últimas `semanas` si el gasto de esas semanas se escala por canal."""
    g = gasto.copy()
    g[-semanas:] *= mult
    return float(contribuciones(g, theta, lam, beta)[-semanas:].sum())


def optimizar(gasto: np.ndarray, theta, lam, beta, limite: float = LIMITE, semanas: int = 52) -> np.ndarray:
    """Multiplicador por canal que maximiza el aporte con el mismo gasto total y cada canal en ±límite."""
    total = gasto[-semanas:].sum(axis=0)
    restr = {"type": "eq", "fun": lambda m: (m * total).sum() / total.sum() - 1}
    r = minimize(lambda m: -ventas_medios(gasto, m, theta, lam, beta, semanas), np.ones(len(CANALES)),
                 method="SLSQP", bounds=[(1 - limite, 1 + limite)] * len(CANALES), constraints=[restr])
    return r.x


# ---------- Corrida completa ----------

def correr(n_boot: int = 100) -> dict[str, pd.DataFrame]:
    series, filas_roi, filas_err, modelos = [], [], [], {}
    rng = np.random.default_rng(SEMILLA + 1)
    for esc in ENDOGENEIDAD:
        df = generar(esc)
        series.append(df)
        gasto = df[[f"gasto_{c}" for c in CANALES]].to_numpy()
        roi_v, marg_v = roi_verdadero(gasto)
        # Experimento de e-mail: ROI verdadero + error de medición (decisión 013: e-mail se escala con prueba).
        roi_exp = float(roi_v[CANALES.index("email")] * (1 + rng.normal(0, ERROR_EXPERIMENTO)))
        for v in VERSIONES:
            m = MMM(v, roi_email=roi_exp).ajustar(gasto, df.ventas.to_numpy(), df.indice_demanda.to_numpy())
            modelos[(esc, v)] = m
            roi_e, marg_e = roi_y_marginal(gasto, m.theta, m.lam, m.beta)
            boot = m.bootstrap(n_boot) if v in ("debil_calibrado", VERSION_ELEGIDA) else None
            aporte_v = df[[f"verdad_aporte_{c}" for c in CANALES]].to_numpy().sum()
            aporte_e = contribuciones(gasto, m.theta, m.lam, m.beta).sum()
            for i, c in enumerate(CANALES):
                filas_roi.append({
                    "escenario": esc, "version": v, "canal": c,
                    "roi_verdad": roi_v[i], "roi_est": roi_e[i],
                    "roi_p05": np.quantile(boot[:, i], 0.05) if boot is not None else np.nan,
                    "roi_p95": np.quantile(boot[:, i], 0.95) if boot is not None else np.nan,
                    "roi_marginal_verdad": marg_v[i], "roi_marginal_est": marg_e[i],
                    "theta_verdad": VERDAD["theta"][i], "theta_est": m.theta[i],
                    "roi_experimento": roi_exp if c == "email" else np.nan,
                })
            filas_err.append({
                "escenario": esc, "version": v,
                "mae_roi": float(np.abs(roi_e - roi_v).mean()),
                "error_rel_roi_pct": float((np.abs(roi_e - roi_v) / roi_v).mean() * 100),
                "aporte_medios_verdad_pct": aporte_v / df.ventas.sum() * 100,
                "aporte_medios_est_pct": aporte_e / df.ventas.sum() * 100,
                "r2": 1 - m.residuos.var() / df.ventas.var(),
            })
    roi = pd.DataFrame(filas_roi)
    err = pd.DataFrame(filas_err)

    # Presupuesto con el escenario realista (endógeno): plan de cada versión del modelo contra la verdad.
    df = series[1]
    gasto = df[[f"gasto_{c}" for c in CANALES]].to_numpy()
    v = VERDAD["theta"], VERDAD["lam"], VERDAD["beta"]
    hoy = np.ones(len(CANALES))
    total = gasto[-52:].sum(axis=0)
    base_v = ventas_medios(gasto, hoy, *v)
    filas_p, filas_r = [], []
    for nombre in VERSIONES + ["verdad"]:
        pm = v if nombre == "verdad" else (lambda x: (x.theta, x.lam, x.beta))(modelos[("endogeno", nombre)])
        mult = optimizar(gasto, *pm)
        for i, c in enumerate(CANALES):
            filas_p.append({"plan": nombre, "canal": c, "gasto_anual_hoy": total[i],
                            "gasto_anual_plan": total[i] * mult[i], "cambio_pct": (mult[i] - 1) * 100})
        filas_r.append({"plan": nombre,
                        "ganancia_real": ventas_medios(gasto, mult, *v) - base_v,
                        "ganancia_prometida": ventas_medios(gasto, mult, *pm) - ventas_medios(gasto, hoy, *pm),
                        "gasto_anual": float((total * mult).sum())})
    resumen = pd.DataFrame(filas_r)
    resumen["ganancia_real_pct_medios"] = resumen.ganancia_real / base_v * 100
    resumen["utilidad_extra"] = resumen.ganancia_real * MARGEN  # mismo gasto: toda la venta extra es margen
    salidas = {
        "mmm_series": pd.concat(series, ignore_index=True),
        "mmm_roi": roi,
        "mmm_error": err,
        "mmm_presupuesto": pd.DataFrame(filas_p),
        "mmm_presupuesto_resumen": resumen,
    }
    for nombre, d in salidas.items():
        d.to_parquet(db.PROCESSED / f"{nombre}.parquet", compression="zstd", index=False)
    return salidas


def robustez(n: int = 10) -> pd.DataFrame:
    """Repite el escenario endógeno con n semillas: ¿el plan de cada versión gana de verdad?"""
    filas = []
    for semilla in range(100, 100 + n):
        df = generar("endogeno", semilla=semilla)
        gasto = df[[f"gasto_{c}" for c in CANALES]].to_numpy()
        v = VERDAD["theta"], VERDAD["lam"], VERDAD["beta"]
        roi_v, _ = roi_verdadero(gasto)
        roi_exp = float(roi_v[CANALES.index("email")] * (1 + np.random.default_rng(semilla).normal(0, ERROR_EXPERIMENTO)))
        hoy = np.ones(len(CANALES))
        base_v = ventas_medios(gasto, hoy, *v)
        optimo = ventas_medios(gasto, optimizar(gasto, *v), *v) - base_v
        for ver in VERSIONES:
            m = MMM(ver, roi_email=roi_exp).ajustar(gasto, df.ventas.to_numpy(), df.indice_demanda.to_numpy(), n_inicios=3)
            roi_e, _ = roi_y_marginal(gasto, m.theta, m.lam, m.beta)
            ganancia = ventas_medios(gasto, optimizar(gasto, m.theta, m.lam, m.beta), *v) - base_v
            filas.append({"semilla": semilla, "version": ver, "ganancia_real": ganancia, "ganancia_optima": optimo,
                          "captura_pct": ganancia / optimo * 100, "mae_roi": float(np.abs(roi_e - roi_v).mean())})
    r = pd.DataFrame(filas)
    r.to_parquet(db.PROCESSED / "mmm_robustez.parquet", compression="zstd", index=False)
    return r


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    for k, d in correr().items():
        if k != "mmm_series":
            print(k)
            print(d.round(3).to_string(index=False))
            print()
    r = robustez()
    print(r.groupby("version")[["ganancia_real", "captura_pct", "mae_roi"]].median().round(2))
    print("planes que pierden:", r[r.ganancia_real < 0].groupby("version").size().to_dict())
