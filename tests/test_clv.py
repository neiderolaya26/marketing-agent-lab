import numpy as np
import pandas as pd
import pytest

from growth_lab import clv


def simular_bgnbd(n, r, alpha, a, b, T, semilla=0):
    """Genera hogares según el proceso BG/NBD: ritmo λ ~ Gamma(r, 1/α), abandono p ~ Beta(a, b)."""
    rng = np.random.default_rng(semilla)
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    x, t_x = np.zeros(n), np.zeros(n)
    for i in range(n):
        t, vivo = 0.0, True
        while vivo:
            t += rng.exponential(1 / lam[i])
            if t > T:
                break
            x[i] += 1
            t_x[i] = t
            vivo = rng.random() > p[i]
    return x, t_x, np.full(n, T)


def test_bgnbd_recupera_parametros_simulados():
    x, t_x, T = simular_bgnbd(4000, r=0.8, alpha=4.0, a=0.6, b=3.0, T=39)
    p = clv.ajustar_bgnbd(x, t_x, T)
    assert p["r"] == pytest.approx(0.8, rel=0.25)
    assert p["alpha"] == pytest.approx(4.0, rel=0.35)
    # Las compras esperadas promedio en 13 semanas deben ser positivas y finitas.
    e = clv.compras_esperadas(p, 13, x, t_x, T)
    assert np.all(np.isfinite(e)) and e.mean() > 0


def test_prob_vivo():
    p = {"r": 1.0, "alpha": 2.0, "a": 0.5, "b": 2.0}
    assert clv.prob_vivo(p, [0], [0], [10])[0] == pytest.approx(1.0)  # sin repetir no hay evidencia de abandono
    # Mismo número de compras: quien compró hace poco está más vivo que quien dejó de venir.
    reciente, viejo = clv.prob_vivo(p, [5, 5], [9.5, 2.0], [10, 10])
    assert reciente > viejo


def test_gamma_gamma_contrae_hacia_la_media():
    g = {"p": 2.0, "q": 3.0, "v": 10.0}
    media_pob = g["p"] * g["v"] / (g["q"] - 1)
    poco, mucho = clv.gasto_medio_esperado(g, [1, 50], [100.0, 100.0])
    assert media_pob < poco < mucho < 100.0  # con más compras se le cree más a su propio promedio
    assert clv.gasto_medio_esperado(g, [0], [np.nan])[0] == pytest.approx(media_pob)


def test_resumen_rfm():
    compras = pd.DataFrame({
        "household_id": ["h1", "h1", "h1", "h2"],
        "dia": pd.to_datetime(["2017-01-02", "2017-01-16", "2017-01-30", "2017-01-09"]),
        "gasto": [10.0, 20.0, 40.0, 5.0], "desc_tienda": 0.0, "desc_cupon": 0.0,
    })
    r = clv.resumen_rfm(compras, "2017-01-01", "2017-02-13")
    assert r.loc["h1", "x"] == 2 and r.loc["h1", "t_x"] == pytest.approx(4.0) and r.loc["h1", "T"] == pytest.approx(6.0)
    assert r.loc["h1", "gasto_medio_rep"] == pytest.approx(30.0)  # sin la primera compra
    assert r.loc["h2", "x"] == 0 and np.isnan(r.loc["h2", "gasto_medio_rep"])


def test_metricas_valor_y_diferencia():
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    m = clv.metricas_valor(y, y)
    assert m["mae"] == 0 and m["r2"] == pytest.approx(1.0) and m["sesgo_total"] == pytest.approx(0.0)
    dif, lo, hi = clv.ic_diferencia_mae(y, y, y + 5, n_boot=50)
    assert dif == pytest.approx(-5.0) and hi < 0
