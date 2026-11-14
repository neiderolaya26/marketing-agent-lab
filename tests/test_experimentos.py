import numpy as np
import pandas as pd
import pytest

from growth_lab import experimentos as ex


def test_tamano_de_muestra_y_mde_son_inversos():
    n = ex.n_por_grupo_media(sd=10.0, efecto_abs=1.0)
    assert n == pytest.approx(1570, abs=2)  # 2·(1,96+0,84)²·100
    assert ex.mde_media(10.0, n) == pytest.approx(1.0, rel=0.01)
    assert ex.n_por_grupo_proporcion(0.10, 0.20) > ex.n_por_grupo_proporcion(0.10, 0.50)


def test_holm():
    assert ex.holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert ex.holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])
    assert ex.holm([0.5, 0.6, 0.9])[-1] <= 1.0


def test_cuped_reduce_varianza_sin_cambiar_la_media():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 20, 5000)
    y = 0.8 * x + rng.normal(0, 5, 5000)
    adj = ex.cuped(y, x)
    assert adj.mean() == pytest.approx(y.mean())
    assert adj.var() < 0.1 * y.var()


def test_efecto_detecta_diferencia():
    rng = np.random.default_rng(1)
    e = ex.efecto(rng.normal(1.2, 1, 4000), rng.normal(1.0, 1, 4000))
    assert e["ic_inf"] > 0 and e["p_valor"] < 0.01


def test_obrien_fleming_controla_el_peeking():
    assert ex.constante_obrien_fleming(1) == pytest.approx(1.96, abs=0.02)
    lim = ex.limites_obrien_fleming(5)
    assert lim[0] > lim[-1] > 1.9  # exigente al principio
    rng = np.random.default_rng(2)
    r = ex.simular_peeking(rng.normal(0, 1, 4000), miradas=10, reps=400)
    assert r["fp_mirando_siempre"] > 0.10  # mirar siempre infla los falsos positivos
    assert r["fp_obrien_fleming"] < 0.09


def test_bayes_proporcion():
    r = ex.bayes_proporcion(x1=200, n1=1000, x0=100, n0=1000, draws=20000)
    assert r["p_mejor"] > 0.99 and r["lift_p05"] > 0.5


def test_srm_y_balance():
    assert ex.srm(pd.Series([1000, 1010, 990])) > 0.5
    assert ex.srm(pd.Series([1000, 1300, 1000])) < 0.001
