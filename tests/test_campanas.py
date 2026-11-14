import numpy as np
import pandas as pd
import pytest

from growth_lab import campanas as cp


def panel_sintetico(efecto: float, n: int = 3000, semilla: int = 0) -> pd.DataFrame:
    """Hogares donde los que gastan más tienen más probabilidad de recibir la campaña (selección),
    y la campaña suma `efecto` al gasto semanal de los tratados."""
    rng = np.random.default_rng(semilla)
    nivel = rng.lognormal(3.5, 0.6, n)  # gasto semanal "normal" del hogar
    p = pd.DataFrame({c: rng.normal(0, 1, n) for c in cp.COVARIABLES})
    p["log_gasto_sem"] = np.log1p(nivel)
    p["gasto_ult_5s"] = nivel * rng.uniform(0.9, 1.1, n)
    prob = 1 / (1 + np.exp(-(np.log(nivel) - 3.5) * 2))
    p["tratado"] = (rng.random(n) < prob).astype(int)
    p["antes"] = nivel + rng.normal(0, 5, n)
    p["durante"] = nivel + rng.normal(0, 5, n) + efecto * p.tratado
    p["cambio"] = p.durante - p.antes
    return p


def test_ingenuo_esta_sesgado_y_did_no():
    p = panel_sintetico(efecto=0.0)
    assert cp.ingenuo(p) > 5  # los tratados ya gastaban más: el ingenuo "ve" un efecto que no existe
    assert abs(cp.did_simple(p)) < 1
    assert abs(cp.did_doble_robusto(p, cp.propension(p))) < 1.5


def test_did_doble_robusto_recupera_el_efecto():
    p = panel_sintetico(efecto=4.0)
    assert cp.did_doble_robusto(p, cp.propension(p)) == pytest.approx(4.0, abs=1.2)


def test_dml_recupera_el_efecto():
    p = panel_sintetico(efecto=4.0)
    theta, ee = cp.dml(p)
    assert abs(theta - 4.0) < 3 * ee + 1.0


def test_combinar_pondera_por_precision():
    e = pd.DataFrame({"metodo": "m", "ventana": "real", "efecto": [1.0, 3.0], "ic_inf": [0.0, -1.0],
                      "ic_sup": [2.0, 7.0], "gasto_semanal_tratados_antes": 50.0, "hogares": 100, "tratados": 50})
    c = cp.combinar(e).iloc[0]
    assert 1.0 < c.efecto < 2.0  # más cerca de la campaña precisa
    assert c.ic_sup - c.ic_inf < 2.0  # más angosto que el más preciso
