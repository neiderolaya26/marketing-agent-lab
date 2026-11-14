import numpy as np
import pytest

from growth_lab import uplift as up


@pytest.fixture
def datos():
    """Experimento sintético: x0 marca persuadibles (efecto +0,3); x1 marca 'seguros' (compran igual, sin efecto)."""
    rng = np.random.default_rng(0)
    n = 8000
    x = rng.integers(0, 2, (n, 3)).astype(float)
    t = rng.integers(0, 2, n)
    p = 0.05 + 0.6 * x[:, 1] + 0.3 * x[:, 0] * t
    y = (rng.random(n) < p).astype(float)
    return x, t, y


def test_curva_qini_termina_en_el_efecto_total():
    t = np.array([1, 0, 1, 0, 1, 0])
    y = np.array([1, 0, 1, 1, 0, 0])
    c = up.curva_qini(np.arange(6)[::-1].astype(float), t, y, fracciones=[1.0])
    efecto = y[t == 1].mean() - y[t == 0].mean()
    assert c.ganancia.iloc[-1] == pytest.approx(efecto * 6)


def test_uplift_encuentra_persuadibles_y_propension_no(datos):
    x, t, y = datos
    tr, te = slice(0, 5000), slice(5000, None)
    met = {}
    for nombre in ("t_learner", "x_learner", "propension"):
        score = up.LEARNERS[nombre](x[tr], t[tr], y[tr], x[te])
        met[nombre] = up.coef_qini(up.curva_qini(score, t[te], y[te]), 3000)
    assert met["t_learner"] > 0.02 and met["x_learner"] > 0.02
    # La propensión ordena por x1 (los seguros, que compran mucho) y no por quién responde al contacto.
    assert met["propension"] < met["t_learner"]


def test_valor_politica_ipw():
    rng = np.random.default_rng(1)
    n = 30000
    asignado = rng.choice(["A", "B", "C"], n)
    y = np.where(asignado == "B", 2.0, 1.0) + rng.normal(0, 0.1, n)
    todos_b = np.full(n, "B")
    assert up.valor_politica(asignado, todos_b, y) == pytest.approx(2.0, rel=0.05)
    assert up.valor_politica(asignado, np.full(n, "A"), y) == pytest.approx(1.0, rel=0.05)
