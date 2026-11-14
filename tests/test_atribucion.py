import numpy as np
import pandas as pd
import pytest

from growth_lab import atribucion as at


@pytest.fixture
def toques():
    """Recorrido 1: social → email → paid_search (3 toques); recorrido 2: solo display."""
    return pd.DataFrame({
        "journey_id": [1, 1, 1, 2],
        "step": [0, 1, 2, 0],
        "channel": ["social", "email", "paid_search", "display"],
        "t_hours": [100.0, 500.0, 700.0, 300.0],
        "cost": [0.8, 0.05, 1.2, 0.4],
    })


@pytest.mark.parametrize("regla, esperado", [
    ("ultimo_clic", {"paid_search": 1.0}),
    ("primer_clic", {"social": 1.0}),
    ("lineal", {"social": 1 / 3, "email": 1 / 3, "paid_search": 1 / 3}),
    ("en_u", {"social": 0.4, "email": 0.2, "paid_search": 0.4}),
])
def test_reglas(toques, regla, esperado):
    c = at.credito_regla(toques, pd.Index([1, 2]), regla)
    for canal, v in esperado.items():
        assert c.loc[1, canal] == pytest.approx(v)
    assert c.loc[2, "display"] == pytest.approx(1.0)
    assert np.allclose(c.sum(axis=1), 1.0)


def test_decaimiento_da_mas_peso_a_lo_reciente(toques):
    c = at.credito_regla(toques, pd.Index([1]), "decaimiento_temporal")
    assert c.loc[1, "paid_search"] > c.loc[1, "email"] > c.loc[1, "social"]


def test_markov_canal_imprescindible():
    # Solo convierte quien pasa por paid_search: quitarlo elimina todas las conversiones.
    cam = pd.DataFrame({"camino": ["social > paid_search", "social"], "conv": [50, 0], "no_conv": [50, 100]})
    cam["total"] = cam.conv + cam.no_conv
    part = at.participacion_markov(cam)
    assert part["paid_search"] == pytest.approx(part["social"])  # ambos imprescindibles en el único camino que convierte
    assert part[["display", "email"]].sum() == pytest.approx(0.0)


class ModeloAditivo:
    """P(conv) = 0,01 + suma de pesos de los canales presentes (para verificar Shapley exacto)."""

    pesos = np.array([0.3, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0])

    def predict_proba(self, x):
        p = 0.01 + (x > 0) @ self.pesos
        return np.column_stack([1 - p, p])


def test_shapley_sobre_modelo_aditivo_reparte_por_peso():
    x = np.array([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    phi = at.shapley_sobre_modelo(ModeloAditivo(), x)[0]
    assert phi[:4] == pytest.approx([0.5, 1 / 6, 0.0, 1 / 3])
    assert phi.sum() == pytest.approx(1.0)
