import numpy as np
import pandas as pd
import pytest

from growth_lab import segmentacion as sg


@pytest.fixture
def hogares():
    """200 hogares en 4 grupos bien separados por RFM; el gasto futuro sigue al grupo."""
    rng = np.random.default_rng(0)
    grupos = np.repeat([0, 1, 2, 3], 50)
    base = {0: (2, 80, 2500), 1: (5, 30, 900), 2: (25, 15, 350), 3: (70, 3, 50)}
    r, f, m = (np.array([base[g][i] for g in grupos]) for i in range(3))
    d = pd.DataFrame({
        "household_id": [f"h{i}" for i in range(200)],
        "recencia_dias": r * rng.uniform(0.8, 1.2, 200),
        "frecuencia": (f * rng.uniform(0.8, 1.2, 200)).round(),
        "monto": m * rng.uniform(0.8, 1.2, 200),
        "ticket_medio": rng.uniform(15, 35, 200),
        "pct_desc_tienda": rng.uniform(0.1, 0.2, 200),
        "pct_cupon": 0.0,
        "pct_frescos": rng.uniform(0.2, 0.3, 200),
        "pct_drug_gm": rng.uniform(0.05, 0.15, 200),
        "pct_semanas_activas": rng.uniform(0, 1, 200),
    })
    d["ventas_futuras"] = m * 0.4 * rng.uniform(0.7, 1.3, 200)
    d["activo_futuro"] = grupos < 3
    return d


def test_puntajes_rfm_son_quintiles_y_la_recencia_se_invierte(hogares):
    p = sg.puntajes_rfm(hogares)
    assert set(p.R) == set(p.F) == set(p.M) == {1, 2, 3, 4, 5}
    mas_reciente = hogares.recencia_dias.idxmin()
    assert p.loc[mas_reciente, "R"] == 5


@pytest.mark.parametrize("rfm, esperado", [
    ((5, 5, 5), "Campeones"), ((1, 5, 5), "Leales en riesgo"), ((3, 4, 2), "Leales"),
    ((5, 1, 1), "Recientes de bajo uso"), ((1, 1, 1), "Hibernando"), ((2, 3, 3), "En riesgo"),
    ((3, 3, 4), "Potenciales"), ((3, 3, 1), "Ocasionales"),
])
def test_segmento_rfm(rfm, esperado):
    assert sg.segmento_rfm(*rfm) == esperado


def test_eta2_extremos():
    y = pd.Series([1.0, 1.0, 5.0, 5.0])
    assert sg.eta2(pd.Series(["a", "a", "b", "b"]), y) == pytest.approx(1.0)
    assert sg.eta2(pd.Series(["a", "b", "a", "b"]), y) == pytest.approx(0.0)


def test_kmeans_recupera_los_grupos_y_los_nombra_por_monto(hogares):
    comp, et = sg.comparar(hogares, ks=range(4, 5))
    seg = sg.nombrar(hogares, et["kmeans|rfm|4"])
    assert seg.iloc[:50].eq("Núcleo").all() and seg.iloc[150:].eq("Dormidos").all()
    fila = comp[(comp.metodo == "kmeans") & (comp.variables == "rfm") & (comp.k == 4)].iloc[0]
    assert fila.eta2_ventas_futuras > 0.8


def test_perfil_agrega_y_suma_uno(hogares):
    seg = pd.Series(np.repeat(["A", "B"], 100))
    p = sg.perfil(hogares, seg)
    assert p.hogares.sum() == 200
    assert p.pct_hogares.sum() == pytest.approx(1.0) and p.pct_ventas_futuras.sum() == pytest.approx(1.0)
