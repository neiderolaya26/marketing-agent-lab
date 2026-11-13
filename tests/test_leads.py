import numpy as np
import pandas as pd
import pytest

from growth_lab import db, leads

T = pd.Timestamp


@pytest.fixture
def con(tmp_path):
    """Landing L1: un lead de 2017 que cierra el 10-ene-2018 y tres leads de 2018."""
    pd.DataFrame({
        "mql_id": ["a", "b", "c", "d"],
        "first_contact_date": [T("2017-12-01"), T("2018-01-05"), T("2018-01-20"), T("2018-01-20")],
        "landing_page_id": ["L1", "L1", "L1", "L2"],
        "origin": ["paid_search", "paid_search", "social", None],
    }).to_parquet(tmp_path / "marketing_qualified_leads.parquet", index=False)
    pd.DataFrame({"mql_id": ["a", "c"], "seller_id": ["s1", "s2"], "won_date": [T("2018-01-10"), T("2018-02-01")]}
                 ).to_parquet(tmp_path / "closed_deals.parquet", index=False)
    return db.conectar(tmp_path)


def test_historia_sin_fuga(con):
    d = leads.poblacion(con).set_index("mql_id")
    assert set(d.index) == {"b", "c", "d"}  # solo 2018
    # b (5-ene): ve el lead previo de L1, pero su cierre (10-ene) aún no se conoce.
    assert d.loc["b", "landing_leads_previos"] == 1 and d.loc["b", "landing_cierres_conocidos"] == 0
    # c (20-ene): ve a y b, y ya conoce el cierre de a; su propio cierre no cuenta.
    assert d.loc["c", "landing_leads_previos"] == 2 and d.loc["c", "landing_cierres_conocidos"] == 1
    assert d.loc["c", "landing_tasa_previa"] > d.loc["b", "landing_tasa_previa"]
    # d: landing nueva y canal sin dato.
    assert d.loc["d", "landing_nueva"] == 1 and d.loc["d", "canal"] == "sin_dato"
    assert d.loc["c", "cerro"] and not d.loc["b", "cerro"]


def test_particion_usa_categorias_comunes(con):
    d = leads.poblacion(con)
    entreno, prueba = leads.particion(d, corte="2018-01-15")
    assert len(entreno) == 1 and len(prueba) == 2
    assert list(entreno.landing.cat.categories) == list(prueba.landing.cat.categories)


def test_deciles_y_lift():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    dec = leads.deciles(y, p, n=5)
    assert dec.lift.iloc[0] == pytest.approx(5.0)  # el top 20% tiene los dos cierres
    assert dec.captura_acumulada.iloc[0] == 1.0


def test_diferencia_auc_pareada():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    bueno, azar = y + rng.normal(0, 0.8, 500), rng.normal(0, 1, 500)
    dif, lo, hi = leads.ic_diferencia_auc(y, bueno, azar, n_boot=200)
    assert dif > 0 and lo > 0  # el modelo informativo le gana al azar con IC que no cruza 0
