import numpy as np
import pandas as pd
import pytest

from growth_lab import db, embudo

T = pd.Timestamp


@pytest.fixture
def con(tmp_path):
    """Mini-Olist: 3 leads de paid_search (2 cierran) y 1 de email (no cierra)."""
    tablas = {
        "marketing_qualified_leads": pd.DataFrame({
            "mql_id": ["m1", "m2", "m3", "m4"],
            "first_contact_date": [T("2018-01-01")] * 4,
            "landing_page_id": ["l"] * 4,
            "origin": ["paid_search", "paid_search", "paid_search", "email"],
        }),
        # s1 maduro (gana en feb); s2 inmaduro (gana 20 días antes del corte)
        "closed_deals": pd.DataFrame({
            "mql_id": ["m1", "m2"], "seller_id": ["s1", "s2"],
            "won_date": [T("2018-02-01"), T("2018-08-09")],
        }),
        "orders": pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "customer_id": ["c1", "c2", "c3", "c4", "c5"],
            "order_status": ["delivered", "delivered", "canceled", "delivered", "delivered"],
            "order_purchase_timestamp": [T("2018-01-15"), T("2018-02-10"), T("2018-02-11"), T("2018-06-01"), T("2018-08-20")],
            "order_delivered_customer_date": [T("2018-01-20")] * 5,
            "order_estimated_delivery_date": [T("2018-01-25")] * 5,
        }),
        # o1 antes de ganar (no cuenta), o2 dentro de la ventana, o3 cancelado, o4 fuera de 90 días, o5 de s2
        "order_items": pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "seller_id": ["s1", "s1", "s1", "s1", "s2"],
            "price": [1000.0, 100.0, 500.0, 700.0, 50.0],
        }),
        "customers": pd.DataFrame({"customer_id": ["c1", "c2", "c3", "c4", "c5"],
                                   "customer_unique_id": ["u1", "u1", "u2", "u3", "u4"]}),
        "order_reviews": pd.DataFrame({"order_id": ["o1"], "review_score": [5]}),
    }
    for nombre, df in tablas.items():
        df.to_parquet(tmp_path / f"{nombre}.parquet", index=False)
    return db.conectar(tmp_path)


def test_embudo_respeta_ventana_madurez_y_estado(con):
    t = embudo.embudo_canal(con, ventana_dias=90, n_boot=50).set_index("canal")
    ps = t.loc["paid_search"]
    assert (ps.leads, ps.cerrados, ps.cerrados_maduros, ps.maduros_que_venden) == (3, 2, 1, 1)
    assert ps.gmv_maduros == pytest.approx(100.0)  # solo o2
    assert ps.tasa_cierre == pytest.approx(2 / 3)
    assert ps.gmv_por_lead == pytest.approx(2 / 3 * 100.0)
    assert ps.ingreso_por_lead == pytest.approx(ps.gmv_por_lead * embudo.COMISION)
    email = t.loc["email"]
    assert email.cerrados == 0 and np.isnan(email.gmv_por_lead)


def test_ic_contiene_la_estimacion(con):
    t = embudo.embudo_canal(con, n_boot=200).set_index("canal")
    ps = t.loc["paid_search"]
    assert ps.gmv_por_lead_p05 <= ps.gmv_por_lead <= ps.gmv_por_lead_p95


def test_gmv_por_lead_sin_maduros_es_nan():
    assert np.isnan(embudo.gmv_por_lead(np.array([True]), np.array([False]), np.array([0.0])))


def test_cohortes_y_kpis(con):
    c = embudo.cohortes_compradores(con, min_compradores=1)
    ene = c[(pd.to_datetime(c.cohorte) == T("2018-01-01")) & (c.mes == 1)].iloc[0]
    assert ene.compradores == 1 and ene.pct_recompra_acum == 1.0  # u1 vuelve en febrero
    k = embudo.kpis_mensuales(con).assign(mes=lambda d: pd.to_datetime(d.mes)).set_index("mes")
    feb = k.loc[T("2018-02-01")]
    assert feb.pedidos == 1 and feb.compradores_nuevos == 0  # o3 cancelado no cuenta; u1 ya era cliente
