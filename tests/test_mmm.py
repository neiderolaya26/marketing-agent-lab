import numpy as np
import pytest

from growth_lab import mmm


def test_adstock_arrastra_y_conserva_el_gasto():
    x = np.zeros((60, 1))
    x[0] = 100.0
    a = mmm.adstock(x, np.array([0.5]))
    assert a[0, 0] == pytest.approx(50.0)  # (1 − θ) del gasto entra la primera semana
    assert a[1, 0] == pytest.approx(25.0)  # y se va apagando a la tasa θ
    assert a.sum() == pytest.approx(100.0)  # normalizado: el total pautado se conserva


def test_saturacion_rendimientos_decrecientes():
    s = mmm.saturacion(np.array([10.0, 20.0, 30.0]), np.array([20.0]))
    assert s[1] - s[0] > s[2] - s[1] > 0


def test_generador_endogeno_correlaciona_gasto_con_demanda():
    limpio, endo = mmm.generar("limpio"), mmm.generar("endogeno")
    r = lambda d: np.corrcoef(d.gasto_paid_search, d.verdad_base)[0, 1]  # noqa: E731
    assert abs(r(limpio)) < 0.2 < r(endo)


def test_mmm_recupera_roi_de_canales_grandes_en_escenario_limpio():
    df = mmm.generar("limpio")
    gasto = df[[f"gasto_{c}" for c in mmm.CANALES]].to_numpy()
    roi_v, _ = mmm.roi_verdadero(gasto)
    m = mmm.MMM("debil").ajustar(gasto, df.ventas.to_numpy(), n_inicios=2)
    roi_e, _ = mmm.roi_y_marginal(gasto, m.theta, m.lam, m.beta)
    assert roi_e[0] == pytest.approx(roi_v[0], rel=0.3)  # TV, el canal con más gasto


def test_calibracion_acerca_el_roi_del_email_al_experimento():
    df = mmm.generar("limpio")
    gasto = df[[f"gasto_{c}" for c in mmm.CANALES]].to_numpy()
    roi_v, _ = mmm.roi_verdadero(gasto)
    i = mmm.CANALES.index("email")
    m = mmm.MMM("debil_calibrado", roi_email=roi_v[i]).ajustar(gasto, df.ventas.to_numpy(), n_inicios=2)
    assert mmm.roi_y_marginal(gasto, m.theta, m.lam, m.beta)[0][i] == pytest.approx(roi_v[i], rel=0.15)


def test_optimizador_respeta_total_y_limites():
    df = mmm.generar("limpio")
    gasto = df[[f"gasto_{c}" for c in mmm.CANALES]].to_numpy()
    v = mmm.VERDAD["theta"], mmm.VERDAD["lam"], mmm.VERDAD["beta"]
    mult = mmm.optimizar(gasto, *v)
    total = gasto[-52:].sum(axis=0)
    assert (mult * total).sum() == pytest.approx(total.sum(), rel=1e-4)
    assert np.all(mult >= 1 - mmm.LIMITE - 1e-6) and np.all(mult <= 1 + mmm.LIMITE + 1e-6)
    assert mmm.ventas_medios(gasto, mult, *v) >= mmm.ventas_medios(gasto, np.ones(len(mmm.CANALES)), *v)
