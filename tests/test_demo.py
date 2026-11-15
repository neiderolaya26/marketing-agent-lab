"""La demo de Streamlit corre sin LLM; necesita el modelo local (modelos/) o descargarlo de Hugging Face."""
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def test_demo_reproduce_el_plan_aprobado():
    pytest.importorskip("streamlit")
    if not (RAIZ / "modelos" / "lead_scoring_logistica.joblib").exists():
        pytest.importorskip("huggingface_hub")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RAIZ / "demo" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception and len(at.tabs) == 5
    at.radio[0].set_value("Aprobado (v3)").run()
    valores = {m.label: m.value for m in at.metric}
    assert valores["Utilidad extra según el MMM"] == "+274" and valores["Utilidad extra real (simulador)"] == "+324"
    at.slider(key="Aprobado (v3)-tv").set_value(30).run()
    assert at.error and "total" in at.error[0].value
