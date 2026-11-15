"""El agente sin llamar a la API: un cliente falso devuelve respuestas guionadas."""
from types import SimpleNamespace as NS

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from growth_lab import agente as ag
from growth_lab import db

USO = NS(input_tokens=10, output_tokens=5)


class ClienteFalso:
    def __init__(self, ruta, propuestas):
        self.ruta, self.propuestas = ruta, list(propuestas)
        self.beta = NS(messages=NS(create=self._create, parse=self._parse))
        self.herramientas_usadas = []

    def _create(self, **kw):
        ultimo = kw["messages"][-1]["content"]
        if isinstance(ultimo, str):  # primer turno del especialista: usa una herramienta propia
            nombre = next(t["name"] for t in kw["tools"] if t["name"] not in {"tablas", "consultar_sql", "leer_decision"})
            self.herramientas_usadas.append(nombre)
            return NS(stop_reason="tool_use", usage=USO,
                      content=[NS(type="tool_use", id="t1", name=nombre, input={})])
        return NS(stop_reason="end_turn", usage=USO, content=[NS(type="text", text="análisis con cifras")])

    def _parse(self, **kw):
        if kw["output_format"] is ag.Ruta:
            return NS(stop_reason="end_turn", usage=USO, parsed_output=self.ruta)
        cambios = self.propuestas.pop(0) if self.propuestas else []
        return NS(stop_reason="end_turn", usage=USO, parsed_output=ag.Respuesta(
            respuesta="respuesta", cifras_clave=["x"], justificacion_propuesta="j",
            propuesta=[ag.Cambio(canal=c, cambio_pct=p) for c, p in cambios]))


@pytest.fixture(scope="module")
def herr():
    return ag.Herramientas()


def ruta(*especialistas, pide_reparto=True):
    return ag.Ruta(encargos=[ag.Encargo(especialista=e, pregunta="?") for e in especialistas], fuera_de_alcance=False,
                   pide_reparto=pide_reparto)


def test_sql_solo_lectura(herr):
    assert herr.sql("SELECT count(*) AS n FROM embudo_canal")["datos"][0]["n"] > 0
    with pytest.raises(db.ConsultaNoPermitida):
        herr.sql("DELETE FROM embudo_canal")


def test_reglas_del_reparto():
    assert ag.validar_reparto([{"canal": "tv", "cambio_pct": 10}, {"canal": "social", "cambio_pct": -16}]) == []
    assert "límite" in ag.validar_reparto([{"canal": "tv", "cambio_pct": 35}])[0]
    assert "total" in ag.validar_reparto([{"canal": "tv", "cambio_pct": 20}])[0]
    assert ag.validar_reparto([{"canal": "tv", "cambio_pct": -20}]) == []  # bajar el total sí se puede (016)
    assert "prueba geo" in ag.validar_reparto([{"canal": "paid_search", "cambio_pct": 5}])[0]  # congelado (016)
    assert ag.validar_reparto([{"canal": "paid_search", "cambio_pct": -10}, {"canal": "tv", "cambio_pct": 9}]) == []


def test_especialistas_reciben_decisiones_vigentes(herr):
    t = herr.resumen_decisiones(ag.DECISIONES_AREA["retencion"])
    assert "007 — " in t and "Decisión:" in t and "Por qué" not in t


def test_consulta_sin_propuesta_no_pide_aprobacion(herr):
    cli = ClienteFalso(ruta("captacion", "retencion"), [[]])
    g = ag.construir(cli, herr, checkpointer=InMemorySaver())
    e = ag.preguntar(g, "¿qué canal trae mejores leads?")
    assert "__interrupt__" not in e
    assert e["final"]["estado"] == "respondida"
    assert set(e["final"]["especialistas"]) == {"captacion", "retencion"}
    assert set(cli.herramientas_usadas) == {"valor_canales", "segmentos_clv"}


def test_propuesta_valida_espera_aprobacion_humana(herr):
    cli = ClienteFalso(ruta("presupuesto"), [[("tv", 10), ("social", -16)]])
    g = ag.construir(cli, herr, checkpointer=InMemorySaver())
    e = ag.preguntar(g, "¿cómo reparto el presupuesto?")
    pendiente = e["__interrupt__"][0].value
    assert pendiente["simulacion"]["fallas"] == []
    f = ag.aprobar(g, e["hilo"], False, "primero la prueba geo")["final"]
    assert f["estado"] == "rechazada" and f["aprobacion"]["comentario"] == "primero la prueba geo"


def test_propuesta_invalida_vuelve_al_coordinador_y_se_descarta(herr):
    cli = ClienteFalso(ruta("presupuesto"), [[("tv", 50)], [("tv", 45)]])
    g = ag.construir(cli, herr, checkpointer=InMemorySaver())
    e = ag.preguntar(g, "sube todo a TV")
    assert "__interrupt__" not in e
    assert e["final"]["estado"] == "propuesta descartada por reglas"
    assert e["intentos"] == ag.INTENTOS


def test_fuera_de_alcance_no_llama_especialistas(herr):
    cli = ClienteFalso(ag.Ruta(encargos=[], fuera_de_alcance=True, pide_reparto=False), [[]])
    e = ag.preguntar(ag.construir(cli, herr, checkpointer=InMemorySaver()), "¿quién ganó el mundial?")
    assert e["final"]["especialistas"] == [] and cli.herramientas_usadas == []


def test_plan_utilidad_respeta_reglas_y_supera_hoy(herr):
    p = herr.plan_utilidad()
    assert p["simulacion"]["fallas"] == []
    assert p["simulacion"]["utilidad_extra_segun_modelo"] > 0
    assert next(c for c in p["cambios"] if c["canal"] == "paid_search")["cambio_pct"] <= 0


def test_propuesta_no_pedida_se_descarta(herr):
    cli = ClienteFalso(ruta("presupuesto", pide_reparto=False), [[("tv", -10)]])
    e = ag.preguntar(ag.construir(cli, herr, checkpointer=InMemorySaver()), "resumen ejecutivo")
    assert "__interrupt__" not in e and e["final"]["estado"] == "respondida" and e["final"]["propuesta"] == []
    assert e["coordinador"]["propuesta_descartada"] == [{"canal": "tv", "cambio_pct": -10.0}]
