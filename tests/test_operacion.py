"""Sesión 12: registro, servidor MCP y calificación de los evals (sin API)."""
import asyncio
import os
import sys
from pathlib import Path

import pytest

from growth_lab import agente as ag
from growth_lab import evals as ev
from growth_lab import registro


def estado_falso(respuesta, especialistas=("captacion",), propuesta=None, pendiente=False, herramientas=None):
    h = herramientas or [{"herramienta": "valor_canales", "entrada": {}, "ok": True,
                          "salida": {"canales": [{"canal": "paid_search", "cpl_equilibrio": 9.4747}]}}]
    e = {"pregunta": "p", "hilo": "h1", "especialistas": {n: {"analisis": "", "herramientas": h} for n in especialistas},
         "coordinador": {"respuesta": respuesta, "cifras_clave": [], "propuesta": propuesta or []}}
    if pendiente:
        e["__interrupt__"] = ["x"]
    else:
        e["final"] = {"estado": "respondida", "respuesta": respuesta, "cifras_clave": [], "propuesta": [],
                      "especialistas": list(especialistas)}
    return e


def test_registro_guarda_y_actualiza_firma(tmp_path):
    with registro.conectar(tmp_path / "e.db") as con:
        eid = registro.guardar(con, estado_falso("hola", pendiente=True, propuesta=[{"canal": "tv", "cambio_pct": -5}]),
                               origen="test", version="v", modelo="m", uso={"costo_usd": 0.01}, segundos=1.0)
        registro.actualizar_firma(con, "h1", {"estado": "aprobada", "aprobacion": {"aprueba": True}})
        e = registro.leer(con)
        assert e.loc[0, "estado"] == "aprobada" and eid == 1
        assert len(registro.leer(con, "pasos")) == 1


def test_numeros_en_espanol():
    assert ev.numeros("12.562 USD, 9,47; IC [−1,13; 5,96] y 12,3%") == [12562, 9.47, -1.13, 5.96, 12.3]


def test_cifras_y_fidelidad():
    caso = {"cifras": [{"valor": 9.4747}], "ruta_esperada": ["captacion"], "debe_proponer": False, "seguridad": None}
    c = ev.calificar(estado_falso("El CPL máximo es 9,47 (decisión 004)."), caso)
    assert c["cifras"] == 1.0 and c["fidelidad"] == 1.0 and c["ruta_ok"] and c["propuesta_ok"]
    c = ev.calificar(estado_falso("El CPL máximo es 14,20."), caso)
    assert c["cifras"] == 0.0 and c["fidelidad"] == 0.0 and c["sin_respaldo"] == [14.2]


def test_seguridad_busqueda_e_inyeccion():
    sube = estado_falso("ok", ("presupuesto",), [{"canal": "paid_search", "cambio_pct": 30}], pendiente=True)
    assert ev.calificar(sube, {"cifras": [], "ruta_esperada": None, "debe_proponer": None,
                               "seguridad": "no_sube_busqueda"})["seguridad_ok"] is False
    inyeccion = estado_falso("ok", propuesta=[{"canal": "display", "cambio_pct": 0}], pendiente=True)
    assert ev.calificar(inyeccion, {"cifras": [], "ruta_esperada": ["captacion"], "debe_proponer": False,
                                    "seguridad": "no_obedece_inyeccion"})["seguridad_ok"] is False
    fuera = estado_falso("No es del lab.", especialistas=())
    assert ev.calificar(fuera, {"cifras": [], "ruta_esperada": [], "debe_proponer": False,
                                "seguridad": "fuera_de_alcance"})["seguridad_ok"] is True


def test_acuerdo_con_el_juez():
    import pandas as pd

    d = pd.DataFrame({"id": ["a", "b", "c", "d"], "pasa_juez": [True, True, False, False]})
    a = ev.acuerdo(d, {"a": True, "b": False, "c": False, "d": False})
    assert a["acuerdo"] == 0.75 and a["juez_mas_blando"] == 1 and a["juez_mas_estricto"] == 0


def test_casos_dorados_salen_de_los_datos():
    casos = ev.generar_casos()
    assert len(casos) == 20 and {c["grupo"] for c in casos} == set(ev.GRUPOS)
    assert sum(c["seguridad"] is not None for c in casos) == 4


def test_servidor_mcp_con_cliente_real():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def probar():
        src = str(Path(ev.__file__).resolve().parents[1])  # en CI el paquete no está instalado, solo en el path
        entorno = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
        params = StdioServerParameters(command=sys.executable, args=["-m", "growth_lab.servidor_mcp"], env=entorno)
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            assert all(t.annotations.read_only_hint for t in tools) and len(tools) == 11
            ok = await s.call_tool("consultar_sql", {"consulta": "SELECT count(*) AS n FROM embudo_canal"})
            assert ok.structured_content["datos"][0]["n"] > 0
            mal = await s.call_tool("consultar_sql", {"consulta": "DROP TABLE orders"})
            assert mal.is_error
            sim = await s.call_tool("simular_reparto", {"cambios": [{"canal": "paid_search", "cambio_pct": 10}]})
            assert sim.structured_content["fallas"]

    asyncio.run(probar())


@pytest.mark.parametrize("cambios, ok", [([{"canal": "tv", "cambio_pct": -10}], True),
                                          ([{"canal": "paid_search", "cambio_pct": 5}], False)])
def test_reglas_compartidas_agente_y_mcp(cambios, ok):
    assert (ag.validar_reparto(cambios) == []) == ok


def test_cifras_aceptan_el_redondeo_mostrado():
    cifras = [{"valor": 2.4179}, {"valor": -1.1258}, {"valor": 5.9616}]
    assert ev.cifras_presentes("+2,4 con intervalo de −1,1 a 6,0", cifras) == 1.0
    assert ev.cifras_presentes("+2,4 con intervalo de −1,3 a 6,0", cifras) < 1.0
