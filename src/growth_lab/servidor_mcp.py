"""Servidor MCP del lab de marketing (decisión 017).

Expone las mismas herramientas que usan los especialistas del agente (sesión 11) para que cualquier cliente
MCP (Claude Code, Claude Desktop, otro agente) las use. Son código sobre data/processed: no llaman a ningún
modelo, no necesitan API key y todas son de solo lectura (el simulador no cambia nada, solo calcula).

Uso:
    python -m growth_lab.servidor_mcp      # transporte stdio (lo registra .mcp.json en la raíz del repo)
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from growth_lab import db, mmm
from growth_lab.agente import DECISIONES, Herramientas, sin_verdad

servidor = MCPServer(
    name="marketing-lab",
    title="Lab de marketing y growth",
    instructions="Herramientas de solo lectura sobre los resultados del lab (Olist, Complete Journey agregado, "
                 "Hillstrom, benchmark de atribución y MMM sintético). Las decisiones vigentes están en el recurso "
                 "marketing://decisiones. Para SQL, primero usa tablas.",
)

SOLO_LECTURA = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


class _Contexto:
    herramientas: Herramientas | None = None

    @classmethod
    def h(cls) -> Herramientas:
        if cls.herramientas is None:
            cls.herramientas = Herramientas()
        return cls.herramientas


class Cambio(BaseModel):
    canal: Literal["tv", "paid_search", "social", "display", "email"]
    cambio_pct: float = Field(ge=-100, le=100, description="Cambio de gasto en %, entre -30 y 30")


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def tablas() -> dict[str, Any]:
    """Tablas Parquet publicadas (data/processed) con sus columnas."""
    return _Contexto.h().tablas()


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def consultar_sql(consulta: str) -> dict[str, Any]:
    """Ejecuta una sola sentencia SELECT de DuckDB sobre las tablas publicadas (máx. 200 filas)."""
    try:
        return _Contexto.h().sql(consulta)
    except db.ConsultaNoPermitida as e:
        raise ToolError(str(e)) from None


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def valor_canales() -> dict[str, Any]:
    """Valor de un lead, CPL de equilibrio y tasa de cierre por canal de Olist (decisiones 002 y 004)."""
    return _Contexto.h().valor_canales()


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def lead_scoring() -> dict[str, Any]:
    """Métricas de los 6 modelos de lead scoring y deciles del campeón (decisión 005)."""
    return _Contexto.h().lead_scoring()


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def segmentos_clv() -> dict[str, Any]:
    """Segmentos RFM con CLV a 13 semanas, churn parcial y gasto en riesgo (decisiones 003, 006 y 007)."""
    return _Contexto.h().segmentos_clv()


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def experimentos_uplift() -> dict[str, Any]:
    """Economía del A/B de Hillstrom y valor de cada política de envío (decisiones 009 y 010)."""
    return _Contexto.h().experimentos_uplift()


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def campanas() -> dict[str, Any]:
    """Efecto de las campañas tipo A con DiD doble robusto y placebos (decisiones 011 y 012)."""
    return _Contexto.h().campanas()


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def atribucion() -> dict[str, Any]:
    """Participación y costo por conversión por canal según cada método de atribución (decisión 013)."""
    return _Contexto.h().atribucion()


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def roi_mmm() -> dict[str, Any]:
    """ROI promedio y marginal por canal del MMM elegido y su plan de presupuesto (decisión 015)."""
    return _Contexto.h().mmm()


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def plan_utilidad() -> dict[str, Any]:
    """Reparto que maximiza la utilidad (cada canal ±30%, total sin subir, búsqueda congelada) (decisión 016)."""
    return sin_verdad(_Contexto.h().plan_utilidad())


@servidor.tool(structured_output=True, annotations=SOLO_LECTURA)
def simular_reparto(cambios: list[Cambio]) -> dict[str, Any]:
    """Ventas y utilidad extra de un reparto según el MMM, y si cumple las reglas de la decisión 016.
    No ejecuta nada: solo calcula."""
    if not cambios:
        raise ToolError("Pasa al menos un cambio")
    return sin_verdad(_Contexto.h().simular_reparto([c.model_dump() for c in cambios]))


@servidor.resource("marketing://decisiones", name="decisiones", mime_type="text/markdown",
                   description="Decisiones vigentes del lab (001 en adelante), en primera persona.")
def decisiones() -> str:
    return DECISIONES.read_text(encoding="utf-8")


@servidor.resource("marketing://reglas-reparto", name="reglas-reparto", mime_type="text/plain",
                   description="Reglas duras que debe cumplir cualquier reparto de presupuesto.")
def reglas_reparto() -> str:
    return (f"Canales: {', '.join(mmm.CANALES)}. Cada canal entre −{mmm.LIMITE:.0%} y +{mmm.LIMITE:.0%}. "
            "El total puede bajar pero no subir. paid_search no sube hasta la prueba geo (decisión 015). "
            f"Objetivo: utilidad = ventas extra × {mmm.MARGEN:.0%} − gasto extra (decisión 016). "
            "Cualquier movimiento de plata lo aprueba una persona.")


def main() -> None:
    servidor.run("stdio")


if __name__ == "__main__":
    main()
