"""Acceso a los datos: DuckDB en memoria con una vista por cada Parquet.

    from growth_lab import db
    con = db.conectar()
    con.sql("SELECT origin, count(*) FROM marketing_qualified_leads GROUP BY 1").df()

Por defecto lee data/interim (todas las tablas). Con `carpeta=PROCESSED` lee los
derivados publicados en el repo, que es lo que usan el agente y los tests de punta a punta.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

RAIZ = Path(__file__).resolve().parents[2]
INTERIM = RAIZ / "data" / "interim"
PROCESSED = RAIZ / "data" / "processed"


def tablas(carpeta: Path = INTERIM) -> dict[str, Path]:
    return {p.stem: p for p in sorted(carpeta.glob("*.parquet"))}


# Decisión 020: lo que solo existe porque los datos son sintéticos (la "verdad" del simulador y del benchmark)
# no lo ve un modelo. Se quedan las métricas de validación (error de cada método, captura del óptimo), que son
# hallazgos del lab, y los resultados observados en datos reales (p. ej. el gasto real del CLV).
OCULTAR_TABLAS = {"agt_demo", "evl_resultados", "evl_resumen"}  # respuestas del propio agente, con la verdad
OCULTAR_COLUMNAS = {
    "mmm_series": r"^verdad_",
    "mmm_roi": r"_verdad$",
    "mmm_error": r"_verdad_",
    "mmm_presupuesto_resumen": r"^(ganancia_real|utilidad_extra)",
    "mmm_robustez": r"^ganancia_",
    "atr_participacion": r"^verdad$",
    "atr_costo_por_conversion": r"^verdad$",
    "attr_journeys": r"^gt_",
}
FILTRAR_FILAS = {"mmm_presupuesto": "plan <> 'verdad'", "mmm_presupuesto_resumen": "plan <> 'verdad'"}


def _select_sin_verdad(con: duckdb.DuckDBPyConnection, nombre: str, ruta: Path) -> str:
    import re

    fuente = f"read_parquet('{ruta.as_posix()}')"
    columnas = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM {fuente}").fetchall()]
    patron = OCULTAR_COLUMNAS.get(nombre)
    visibles = [c for c in columnas if not (patron and re.search(patron, c))]
    donde = f" WHERE {FILTRAR_FILAS[nombre]}" if nombre in FILTRAR_FILAS else ""
    return f"SELECT {', '.join(chr(34) + c + chr(34) for c in visibles)} FROM {fuente}{donde}"


def conectar(carpeta: Path = INTERIM, *, solo_lectura: bool = True, sin_verdad: bool = False) -> duckdb.DuckDBPyConnection:
    """Conexión en memoria con `CREATE VIEW <tabla>` sobre cada Parquet de la carpeta.

    Con `sin_verdad=True` (la conexión del agente, decisión 020) las tablas se cargan en memoria sin las
    columnas, filas y tablas de verdad, y la conexión no puede leer ningún archivo: así un
    `read_parquet(...)` escrito por el modelo tampoco llega a la verdad."""
    con = duckdb.connect()
    for nombre, ruta in tablas(carpeta).items():
        if not sin_verdad:
            con.execute(f"CREATE VIEW {nombre} AS SELECT * FROM read_parquet('{ruta.as_posix()}')").fetchall()
        elif nombre not in OCULTAR_TABLAS:
            con.execute(f"CREATE TABLE {nombre} AS {_select_sin_verdad(con, nombre, ruta)}").fetchall()
    if sin_verdad:
        con.execute("SET enable_external_access = false").fetchall()
        con.execute("SET lock_configuration = true").fetchall()
    elif solo_lectura:
        # Una consulta (p. ej. generada por el agente) solo puede leer esta carpeta:
        # no escribe archivos, no lee fuera de ella y no puede deshacer la configuración.
        con.execute(f"SET allowed_directories = ['{carpeta.as_posix()}/']").fetchall()
        con.execute("SET enable_external_access = false").fetchall()
        con.execute("SET lock_configuration = true").fetchall()
    return con


class ConsultaNoPermitida(ValueError):
    """La consulta no es una única sentencia SELECT."""


def validar_select(sql: str) -> None:
    """Acepta solo una sentencia SELECT (incluye WITH y DESCRIBE); rechaza COPY, DDL, DML, PRAGMA, SET."""
    sentencias = duckdb.extract_statements(sql)
    if len(sentencias) != 1:
        raise ConsultaNoPermitida(f"Se esperaba 1 sentencia y llegaron {len(sentencias)}")
    if sentencias[0].type != duckdb.StatementType.SELECT:
        raise ConsultaNoPermitida(f"Solo se permiten consultas SELECT, no {sentencias[0].type.name}")


def consultar(con: duckdb.DuckDBPyConnection, sql: str, limite: int = 1000):
    """Ejecuta una consulta de solo lectura y devuelve un DataFrame con a lo sumo `limite` filas.

    Es la puerta que usa el agente (text-to-SQL): dos barreras, el parser (solo SELECT)
    y la conexión aislada de `conectar()` (solo lee su carpeta).
    """
    validar_select(sql)
    return con.sql(sql).limit(limite).df()
