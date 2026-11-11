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


def conectar(carpeta: Path = INTERIM, *, solo_lectura: bool = True) -> duckdb.DuckDBPyConnection:
    """Conexión en memoria con `CREATE VIEW <tabla>` sobre cada Parquet de la carpeta."""
    con = duckdb.connect()
    for nombre, ruta in tablas(carpeta).items():
        con.execute(f"CREATE VIEW {nombre} AS SELECT * FROM read_parquet('{ruta.as_posix()}')").fetchall()
    if solo_lectura:
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
