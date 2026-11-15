"""Registro de ejecuciones del agente en SQLite (decisión 017).

Cada pregunta deja:
- una fila en `ejecuciones`: origen (cli, demo, eval), versión, pregunta, especialistas, estado, respuesta,
  propuesta y simulación, tokens, costo, tiempo y, cuando la hay, la firma humana;
- una fila por llamada a herramienta en `pasos` (especialista, herramienta, entrada, ok/error).

La base vive en data/agente/ejecuciones.db y no se sube a git: es la bitácora para auditar quién preguntó
qué, con qué cifras se respondió y quién aprobó cada movimiento de plata.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from growth_lab import db

RUTA = db.RAIZ / "data" / "agente" / "ejecuciones.db"

ESQUEMA = """
CREATE TABLE IF NOT EXISTS ejecuciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    origen TEXT NOT NULL,
    version TEXT NOT NULL,
    modelo TEXT NOT NULL,
    hilo TEXT,
    pregunta TEXT NOT NULL,
    especialistas TEXT,
    estado TEXT NOT NULL,
    respuesta TEXT,
    propuesta TEXT,
    simulacion TEXT,
    fallas TEXT,
    llamadas_llm INTEGER,
    tokens_entrada INTEGER,
    tokens_salida INTEGER,
    costo_usd REAL,
    segundos REAL,
    aprobacion TEXT
);
CREATE TABLE IF NOT EXISTS pasos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ejecucion_id INTEGER NOT NULL REFERENCES ejecuciones(id),
    especialista TEXT NOT NULL,
    herramienta TEXT NOT NULL,
    entrada TEXT,
    ok INTEGER NOT NULL,
    error TEXT
);
"""


@contextmanager
def conectar(ruta: Path = RUTA) -> Iterator[sqlite3.Connection]:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ruta)
    con.executescript(ESQUEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _json(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


def resultado(estado: dict) -> dict:
    """Lo esencial de un estado del grafo, haya terminado o esté esperando la firma."""
    pendiente = "__interrupt__" in estado
    c = estado.get("coordinador", {})
    f = estado.get("final") or {}
    return {
        "estado": "pendiente de aprobación" if pendiente else f.get("estado", "sin terminar"),
        "respuesta": f.get("respuesta", c.get("respuesta", "")),
        "cifras_clave": f.get("cifras_clave", c.get("cifras_clave", [])),
        "propuesta": c.get("propuesta", []) if pendiente else f.get("propuesta", []),
        "especialistas": sorted(estado.get("especialistas", {})),
        "simulacion": estado.get("simulacion", {}),
        "fallas": estado.get("fallas", []),
    }


def guardar(con: sqlite3.Connection, estado: dict, *, origen: str, version: str, modelo: str,
            uso: dict | None = None, segundos: float | None = None) -> int:
    r = resultado(estado)
    uso = uso or {}
    cur = con.execute(
        "INSERT INTO ejecuciones (fecha, origen, version, modelo, hilo, pregunta, especialistas, estado, respuesta,"
        " propuesta, simulacion, fallas, llamadas_llm, tokens_entrada, tokens_salida, costo_usd, segundos)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), origen, version, modelo, estado.get("hilo"),
         estado.get("pregunta", ""), ", ".join(r["especialistas"]), r["estado"], r["respuesta"], _json(r["propuesta"]),
         _json(r["simulacion"]), _json(r["fallas"]), uso.get("llamadas"), uso.get("tokens_entrada"),
         uso.get("tokens_salida"), uso.get("costo_usd"), segundos))
    eid = cur.lastrowid
    for nombre, esp in estado.get("especialistas", {}).items():
        for h in esp.get("herramientas", []):
            con.execute("INSERT INTO pasos (ejecucion_id, especialista, herramienta, entrada, ok, error)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (eid, nombre, h["herramienta"], _json(h["entrada"]), int(h["ok"]), h.get("error")))
    return eid


def actualizar_firma(con: sqlite3.Connection, hilo: str, final: dict) -> None:
    con.execute("UPDATE ejecuciones SET estado = ?, aprobacion = ? WHERE hilo = ?",
                (final["estado"], _json(final.get("aprobacion")), hilo))


def leer(con: sqlite3.Connection, tabla: str = "ejecuciones") -> pd.DataFrame:
    assert tabla in {"ejecuciones", "pasos"}
    return pd.read_sql_query(f"SELECT * FROM {tabla}", con)


def resumen(con: sqlite3.Connection) -> pd.DataFrame:
    """Por origen y versión: cuántas preguntas, costo, tiempo y cuántas esperan o recibieron firma."""
    return pd.read_sql_query("""
        SELECT origen, version, count(*) AS preguntas, round(sum(costo_usd), 3) AS costo_usd,
               round(avg(segundos), 1) AS segundos_promedio,
               sum(estado = 'pendiente de aprobación') AS pendientes,
               sum(estado = 'aprobada') AS aprobadas, sum(estado = 'rechazada') AS rechazadas
        FROM ejecuciones GROUP BY origen, version ORDER BY version, origen""", con)
