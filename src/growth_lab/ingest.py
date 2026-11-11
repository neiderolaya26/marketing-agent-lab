"""Ingesta: datos crudos -> un Parquet por tabla en data/interim (zstd).

Uso:
    python -m growth_lab.ingest

Fuentes (ver data/README.md):
- Olist E-Commerce + Marketing Funnel (CSV) -> tablas sin prefijo (`orders`, `marketing_qualified_leads`, ...)
- The Complete Journey (R .rda/.rds)        -> prefijo `cj_`
- Synthetic attribution benchmark (Parquet) -> prefijo `attr_`
- Hillstrom (CSV) y Lenta (CSV.gz)          -> `hillstrom`, `lenta`

No se guarda ninguna base DuckDB: `growth_lab.db.conectar()` crea vistas en memoria
sobre estos Parquet, así los datos viven una sola vez en disco.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CRUDO = RAIZ / "data" / "raw"
INTERIM = RAIZ / "data" / "interim"

# Columnas de fecha de Olist (se convierten a TIMESTAMP).
FECHAS: dict[str, list[str]] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
    "marketing_qualified_leads": ["first_contact_date"],
    "closed_deals": ["won_date"],
}


def nombre_tabla(archivo: Path, prefijo: str = "") -> str:
    """`olist_order_items_dataset.csv` -> `order_items`; `campaigns.rda` con prefijo `cj_` -> `cj_campaigns`."""
    n = archivo.name.split(".")[0]
    n = n.removeprefix("olist_").removesuffix("_dataset")
    return f"{prefijo}{n}"


def fuentes() -> list[tuple[str, Path]]:
    """(tabla, archivo) de todas las fuentes presentes en data/raw."""
    pares: list[tuple[str, Path]] = []
    for a in sorted((CRUDO / "olist").glob("*.csv")):
        pares.append((nombre_tabla(a), a))
    for a in sorted((CRUDO / "completejourney").glob("*.rd[as]")):
        pares.append((nombre_tabla(a, "cj_"), a))
    for a in sorted((CRUDO / "atribucion").glob("*.parquet")):
        pares.append((nombre_tabla(a, "attr_"), a))
    for carpeta in ("hillstrom", "lenta"):
        for a in sorted((CRUDO / carpeta).glob("*.csv*")):
            pares.append((carpeta, a))
    return pares


def consulta_origen(tabla: str, archivo: Path) -> str:
    """SELECT que lee el archivo crudo y convierte las fechas de Olist."""
    if archivo.suffix == ".parquet":
        origen = f"read_parquet('{archivo.as_posix()}')"
    else:
        origen = f"read_csv_auto('{archivo.as_posix()}', header=true, sample_size=-1)"
    fechas = FECHAS.get(tabla, [])
    if not fechas:
        return f"SELECT * FROM {origen}"
    reemplazos = ", ".join(f"CAST({c} AS TIMESTAMP) AS {c}" for c in fechas)
    return f"SELECT * REPLACE ({reemplazos}) FROM {origen}"


def _leer_r(archivo: Path) -> pd.DataFrame:
    import pyreadr

    return next(iter(pyreadr.read_r(str(archivo)).values()))


def ingerir() -> dict[str, int]:
    INTERIM.mkdir(parents=True, exist_ok=True)
    filas: dict[str, int] = {}
    con = duckdb.connect()
    for tabla, archivo in fuentes():
        destino = (INTERIM / f"{tabla}.parquet").as_posix()
        es_r = archivo.suffix in {".rda", ".rds"}
        if es_r:
            con.register("_df", _leer_r(archivo))
        consulta = "SELECT * FROM _df" if es_r else consulta_origen(tabla, archivo)
        con.execute(f"COPY ({consulta}) TO '{destino}' (FORMAT parquet, COMPRESSION zstd)").fetchall()
        if es_r:
            con.unregister("_df")
        filas[tabla] = con.execute(f"SELECT count(*) FROM read_parquet('{destino}')").fetchall()[0][0]
    con.close()
    return filas


if __name__ == "__main__":
    for tabla, n in ingerir().items():
        print(f"{tabla:<35} {n:>12,}")
    print(f"\nParquet en: {INTERIM}")
