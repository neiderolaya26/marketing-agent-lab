"""Publica en data/processed/ las fuentes cuya licencia permite redistribuirlas (Parquet zstd).

Uso:
    python -m growth_lab.publicar

Qué se publica (decisión 001 y data/README.md):
- Olist (CC BY-NC-SA 4.0): todas las tablas, sin el texto libre de las reseñas, y la
  geolocalización agregada por estado en vez de por prefijo postal y coordenada.
- Hillstrom (uso libre en investigación) y el benchmark de atribución (CC BY 4.0): completos.
- Complete Journey y Lenta: NO (licencia); de ellos solo se publican agregados de cada sesión.
"""

from __future__ import annotations

import duckdb

from growth_lab.ingest import INTERIM
from growth_lab.db import PROCESSED

OLIST = [
    "orders",
    "order_items",
    "order_payments",
    "customers",
    "sellers",
    "products",
    "product_category_name_translation",
    "marketing_qualified_leads",
    "closed_deals",
]

# tabla publicada -> SELECT sobre la tabla de data/interim (`{t}` = read_parquet de la fuente)
CONSULTAS: dict[str, str] = {
    **{t: "SELECT * FROM {t}" for t in OLIST},
    # Sin review_comment_title ni review_comment_message (texto libre).
    "order_reviews": (
        "SELECT review_id, order_id, review_score, review_creation_date, review_answer_timestamp, "
        "review_comment_message IS NOT NULL AS tiene_comentario FROM {t}"
    ),
    # Coordenada media por estado: suficiente para mapas, sin ubicación fina.
    "geolocation": (
        "SELECT geolocation_state AS estado, count(*) AS n_registros, "
        "avg(geolocation_lat) AS lat_media, avg(geolocation_lng) AS lng_media FROM {t} GROUP BY 1 ORDER BY 1"
    ),
    "hillstrom": "SELECT * FROM {t}",
    "attr_journeys": "SELECT * FROM {t}",
    "attr_touchpoints": "SELECT * FROM {t}",
}

NO_PUBLICABLES_PREFIJOS = ("cj_", "lenta")


def publicar() -> dict[str, int]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    filas: dict[str, int] = {}
    for tabla, sql in CONSULTAS.items():
        assert not tabla.startswith(NO_PUBLICABLES_PREFIJOS), f"{tabla} no se puede publicar por licencia"
        fuente = f"read_parquet('{(INTERIM / f'{tabla}.parquet').as_posix()}')"
        destino = (PROCESSED / f"{tabla}.parquet").as_posix()
        con.execute(f"COPY ({sql.format(t=fuente)}) TO '{destino}' (FORMAT parquet, COMPRESSION zstd)").fetchall()
        filas[tabla] = con.execute(f"SELECT count(*) FROM read_parquet('{destino}')").fetchall()[0][0]
    con.close()
    return filas


if __name__ == "__main__":
    for tabla, n in publicar().items():
        print(f"{tabla:<35} {n:>12,}")
    print(f"\nPublicado en: {PROCESSED}")
