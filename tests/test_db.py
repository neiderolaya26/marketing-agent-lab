import duckdb
import pytest

from growth_lab import db


@pytest.fixture
def carpeta(tmp_path):
    con = duckdb.connect()
    con.execute(f"COPY (SELECT 1 AS a UNION ALL SELECT 2) TO '{(tmp_path / 'uno.parquet').as_posix()}' (FORMAT parquet)").fetchall()
    con.close()
    return tmp_path


def test_crea_una_vista_por_parquet(carpeta):
    con = db.conectar(carpeta)
    assert con.sql("SELECT sum(a) FROM uno").fetchall() == [(3,)]


def test_solo_lectura_no_escribe_ni_lee_fuera(carpeta, tmp_path_factory):
    afuera = tmp_path_factory.mktemp("afuera")
    con = db.conectar(carpeta)
    with pytest.raises(db.ConsultaNoPermitida):
        db.consultar(con, f"COPY uno TO '{(carpeta / 'x.csv').as_posix()}'")
    with pytest.raises(db.ConsultaNoPermitida):
        db.consultar(con, "SELECT 1; DROP VIEW uno")
    with pytest.raises(duckdb.Error):
        db.consultar(con, f"SELECT * FROM read_csv('{(afuera / 'y.csv').as_posix()}')")
    with pytest.raises(duckdb.Error):
        con.sql("SET enable_external_access = true").fetchall()
    assert not (carpeta / "x.csv").exists()
    assert db.consultar(con, "WITH t AS (SELECT * FROM uno) SELECT count(*) AS n FROM t")["n"].iloc[0] == 2


@pytest.mark.skipif(not db.tablas(db.PROCESSED), reason="sin data/processed")
def test_datos_publicados_no_traen_texto_libre_ni_fuentes_restringidas():
    publicadas = db.tablas(db.PROCESSED)
    assert not any(t.startswith(("cj_", "lenta")) for t in publicadas)
    con = db.conectar(db.PROCESSED)
    columnas = {c for (c,) in con.sql("SELECT column_name FROM (DESCRIBE order_reviews)").fetchall()}
    assert "review_comment_message" not in columnas
    assert con.sql("SELECT count(*) FROM geolocation").fetchall()[0][0] < 100
