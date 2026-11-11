from pathlib import Path

from growth_lab.ingest import nombre_tabla


def test_nombre_tabla_quita_prefijo_y_sufijo():
    assert nombre_tabla(Path("olist_order_items_dataset.csv")) == "order_items"
    assert nombre_tabla(Path("olist_marketing_qualified_leads_dataset.csv")) == "marketing_qualified_leads"
    assert nombre_tabla(Path("product_category_name_translation.csv")) == "product_category_name_translation"
    assert nombre_tabla(Path("hillstrom.csv")) == "hillstrom"


def test_nombre_tabla_con_prefijo_y_doble_extension():
    assert nombre_tabla(Path("campaigns.rda"), "cj_") == "cj_campaigns"
    assert nombre_tabla(Path("journeys.parquet"), "attr_") == "attr_journeys"
    assert nombre_tabla(Path("lenta_dataset.csv.gz")) == "lenta"
