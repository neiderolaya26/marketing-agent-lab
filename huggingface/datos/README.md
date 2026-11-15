---
license: cc-by-nc-sa-4.0
language:
- es
tags:
- marketing
- growth
- causal-inference
- attribution
- marketing-mix-modeling
pretty_name: marketing-agent-lab data
size_categories:
- 1M<n<10M
---

# marketing-agent-lab · datos

Los Parquet (zstd) publicables del [marketing-agent-lab](https://github.com/neiderolaya26/marketing-agent-lab). Son los mismos de `data/processed/` en el repositorio. Se consultan con DuckDB, sin una base de datos aparte.

## Fuentes y licencias

| Tablas | Fuente | Licencia | Qué se quitó |
|---|---|---|---|
| `orders`, `order_items`, `order_payments`, `customers`, `sellers`, `products`, `marketing_qualified_leads`, `closed_deals`, `order_reviews`, `geolocation`, `product_category_name_translation` | Olist + Marketing Funnel (Kaggle) | CC BY-NC-SA 4.0 | Texto libre de las reseñas; geolocalización agregada por estado |
| `hillstrom` | MineThatData E-Mail Analytics Challenge (Kevin Hillstrom, 2008) | Publicado por el autor para análisis libre | — |
| `attr_journeys`, `attr_touchpoints` | Synthetic attribution benchmark (Hugging Face) | CC BY 4.0 | — |
| `embudo_*`, `segmentos_*`, `leads_*`, `canales_rentabilidad`, `clv_*`, `exp_*`, `upl_*`, `cmp_*`, `atr_*`, `mmm_*`, `agt_demo`, `evl_*` | Resultados de cada sesión del lab | Las de su fuente | De The Complete Journey (84.51°) y de Lenta **solo hay agregados**: segmentos de al menos 20 hogares, curvas y métricas |

Por la licencia de Olist, el conjunto se publica bajo **CC BY-NC-SA 4.0** (uso no comercial, citando las fuentes). Las tablas `mmm_*` y las de atribución son **sintéticas**, generadas con efectos verdaderos conocidos.

```python
import duckdb
from huggingface_hub import snapshot_download

ruta = snapshot_download("neiderolaya26/marketing-agent-lab-data", repo_type="dataset")
duckdb.sql(f"SELECT canal, cpl_equilibrio FROM '{ruta}/processed/canales_rentabilidad.parquet' ORDER BY 2 DESC").show()
```
