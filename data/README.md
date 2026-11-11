# Datos

| Carpeta | Contenido | ¿Se sube a GitHub? |
|---|---|---|
| `raw/olist/` | CSV de Olist E-Commerce + Marketing Funnel (Kaggle) | No (se descarga con los comandos del README) |
| `raw/completejourney/` | Archivos `.rda`/`.rds` de The Complete Journey | No |
| `raw/hillstrom/` | CSV de Hillstrom MineThatData | No |
| `raw/lenta/` | CSV comprimido de Lenta (scikit-uplift) | No |
| `raw/atribucion/` | Parquet del synthetic attribution benchmark | No |
| `interim/` | Parquet + `growth.duckdb` | No |
| `processed/` | Derivados y agregados para reproducir los resultados | Sí, < 50 MB y según la licencia de cada fuente |

## Fuentes y licencias

- **Olist** (`olistbr/brazilian-ecommerce`, `olistbr/marketing-funnel-olist`): CC BY-NC-SA 4.0. Uso no comercial, con atribución y la misma licencia para los derivados. Los ids ya vienen anonimizados (hash).
- **The Complete Journey** (84.51° / dunnhumby, vía el paquete de R `completejourney` 1.1.1): el paquete declara CC0, pero los términos de dunnhumby limitan el uso a investigación o fines no comerciales y no permiten redistribuir sin permiso. **Regla del repo:** se descarga con script y solo se publican agregados y resultados de modelos, nunca transacciones.
- **Hillstrom** (Kevin Hillstrom, MineThatData, 2008): publicado para el reto de analítica de e-mail; uso libre en investigación y docencia.
- **Lenta** (competencia de uplift, distribuido por scikit-uplift): uso en investigación; no se redistribuye, solo métricas y agregados.
- **Synthetic attribution benchmark** (Hugging Face, `lucianfialho/synthetic-attribution-benchmark`, v2): CC BY 4.0. Datos sintéticos con semilla 42; se puede publicar con atribución.

## Qué no se publica

- Texto libre de reseñas de Olist (`review_comment_title`, `review_comment_message`): puede traer nombres o datos de contacto. Para NLP se publican solo agregados o features derivadas.
- `geolocation` de Olist a nivel de coordenada y prefijo postal: se publica agregada por estado.
- Filas de Complete Journey y Lenta: por licencia, solo agregados.
