# marketing-agent-lab

> 🚧 v0.1 en construcción · Proyecto construido sesión a sesión, con cada decisión registrada en [docs/decisiones.md](docs/decisiones.md). · Repo 2 del portafolio (el 1 es `credit-risk-agent-lab`).

- **Fecha de inicio:** 2026-11-10
- **Estado:** En curso (modo guiado: Neider decide y escribe las decisiones; Claude arma el esqueleto y explica)

## Pregunta de negocio (decisión 001)

¿Se puede, **solo con datos públicos**, construir un sistema que diga **dónde poner el presupuesto de marketing**, es decir qué canal, qué cliente y qué campaña generan **valor incremental** y no solo conversiones, y que un **agente de IA "analista de growth"** responda preguntas de negocio y proponga campañas con su impacto estimado?

## Resultados hasta ahora

| Pieza | Resultado | Decisión |
|---|---|---|
| Valor de un lead | A 90 días, un lead de búsqueda pagada vale 63 BRL de GMV (IC 90% [40; 91]) y uno de orgánica 52 [28; 84]: no se distinguen, pero ambos valen de 4 a 5 veces más que uno de social (13 [6; 21]). Con una comisión supuesta del 15%, eso es 9,5 y 7,8 BRL de ingreso por lead. Elegí 90 días: a 180 el orden cambia, pero con solo 194 contratos | 002 |
| Recompra | Del 0,8% al primer mes al 3,1% a los 12 meses, estable entre cohortes: el CLV se trabaja con Complete Journey | 001 |
| Segmentación | K-means sobre RFM con 4 segmentos explica el 47% del gasto futuro (RFM clásico: 36%). El Núcleo es el 32% de los hogares y hace el 64% del gasto; Enfriándose (24%) sigue activo pero compra menos de la mitad: es la apuesta de retención. Las campañas llegaron al 97% del Núcleo y al 36% de Enfriándose: no fueron al azar | 003 |
| Lead scoring | Entre 6 modelos y 240 intentos de Optuna, la regresión logística (AUC 0,681 en abr–may) empata con LightGBM y gana a los demás; el 20% de leads con mejor score trae el 39% de los cierres. Por unidad de costo, social rinde 0,17 y display 0,03 de lo que rinde la búsqueda pagada | 004–005 |
| Valor del cliente | "El próximo trimestre = el último" es difícil de batir: LightGBM lo mejora por poco (MAE 177 vs. 181 USD, no significativo) y el BG/NBD lo empeora (186), porque aquí nadie se va del todo: compra menos. El churn parcial (gasto a la mitad) se anticipa con AUC 0,68. El 20% de hogares con más riesgo junta el 58% de la plata en riesgo, sobre todo del Núcleo | 006–007 |
| Experimentos | El e-mail de hombres sube el gasto +0,77 USD por cliente (+118%, IC [0,49; 1,05]) y el de mujeres +0,42 (+65%); con 30% de margen, cada correo puede costar hasta 0,23 USD. CUPED casi no ayuda aquí (< 0,1%). Mirar 20 veces sin corregir sube los falsos positivos de 5% a 23%; con O'Brien-Fleming calibrado, 4,7% | 008–009 |
| Uplift | En el gasto no hay diferencias medibles entre clientes: personalizar con uplift dio 1,23 USD por cliente contra 1,42 de mandar el correo de hombres a todos. Sí hay persuadibles en la visita del correo de mujeres y en Lenta, donde el X-learner consigue con el 30% del presupuesto 1,6 veces lo que el azar | 010 |
| Efecto de campañas | La comparación ingenua dice +57% de gasto, pero es selección: corregida con DiD doblemente robusto, las campañas tipo A dan +2,4 USD por hogar y semana [−1,1; +6,0], no distinguible de 0 (los placebos pasan). Medir el efecto que pagaría los cupones exige 4.300–6.000 hogares de control sorteados por campaña | 011–012 |
| Operación | Los meses con más entregas tarde (nov-2017, feb–mar-2018: 14–21%) son los de peor reseña (correlación mensual −0,81) | — |

![Valor de un lead por canal](docs/img/embudo_valor_por_lead.png)

![Segmentos: hogares y gasto futuro](docs/img/segmentos_valor.png)

![Búsqueda de modelos de lead scoring](docs/img/leads_modelos.png)

![Plata en riesgo por segmento](docs/img/clv_riesgo_segmento.png)

![Lectura secuencial del experimento](docs/img/exp_secuencial.png)

![Curvas Qini de uplift](docs/img/uplift_qini.png)

![Efecto de campañas: real vs. placebo](docs/img/cmp_efectos.png)

## Para aprender

- [Guía del lab](docs/guia.md): qué hacemos en cada sesión, qué hay que saber y qué salió, en palabras simples.
- [Glosario](docs/glosario.md): los términos de marketing con su equivalente en riesgo de crédito.

## Base conceptual

Los temas salen de la ruta de aprendizaje **Paid Media** (`learning/rutas/paid-media`): el plan de 10 semanas (Columbia, Wharton) y el catálogo de modelos. Cada sesión toma un bloque de la ruta y lo aterriza en el dataset público que mejor lo resuelve.

## Datos

Ningún dataset público trae a la vez embudo, recompra, experimentos y gasto por canal. Por eso cada parte del lab usa la fuente con señal real para esa pregunta (revisión del 2026-11-10).

| Dataset | Para qué | Tamaño | Licencia |
|---|---|---|---|
| [Olist E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) + [Marketing Funnel](https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist) | **Captación**: embudo de leads con canal de origen, de lead a vendedor que vende | 99.441 pedidos · 8.000 leads · 842 cerrados | CC BY-NC-SA 4.0 |
| [The Complete Journey](https://cran.r-project.org/package=completejourney) (84.51° / dunnhumby) | **Retención y campañas**: hogares que recompran, campañas dirigidas, cupones y canjes, promociones | 2.469 hogares · 1,47 M transacciones · 27 campañas | Paquete CC0; datos de dunnhumby solo para uso no comercial, sin redistribuir |
| [Hillstrom MineThatData](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) | **Experimentos y uplift**: e-mail aleatorizado en 3 brazos | 64.000 clientes | Uso libre en investigación |
| [Lenta](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html) (vía scikit-uplift) | **Uplift a escala**: SMS aleatorizado con 193 variables, chequeo de robustez | 687.029 clientes | Datos de competencia, uso en investigación |
| [Synthetic attribution benchmark](https://huggingface.co/datasets/lucianfialho/synthetic-attribution-benchmark) | **Atribución**: recorridos multitoque con la atribución Shapley verdadera y costo por toque | 500.000 recorridos | CC BY 4.0 |
| MMM sintético (sesión 10) | **Presupuesto**: gasto endógeno por canal con efecto verdadero conocido, al estilo de Heusch (2026) | por generar | se genera en el repo |

Primeros hechos (`python -m growth_lab.ingest`):

- **Olist, compradores:** solo el 3,0% de 93.358 compradores recompra, así que el CLV del comprador se trabaja con Complete Journey.
- **Olist, embudo:** la conversión de lead a contrato va del 3% (e-mail) al 12,3% (búsqueda pagada). De 842 vendedores cerrados, 380 venden en la ventana.
- **Complete Journey:** el 98,7% de los hogares vuelve (mediana de 43 canastas al año). 1.559 de los 2.469 hogares recibieron al menos una campaña (tipo A, B o C): hay tratados y no tratados, pero sin aleatorizar.
- **Hillstrom:** el e-mail de hombres lleva la conversión de 0,57% a 1,25%. **Lenta:** el SMS lleva la respuesta de 10,3% a 11,0%.
- **Atribución:** 6,0% de los recorridos convierte, con 2,7 toques en promedio. Los canales se llaman igual que en el embudo de Olist (paid_search, social, email…), así que su costo por toque sirve para suponer el CAC por canal.

## Hoja de ruta (tentativa)

| Sesión | Tema (bloque de la ruta) | Técnicas del catálogo | Datos |
|---|---|---|---|
| 1 ✅ | Entorno, datos y pregunta de negocio (decisión 001: ciclo completo) | Ingesta a Parquet, DuckDB en memoria, panorama de fuentes (notebook 01) | Todas |
| 2 ✅ | Inteligencia de mercado: embudo y cohortes | Valor de un lead = P(cierre) × GMV a 90 días con IC bootstrap, cohortes de recompra, KPIs mensuales (notebook 02) | Olist |
| 3 ✅ | Segmentación del cliente | RFM clásico vs. K-means vs. GMM validados fuera de tiempo (η² del gasto futuro), estabilidad, exposición a campañas (notebook 03) | Complete Journey |
| 4 ✅ | Lead scoring y calidad del lead | 6 modelos con Optuna y validación temporal, lift y curva de ganancia, importancia por permutación, PSI, CPU vs. GPU, valor por unidad de costo por canal (notebook 04) | Olist + benchmark |
| 5 ✅ | Valor del cliente | BG/NBD + Gamma-Gamma desde cero vs. LightGBM vs. ingenuo, churn parcial, gasto en riesgo por hogar (notebook 05) | Complete Journey |
| 6 ✅ | Experimentos | SRM y balance, potencia y MDE, efectos con Holm y bootstrap, CUPED y ajuste por regresión, *peeking* A/A y O'Brien-Fleming calibrado, lectura bayesiana, economía (notebook 06) | Hillstrom |
| 7 ✅ | Uplift | S/T/X-learner, resultado transformado y propensión, cross-fitting, Qini con IC, políticas con IPW, presupuesto parcial (notebook 07) | Hillstrom + Lenta |
| 8 ✅ | Efecto de campañas y cupones | Ingenuo vs. DiD simple vs. DiD doblemente robusto vs. DML, placebo, estudio de eventos, efecto de equilibrio y tamaño del grupo de control (notebook 08) | Complete Journey |
| 9 | Atribución multitoque | Último clic, lineal, Markov y Shapley contra la verdad conocida | Benchmark de atribución |
| 10 | Marketing Mix Modeling | MMM bayesiano (adstock, saturación), optimización de presupuesto | MMM sintético |
| 11 | Agente "analista de growth" | LangGraph: text-to-SQL sobre DuckDB + modelos como herramientas + aprobación humana del presupuesto | Todas |
| 12 | Operación | Servidor MCP, registro de ejecuciones, evals (casos dorados, juez) | Todas |
| 13 | Caso de consultoría | Recomendación de presupuesto, demo, README final, release v1.0.0 | Todas |

```mermaid
flowchart LR
    O["Olist<br/>embudo por canal"] --> A["Captación<br/>embudo · lead scoring · CAC"]
    CJ["Complete Journey<br/>hogares · campañas · cupones"] --> R["Retención<br/>RFM · CLV · efecto de campañas"]
    H["Hillstrom + Lenta<br/>experimentos"] --> E["A/B · uplift"]
    AT["Benchmark de atribución"] --> M["Atribución<br/>Markov · Shapley"]
    MM["MMM sintético"] --> P["Presupuesto<br/>MMM · optimización"]
    A & R & E & M & P --> DB[("DuckDB")]
    DB --> AG["Agente analista de growth<br/>SQL + modelos como herramientas"]
    AG --> HU{"Humano aprueba<br/>presupuesto"}
```

## Cómo reproducirlo

Requiere Python 3.12 o superior y un token de Kaggle. Las descargas pesan unos 330 MB.

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest                           # corre sin descargar nada: usa data/processed/
```

Para regenerar todo desde las fuentes (unos 330 MB de descargas):

```bash
# Olist (Kaggle)
kaggle datasets download olistbr/brazilian-ecommerce -p data/raw/olist --unzip
kaggle datasets download olistbr/marketing-funnel-olist -p data/raw/olist --unzip

# The Complete Journey (archivos R del paquete completejourney)
for f in campaign_descriptions.rda campaigns.rda coupon_redemptions.rda coupons.rda demographics.rda products.rda promotions.rds transactions.rds; do
  curl -L -o data/raw/completejourney/$f https://github.com/bradleyboehmke/completejourney/raw/master/data/$f
done

# Hillstrom y Lenta
curl -L -o data/raw/hillstrom/hillstrom.csv http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
curl -L -o data/raw/lenta/lenta_dataset.csv.gz https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz

# Benchmark de atribución (Hugging Face)
for f in journeys.parquet touchpoints.parquet ground_truth.json README.md; do
  curl -L -o data/raw/atribucion/$f https://huggingface.co/datasets/lucianfialho/synthetic-attribution-benchmark/resolve/main/$f
done

python -m growth_lab.ingest      # 23 Parquet (zstd) en data/interim/, 233 MB
python -m growth_lab.publicar    # 14 Parquet publicables en data/processed/, 33 MB
```

**Todo vive en Parquet**, sin una base de datos aparte: `growth_lab.db.conectar()` abre DuckDB en memoria con una vista por archivo. Las tablas de Olist van sin prefijo, las de Complete Journey con `cj_` y las de atribución con `attr_`.

```python
from growth_lab import db
con = db.conectar(db.PROCESSED)          # lo publicado; db.conectar() lee data/interim
db.consultar(con, "SELECT origin, count(*) AS leads FROM marketing_qualified_leads GROUP BY 1")
```

`db.consultar()` es la puerta del agente: solo acepta una sentencia `SELECT` y la conexión solo puede leer su carpeta.

### Datos publicados en el repo (`data/processed/`)

| Tablas | Fuente y licencia | Qué se quitó |
|---|---|---|
| `orders`, `order_items`, `order_payments`, `customers`, `sellers`, `products`, `marketing_qualified_leads`, `closed_deals`, `order_reviews`, `geolocation` | Olist, CC BY-NC-SA 4.0 | Texto libre de reseñas; geolocalización agregada por estado |
| `hillstrom` | MineThatData | — |
| `attr_journeys`, `attr_touchpoints` | Benchmark de atribución, CC BY 4.0 | — |

Derivados por sesión: `embudo_canal`, `cohortes_compradores` y `kpis_mensuales` (sesión 2, `python -m growth_lab.embudo`); `segmentos_perfil` y `segmentos_comparacion` (sesión 3, `python -m growth_lab.segmentacion`, solo agregados de segmentos con ≥ 20 hogares); `leads_optuna`, `leads_metricas`, `leads_deciles`, `leads_importancia`, `leads_score`, `leads_estabilidad`, `leads_coeficientes`, `leads_cpu_vs_gpu` y `canales_rentabilidad` (sesión 4, `python -m growth_lab.leads_modelos 40` y `python -m growth_lab.leads`); `clv_metricas`, `clv_deciles`, `clv_churn`, `clv_churn_importancia`, `clv_parametros` y `clv_por_segmento` (sesión 5, `python -m growth_lab.clv`, solo agregados); `exp_balance`, `exp_potencia`, `exp_efectos`, `exp_cuped`, `exp_peeking`, `exp_secuencial`, `exp_bayes` y `exp_economia` (sesión 6, `python -m growth_lab.experimentos`); `upl_qini`, `upl_metricas` y `upl_politicas` (sesión 7, `python -m growth_lab.uplift`; de Lenta solo curvas y métricas); `cmp_efectos`, `cmp_evento` y `cmp_economia` (sesión 8, `python -m growth_lab.campanas`, solo agregados).

Complete Journey y Lenta no se publican por licencia: solo sus agregados y resultados.

## Estructura

```
data/raw/          crudos descargados (no se suben)
data/interim/      Parquet de todas las fuentes (no se suben)
data/processed/    Parquet publicados: fuentes con licencia abierta + derivados de cada sesión
docs/              decisiones.md, imágenes
notebooks/         un notebook por sesión, con mis conclusiones
src/growth_lab/    código del pipeline, modelos y agente
tests/             pruebas que no necesitan datos
```

## Bitácora

- 2026-11-10: inicio. Datos de Olist y Hillstrom ingeridos, hoja de ruta tomada de la ruta Paid Media, decisión 001 planteada.
- 2026-11-13: sesión 8: efecto de campañas sin experimento, decisiones 011–012.
- 2026-11-13: sesión 7: uplift con Hillstrom y Lenta, decisión 010.
- 2026-11-13: sesión 6: experimentos con Hillstrom, decisiones 008–009.
- 2026-11-13: sesión 5: CLV y churn parcial, decisiones 006–007.
- 2026-11-13: sesión 4: lead scoring con búsqueda de 6 modelos y decisiones 004–005; guía y glosario.
- 2026-11-13: repo público `marketing-agent-lab`; decisión 002 (ventana de 90 días, comisión 15%). Sesión 3: segmentación y decisión 003.
- 2026-11-10: decisión 001 (ciclo completo), todo en Parquet y fuentes abiertas publicadas en el repo. Sesión 2: embudo y cohortes.
- 2026-11-10: revisión de datasets. Se suman Complete Journey (retención y campañas), Lenta (uplift a escala) y el benchmark de atribución. El MMM usará un generador propio porque el benchmark de Heusch (2026) no publica sus datos.
