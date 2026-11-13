# Registro de decisiones

Cada decisión importante del proyecto queda aquí, **escrita por Neider**: qué se decidió, qué alternativas había y por qué.

---

## 001 — Pregunta de negocio y unidad de análisis

- **Fecha:** 2026-11-10
- **Contexto** (hechos de `python -m growth_lab.ingest`):
  - **Compradores (B2C):** 99.441 pedidos de sep-2016 a ago-2018 (sep y oct-2018 casi vacíos). Hay 93.358 compradores únicos (`customer_unique_id`) con pedidos entregados, y **solo el 3,0% recompra**. El 41% de las reseñas trae texto. No hay datos de gasto en marketing ni de canal de adquisición del comprador.
  - **Vendedores (B2B):** 8.000 leads (MQL) entre jun-2017 y may-2018, con canal de origen. 842 se cerraron, y de ellos **380 tienen ventas** en la ventana. La conversión va del 3,0% (e-mail) y 5,6% (social) al 11,8% (orgánico) y 12,3% (búsqueda pagada).
  - **Hogares (Complete Journey):** 2.469 hogares durante 2017 con 1,47 M transacciones; el **98,7% recompra** (mediana de 43 canastas). 1.559 hogares recibieron al menos una de 27 campañas (4 tipo A, 17 tipo B, 6 tipo C), sin aleatorizar. Hay 116 mil cupones entregados y 2.102 canjes.
  - **Experimentos:** Hillstrom (64 mil clientes, 3 brazos de e-mail) y Lenta (687 mil clientes, SMS, 75% tratados, 193 variables).
  - **Atribución:** benchmark sintético de 500 mil recorridos con la atribución Shapley verdadera y el costo por toque de cada canal (los mismos nombres de canal del embudo de Olist).
- **Opciones:**
  - **A) Retención:** ¿a qué hogares y con qué campaña o cupón conviene impactar para aumentar su valor? Usa Complete Journey (RFM, CLV y efecto de campañas) con Hillstrom y Lenta para el uplift.
  - **B) Captación:** ¿qué canal trae leads que cierran **y además venden**? Lead scoring con propensión × valor en el embudo de Olist, con el CAC por canal supuesto a partir del costo del benchmark de atribución. Riesgo: muestra pequeña (842 cerrados).
  - **C) Ciclo completo:** una sola pregunta de presupuesto, ¿cuánto invertir en captar (B) y cuánto en retener (A)?, que el MMM y el agente cierran al final. Es la que más usa los datos, pero exige una métrica común, por ejemplo valor incremental por dólar invertido.
- **Decisión:** **C) Ciclo completo, de punta a punta.** La pregunta del lab es cuánto invertir en **captar** (canales del embudo de Olist) y cuánto en **retener** (campañas y cupones de Complete Journey), con experimentos, atribución y MMM para medir lo incremental, y un agente que responda y proponga el reparto con aprobación humana.
- **Métrica de éxito:** valor incremental por dólar invertido (ROI incremental) en cada palanca: canal de captación, campaña de retención y canal de medios. Cada sesión reporta su pieza en esa unidad o deja claro por qué no se puede.
- **Datos en el repo:** todo en Parquet (zstd). Se publican en `data/processed/` las fuentes con licencia que lo permite (Olist sin texto libre ni geolocalización fina, Hillstrom, benchmark de atribución) y los derivados de cada sesión. De Complete Journey y Lenta solo se publican agregados.
- **Por qué (en mis palabras):** Quiero el ciclo completo, de punta a punta, porque en la vida real el presupuesto se reparte entre captar y retener, y cada fuente sirve para una parte distinta. La gracia es que el lab quede muy completo en GitHub: datos en Parquet que cualquiera pueda usar, modelos, experimentos y un agente que conecte todo.

---

## 002 — Valor de un lead: ventana y definición

- **Fecha:** 2026-11-13
- **Contexto** (`notebooks/02_embudo_cohortes.ipynb`, `python -m growth_lab.embudo`):
  - Valor de un lead = P(cierre) × GMV medio de los contratos **maduros** en sus primeros N días. Un contrato está maduro si `won_date + N` cae antes del último pedido (29-ago-2018); si no, no entra al promedio para no castigar a los canales que cerraron tarde.
  - Contratos maduros según la ventana: 722 (60 días), 663 (90), 543 (120) y 194 (180), de 842 cerrados.
  - A 90 días: búsqueda pagada 63 BRL por lead (IC 90% [40; 91]), referidos 54 [11; 111], orgánica 52 [28; 84], tráfico directo 21 [11; 30], e-mail 14 [0; 40] y social 13 [6; 21].
  - El orden cambia con la ventana: a 60 días pagada y orgánica empatan (39 y 37); a 180 días la orgánica (132) supera a la pagada (82), pero con muy pocos contratos.
  - GMV = `price` de los ítems sin flete, en BRL, sin pedidos cancelados ni indisponibles. Olist no publica su comisión.
- **Opciones:** ventana de 60, 90, 120 o 180 días; valor como solo GMV, comisión fija o GMV con comisión como parámetro.
- **Decisión:**
  - **Ventana de 90 días.** 60, 120 y 180 quedan como sensibilidad en el notebook 02.
  - **El modelo trabaja en GMV** y el ingreso de Olist se calcula con una **comisión supuesta del 15%** (`embudo.COMISION`), con sensibilidad de 10% a 20%. Con eso, un lead de búsqueda pagada deja unos 9,5 BRL de ingreso y uno de orgánica 7,8.
- **Por qué (en mis palabras):** Con 90 días veo ventas reales del vendedor sin perder demasiados contratos: me quedo con 663 de 842. A 180 días el orden de los canales cambia, pero con 194 contratos es ruido, no señal. Uso GMV porque es lo que se observa, y la comisión la dejo como parámetro visible porque cuando compare contra el costo de captación necesito ingreso, no ventas.

---

## 003 — Segmentación de hogares

- **Fecha:** 2026-11-13
- **Contexto** (`notebooks/03_segmentacion.ipynb`, `python -m growth_lab.segmentacion`): 2.446 hogares de Complete Journey. Segmento con las compras de ene–sep 2017 y valido con oct–dic: η² = parte de la varianza del gasto futuro (log) que explica el segmento.
  - RFM clásico (8 segmentos): η² del gasto futuro 0,364; de la actividad 0,085.
  - K-means sobre RFM: 0,426 (k=3), **0,465 (k=4)** y 0,472 (k=8). De k=4 en adelante casi no mejora.
  - Agregar comportamiento (ticket, descuentos, cupones, % frescos, % droguería, regularidad) empeora: 0,35–0,42 con K-means y 0,23–0,35 con GMM.
  - K-means con k=4 es estable: índice de Rand ajustado de 0,95 entre 10 semillas.
- **Opciones:** RFM clásico, K-means RFM con k=3 o k=4, clustering con comportamiento.
- **Decisión:** **K-means sobre RFM (log y estandarizado) con k = 4**, nombrados por monto: Núcleo (778 hogares), Regulares (820), Enfriándose (590) y Dormidos (258). Las variables de comportamiento **solo describen** cada segmento, no entran al clustering.
- **Por qué (en mis palabras):** Elegí la segmentación que mejor predice el gasto de los meses siguientes con pocos grupos: 4 segmentos explican el 47% contra el 36% del RFM clásico, y salen iguales con cualquier semilla. Agregar comportamiento suena más completo, pero predice peor; prefiero que el segmento diga a quién y que el comportamiento diga qué ofrecerle.

---

## 004 — Lead scoring: población, objetivo y costos

- **Fecha:** 2026-11-13
- **Contexto:** los contratos del dataset empiezan en dic-2017, así que en 2017 solo cierra el 3% de los leads y desde ene-2018 el 13%. Es un problema de cobertura, no de calidad. Olist no publica cuánto le cuesta cada canal. En el benchmark de atribución, orgánica y directo cuestan 0.
- **Decisión:**
  - **Población:** leads de 2018 (5.998). Entreno con ene–mar (3.343) y pruebo con abr–may (2.655). Los leads de 2017 solo sirven como historia.
  - **Objetivo:** P(cierre). Calidad del lead = P(cierre) × GMV por contrato de su canal (sesión 2).
  - **Variables:** solo lo que se sabe cuando llega el lead: canal, landing (las 40 más frecuentes del entrenamiento; el resto queda como "otra"), día de la semana y tasas previas calculadas solo con cierres ya ocurridos. Nada de `closed_deals`, porque solo existe si el lead cerró.
  - **Costos:** CPL de equilibrio (= ingreso por lead con 15% de comisión) e índice de costo por conversión del benchmark (búsqueda pagada = 1). Orgánica y directo son costo fijo: se reporta su valor, sin ROI de pauta.
- **Por qué (en mis palabras):** Si entreno con 2017, el modelo aprende que "2017 no cierra", que es un error de los datos y no de los leads. Prefiero medir rentabilidad con un costo relativo explícito que inventar pesos, y no le calculo ROI de pauta al SEO porque su costo no depende de cuántos leads trae.

---

## 005 — Modelo campeón del lead scoring

- **Fecha:** 2026-11-13
- **Contexto** (`notebooks/04_lead_scoring.ipynb`, `python -m growth_lab.leads_modelos 40`): 6 modelos, 40 intentos de Optuna cada uno, ajustados con validación temporal dentro de ene–mar y medidos una sola vez en abr–may.

  | Modelo | AUC validación | AUC prueba (IC 90%) | vs. logística |
  |---|---|---|---|
  | Logística | 0,611 | 0,681 [0,655; 0,706] | — |
  | LightGBM | 0,609 | 0,680 [0,654; 0,706] | −0,001 [−0,010; +0,009] |
  | XGBoost | 0,609 | 0,672 [0,646; 0,697] | −0,009 [−0,017; −0,000] |
  | Random Forest | 0,611 | 0,665 [0,637; 0,692] | −0,016 [−0,028; −0,004] |
  | CatBoost | 0,616 | 0,634 [0,605; 0,662] | −0,046 [−0,061; −0,032] |
  | Red neuronal | 0,534 | 0,361 [0,333; 0,387] | −0,319 |

  - Las tasas previas tienen PSI > 2 entre entreno y prueba: se duplican porque los cierres conocidos se acumulan desde dic-2017. Sin ellas la logística da 0,677 (dentro del ruido).
  - Con ~3.300 filas la GPU es más lenta: XGBoost 2,4 s frente a 0,9 s en CPU; CatBoost 47 s frente a 38 s.
- **Decisión:** **regresión logística** como campeona (`leads_modelos.CAMPEON`). Las variables de historia **se mantienen** y se monitorea su PSI (`leads_estabilidad`).
- **Por qué (en mis palabras):** La logística y LightGBM empatan, y con un empate gana lo simple: se lee como un scorecard y entrena en una décima de segundo. CatBoost me enseñó que el mejor en validación no es el mejor en prueba. Mantengo las tasas previas porque ayudan con las landings nuevas y su deriva es un artefacto de este dataset, pero las dejo vigiladas con PSI.
