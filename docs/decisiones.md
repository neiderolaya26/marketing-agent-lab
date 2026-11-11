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
- **Por qué (en mis palabras):** "hagamos el ciclo completo, end to end, la gracia es tener muy enriquecido el lab en git" _(Neider puede ampliarlo)_.
