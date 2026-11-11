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

---

## 002 — Valor de un lead: ventana y definición

- **Fecha:** 2026-11-10
- **Contexto** (`notebooks/02_embudo_cohortes.ipynb`, `python -m growth_lab.embudo`):
  - Valor de un lead = P(cierre) × GMV medio de los contratos **maduros** en sus primeros N días. Un contrato está maduro si `won_date + N` cae antes del último pedido (29-ago-2018); si no, no entra al promedio para no castigar a los canales que cerraron tarde.
  - Con **90 días**: búsqueda pagada 63 BRL por lead (IC 90% [40; 94]), referidos 54 [12; 110], orgánica 52 [29; 83], tráfico directo 21 [11; 31], e-mail 14 [0; 39] y social 13 [6; 21]. Pagada y orgánica no se distinguen entre sí, pero ambas superan con claridad a social y directo.
  - **El orden cambia con la ventana:** a 180 días la orgánica (132) supera a la pagada (82), pero con solo 67 y 39 contratos maduros. A 60 días están empatadas (37 y 39).
  - El canal `unknown` vale 137 BRL por lead (IC [41; 293]). Probablemente son leads que llegaron por el equipo comercial sin canal registrado; no se puede invertir en él.
  - GMV = `price` de los ítems sin flete, en BRL, sin pedidos cancelados ni indisponibles. Olist no publica su comisión, así que el valor es GMV, no margen.
- **Opciones:** ventana de 60, 90, 120 o 180 días; valor como GMV o como comisión supuesta (p. ej. 10–20% del GMV, dejándola como parámetro).
- **Decisión:** _(por escribir; la implementación usa 90 días y GMV como valor por defecto)_
- **Por qué (en mis palabras):** _(por escribir)_
