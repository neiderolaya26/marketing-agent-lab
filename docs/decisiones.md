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

---

## 006 — CLV: horizonte, churn y modelos a comparar

- **Fecha:** 2026-11-13
- **Contexto:** Complete Journey trae un solo año (2017). La estacionalidad es plana (370–420 mil USD al mes). Sin compra en 13 semanas solo queda el 5,8% de los hogares; el 17% reduce su gasto a menos de la mitad.
- **Decisión:**
  - **Horizonte de 13 semanas:** calibro con ene–sep y valido con oct–dic, igual que la segmentación.
  - **Churn parcial:** el gasto de oct–dic cae a menos de la mitad del de jul–sep (18% de los hogares con compras en jul–sep).
  - **Compiten** BG/NBD + Gamma-Gamma (implementados desde cero con scipy), LightGBM (aprende de ene–jun → jul–sep) y la línea base "el próximo trimestre = el último".
- **Por qué (en mis palabras):** 13 semanas es el horizonte de una campaña trimestral y deja 9 meses para aprender. En un supermercado nadie cancela: el cliente se va de a poco, así que mido la caída a la mitad, que es cuando todavía se puede recuperar. Y comparo contra lo ingenuo porque, si un modelo no le gana, no vale la pena.

---

## 007 — Modelo de CLV, score de churn y prioridad de retención

- **Fecha:** 2026-11-13
- **Contexto** (`notebooks/05_clv_churn.ipynb`, `python -m growth_lab.clv`):

  | Modelo | MAE (USD, 13 semanas) | vs. ingenuo (IC 90%) | Spearman |
  |---|---|---|---|
  | Ingenuo (último trimestre) | 180,5 | — | 0,814 |
  | BG/NBD + Gamma-Gamma | 186,0 | +5,5 [+0,6; +10,3] | 0,805 |
  | LightGBM | 176,9 | −3,6 [−7,7; +0,6] | 0,823 |

  - El BG/NBD estima 1,02 compras por semana y una probabilidad de abandonar tras cada compra de 0,1%: aquí casi nadie se va del todo, que es justo lo que ese modelo sabe predecir.
  - Churn parcial (AUC): clasificador LightGBM 0,68; P(no vivo) del BG/NBD 0,62; caída esperada según los modelos de valor ≈ 0,51.
  - Por segmento, la tasa de churn es Dormidos 37%, Enfriándose 27%, Regulares 16% y Núcleo 10%, pero el gasto en riesgo es Núcleo 37 mil USD, Regulares 23 mil, Enfriándose 12,6 mil y Dormidos 1,2 mil. El 20% de hogares con más riesgo concentra el 58% de ese riesgo.
- **Decisión:** **CLV con LightGBM**, **churn con el clasificador LightGBM** y la retención se **prioriza por hogar** con el gasto en riesgo = P(churn) × la mitad del gasto de jul–sep, no por segmento.
- **Por qué (en mis palabras):** LightGBM le gana por poco al ingenuo (no es significativo y lo digo así), pero es el único que además da una probabilidad de churn que funciona. El BG/NBD está pensado para el cliente que se va del todo y aquí casi nadie lo hace. Cambio mi apuesta de la sesión 3: no priorizo un segmento sino a los hogares con más plata en riesgo, y la sesión 7 dirá a cuáles de ellos la campaña les cambia la decisión.

---

## 008 — Diseño del experimento: métrica, efecto mínimo y lectura

- **Fecha:** 2026-11-13
- **Contexto:** Hillstrom sorteó 64.000 clientes en tres grupos de ~21.300 (sin e-mail, e-mail de hombres, e-mail de mujeres) y midió visita, conversión y gasto a 2 semanas. El gasto tiene mucho ruido: en el control la media es 0,65 USD con desviación de 11,6 (casi todos gastan 0).
- **Decisión:**
  - **Métrica principal: gasto** por cliente; **visita** como métrica guía. Si discrepan, manda el gasto.
  - **Efecto mínimo a detectar: +50% del gasto** (80% de potencia, α = 5%). Necesita 19.788 por grupo y hay 21.306: con esta muestra se detecta desde +48%. Un efecto menor no se distinguiría del ruido y lo digo así.
  - **Lectura clásica y bayesiana:** IC y p-valor con corrección de Holm por comparar dos e-mails, y probabilidad de que cada e-mail sea mejor con su pérdida esperada.
- **Por qué (en mis palabras):** Fijo la métrica antes de mirar para no quedarme con la que salió mejor. Elijo el gasto porque es la plata, aunque tenga ruido, y uso la visita para ver rápido si algo se mueve. Leer de las dos formas me deja comunicarlo con probabilidades sin perder el rigor del p-valor.

---

## 009 — Resultado, margen y regla de lectura de experimentos

- **Fecha:** 2026-11-13
- **Contexto** (`notebooks/06_experimentos.ipynb`, `python -m growth_lab.experimentos`):
  - El sorteo salió bien: SRM p = 0,90 y |SMD| máximo = 0,014.
  - Gasto: e-mail de hombres **+0,77 USD por cliente** (IC 95% [0,49; 1,05], +118%) y de mujeres **+0,42** ([0,17; 0,68], +65%), ambos significativos con Holm; el bootstrap da el mismo IC. Hombres le gana a mujeres por +0,35 (p = 0,03).
  - CUPED con `history` reduce la varianza del gasto 0,04%, y el ajuste con todas las variables previas, 0,1% (2,8% en la visita): las variables previas casi no predicen el resultado.
  - *Peeking*: mirar 20 veces y parar al primer p < 0,05 da **23% de falsos positivos** (simulación A/A); con límites de O'Brien-Fleming calibrados, 4,7%. Ambos e-mails cruzan el límite con la mitad de la muestra.
  - Bayesiano: P(mejor) > 99,9% para ambos en gasto; mejora esperada +121% (hombres) y +67% (mujeres).
- **Decisión:**
  - **Recomendación:** mandar el **e-mail de hombres a todos** mientras la sesión 7 (uplift) dice a quién conviene personalizar.
  - **Margen supuesto de 30%** (`experimentos.MARGEN`, sensibilidad 20–40%): el costo máximo por correo es 0,23 USD (hombres) y 0,13 (mujeres); con el límite inferior del IC, 0,15 y 0,05.
  - **Regla de lectura:** los experimentos del lab se leen de forma **secuencial con límites de O'Brien-Fleming** calibrados por simulación.
- **Por qué (en mis palabras):** Los dos correos funcionan y el de hombres es el mejor en promedio, así que es la opción segura hoy. Un correo cuesta centavos y deja hasta 0,23 USD de margen, así que es rentable con cualquier margen razonable. Y aprendí que mirar el resultado todos los días sin corregir convierte un 5% de error en un 23%.
