# Glosario

Términos del lab en palabras simples. Crece en cada sesión. La columna "En crédito" da la analogía cuando existe.

| Término | Qué es | En crédito |
|---|---|---|
| **A/B test** | Experimento: se reparte gente al azar entre dos versiones (correo, landing, oferta) y se compara el resultado. Como es al azar, la diferencia es causal | Piloto de política con grupo de control |
| **Adstock** | El efecto de la publicidad no desaparece el día que se pauta: se arrastra y se va apagando. El MMM lo modela | — |
| **Agente** | Un modelo de lenguaje que decide qué herramienta usar, lee el resultado y sigue hasta poder responder. En el lab, los números siempre salen de las herramientas | Analista junior con acceso a los sistemas |
| **AOV** (*average order value*) | Ticket medio: GMV / número de pedidos | Monto medio desembolsado |
| **ARI** (índice de Rand ajustado) | Mide cuánto coinciden dos agrupaciones de las mismas personas: 1 = idénticas, 0 = como al azar. Lo usamos para ver si K-means da lo mismo con otra semilla | — |
| **Atribución** | Repartir el mérito de una venta entre los canales que tocó el cliente antes de comprar (vio un anuncio, buscó en Google, abrió un correo) | — |
| **Bootstrap** | Remuestrear los datos con reemplazo muchas veces para ver cuánto varía una cifra; de ahí sale el intervalo de confianza | El mismo que usamos para la diferencia de AUC |
| **CAC** (costo de adquisición de cliente) | Cuánto cuesta conseguir un cliente: gasto en marketing / clientes nuevos | Costo por desembolso |
| **Calibración del juez** | Comparar lo que califica el juez con lo que califica una persona en las mismas respuestas. Se mide con el acuerdo y con kappa | Validar un modelo contra la revisión de un analista |
| **Canal** | Por dónde llegó el lead: búsqueda pagada (Google Ads), orgánica (SEO), social, e-mail, display (banners), referidos, directo | Canal de originación |
| **Caso de consultoría** | El entregable para un gerente: problema, hallazgos, recomendación con su valor y confianza, plan y límites | Memo de comité de la política final |
| **Caso dorado** | Una pregunta fija con su respuesta correcta conocida (las cifras que debe dar, a quién debe consultar). Se corren en cada cambio del agente | Backtest de la política con casos conocidos |
| **Cohorte** | Grupo que empezó en el mismo período (mes de primera compra) y se sigue en el tiempo | Cosecha |
| **Comisión** | Lo que Olist se queda de cada venta de un vendedor. No es pública; suponemos 15% | Tasa / margen financiero |
| **Contrato maduro** | Un contrato con al menos 90 días observables después de firmar; solo esos entran al valor por lead | Préstamo con ventana de desempeño completa |
| **Control de demanda** | Una serie que se mueve con la demanda (búsquedas de la categoría en Google Trends, por ejemplo). En el MMM evita que la venta que iba a llegar igual se le atribuya a la pauta | Controlar por el ciclo económico al medir una política |
| **CPL** (costo por lead) | Cuánto se paga por cada lead | Costo por solicitud |
| **CPL de equilibrio** | El máximo que puedes pagar por un lead sin perder plata = ingreso esperado por lead | Tasa de equilibrio (ROI = 0) |
| **Deuda conocida** | Lo que queda pendiente y se deja escrito con su porqué, para no dar por bueno lo que no se midió | Limitaciones del modelo en la documentación de validación |
| **Display** | Anuncios gráficos (banners) en sitios web | — |
| **Embudo** (*funnel*) | Las etapas que recorre un prospecto: lead → contrato → vendedor que vende | Solicitud → aprobación → desembolso → buen pago |
| **η² (eta cuadrado)** | Qué parte de la variación de una cifra (el gasto futuro) explican los grupos (los segmentos). 0,47 = los segmentos explican el 47% | Como el R² de una regresión con una variable categórica |
| **Enrutador** | El primer paso del agente: decide qué especialistas necesita la pregunta y qué le pregunta a cada uno | Mesa de entrada que reparte las solicitudes |
| **Ficha del modelo (model card)** | Documento que acompaña a un modelo publicado: con qué datos se entrenó, cómo se midió, para qué sirve y sus límites | Documentación de validación del modelo |
| **Fidelidad de cifras** | Qué parte de los números de una respuesta sale de lo que devolvieron las herramientas. Los cálculos derivados (sensibilidades) bajan la fidelidad sin ser inventos | — |
| **Gasto endógeno** | Se gasta más cuando ya se espera vender más (temporada, promociones). El MMM lo confunde con efecto de la pauta y la infla | Dar más cupo a los buenos clientes y concluir que el cupo los hace buenos |
| **GMM** (mezcla de gaussianas) | Clustering "blando": supone que cada grupo es una nube con forma de campana y da la probabilidad de pertenecer a cada uno | — |
| **GMV** (*gross merchandise value*) | Valor total vendido en la plataforma, antes de comisiones | Monto colocado |
| **Herramienta (tool)** | Una función de código que el agente puede llamar: una consulta SQL, el valor por canal, el simulador de reparto | — |
| **Hugging Face Hub** | Repositorio público de modelos y datos, con su ficha; desde ahí la demo descarga los modelos | — |
| **Importancia por permutación** | Cuánto empeora el modelo (cae el AUC) si se desordena una variable al azar. Funciona para cualquier modelo | Alternativa a SHAP o al IV |
| **Interrupt (aprobación humana)** | Pausa del grafo de LangGraph hasta que una persona aprueba o rechaza; el estado queda guardado y se retoma después, incluso desde otro proceso | Firma del analista en el comité de crédito |
| **Inyección de instrucciones** | Texto escondido en una pregunta o en un dato que intenta darle órdenes al agente ("ignora tus reglas"). Se prueba en los casos de seguridad | Fraude por ingeniería social |
| **Juez (LLM-as-a-judge)** | Otro modelo (aquí Opus 5.5) que califica cada respuesta con una rúbrica de criterios concretos | Segundo revisor del comité |
| **K-means** | Clustering que agrupa a los clientes en k grupos, cada uno alrededor de un "centro" (el cliente promedio del grupo) | — |
| **Kappa** | Acuerdo entre dos calificadores descontando el que se daría por azar. 1 = acuerdo perfecto, 0 = no mejor que el azar | — |
| **Landing page** | La página a la que llega quien hace clic en un anuncio; ahí deja sus datos y se vuelve lead | El formulario de solicitud |
| **LangGraph** | Librería para armar un agente como un grafo de pasos (nodos) con estado, ramas en paralelo y pausas para humanos | — |
| **Lead** | Un prospecto que dejó sus datos y podría volverse cliente | Una solicitud |
| **Lead scoring** | Modelo que da a cada lead una probabilidad de cerrar, para que ventas atienda primero a los mejores | Scorecard de originación |
| **Lift** | Cuántas veces mejor convierte un grupo (el decil 1 del score) que el promedio. Lift 2 = el doble | Lift por deciles (mismo concepto) |
| **Calidad del lead** | Probabilidad de cerrar × valor esperado si cierra (propensión × valor) | PD × pérdida, pero al revés: aquí se suma valor |
| **MAP** | Estimación que busca los parámetros más probables combinando los datos con los *priors*. Es la versión de un solo punto de un modelo bayesiano | — |
| **MCP (Model Context Protocol)** | Estándar para que cualquier cliente (Claude Code, Claude Desktop, otro agente) use las mismas herramientas sin reescribirlas | — |
| **MMM** (*marketing mix modeling*) | Regresión sobre series semanales de ventas y gasto por canal para estimar cuánto aporta cada canal y dónde poner el siguiente peso | — |
| **MQL** (*marketing qualified lead*) | Lead que marketing considera calificado y pasa a ventas | Solicitud que pasa los filtros mínimos |
| **Optuna** | Librería que prueba combinaciones de hiperparámetros de forma inteligente (TPE: aprende de los intentos anteriores) | La misma del lab de crédito |
| **Prior** | Lo que se cree razonable antes de ver los datos (por ejemplo, que el ROI de un canal está entre 0 y 10). Evita que el modelo invente valores absurdos cuando hay pocos datos | Juicio experto en un scorecard con pocos malos |
| **Prueba geográfica** | Experimento por regiones: se apaga o sube un canal en unas regiones y se mantiene en otras, y se comparan las ventas. Mide el efecto real de un canal sin seguir a cada cliente | Piloto de política en unas sucursales |
| **Registro de ejecuciones** | Bitácora de cada pregunta: respuesta, herramientas usadas, costo, tiempo y firma humana | Log de decisiones del motor |
| **Release** | Versión etiquetada del repositorio (v1.0.0) con notas de lo que incluye; es la foto que se muestra en el portafolio | Versión aprobada de un modelo en producción |
| **RFM** | Segmentación clásica de clientes por **R**ecencia (días desde la última compra), **F**recuencia (cuántas compras) y **M**onto (cuánto gastó) | Comportamiento de pago reciente |
| **ROI incremental** | Valor que trajo la acción **y que no habría ocurrido sin ella**, dividido por lo que costó | ROI ajustado por riesgo |
| **ROI marginal** | Lo que rinde el **próximo** peso invertido en un canal, no el promedio de todo lo gastado. Es el que manda para mover presupuesto | Rentabilidad del siguiente tramo de cupo |
| **Saturación** | Cada peso adicional en un canal rinde menos que el anterior; pasado cierto gasto, más inversión casi no suma | Rendimientos decrecientes del cupo |
| **SEO / orgánica** | Tráfico que llega desde Google sin pagar por clic; cuesta contenido y equipo (costo fijo) | — |
| **Shapley (atribución)** | Reparte el mérito de una conversión según cuánto aporta cada canal en promedio en todas las combinaciones posibles; viene de la teoría de juegos | — |
| **Silhouette** | Qué tan bien separados están los clusters: cerca de 1 = grupos nítidos, cerca de 0 = se traslapan | — |
| **Streamlit** | Librería para hacer una app web con Python; Streamlit Community Cloud la hospeda gratis desde un repo de GitHub | — |
| **Text-to-SQL** | El agente escribe la consulta SQL a partir de la pregunta; aquí solo puede ejecutar SELECT sobre los Parquet publicados | — |
| **Traza (LangSmith)** | Registro paso a paso de una ejecución del agente: nodos, llamadas al modelo, herramientas y tiempos | Bitácora detallada de un proceso de decisión |
| **Uplift** | El efecto de contactar a una persona: cuánto sube su probabilidad de comprar **por** el contacto. Sirve para no gastar en quien iba a comprar igual | Efecto de una oferta de retención |
| **Utilidad extra** | Ventas extra × margen (30%) − gasto extra. Es el objetivo correcto para repartir presupuesto: vender más gastando de más puede restar | Margen financiero neto, no monto colocado |
| **Validación temporal** | Ajustar con meses pasados y evaluar con meses posteriores | *Out-of-time* (OOT) |
