# Glosario

Términos del lab en palabras simples. Crece en cada sesión. La columna "En crédito" da la analogía cuando existe.

| Término | Qué es | En crédito |
|---|---|---|
| **A/B test** | Experimento: se reparte gente al azar entre dos versiones (correo, landing, oferta) y se compara el resultado. Como es al azar, la diferencia es causal | Piloto de política con grupo de control |
| **Adstock** | El efecto de la publicidad no desaparece el día que se pauta: se arrastra y se va apagando. El MMM lo modela | — |
| **AOV** (*average order value*) | Ticket medio: GMV / número de pedidos | Monto medio desembolsado |
| **ARI** (índice de Rand ajustado) | Mide cuánto coinciden dos agrupaciones de las mismas personas: 1 = idénticas, 0 = como al azar. Lo usamos para ver si K-means da lo mismo con otra semilla | — |
| **Atribución** | Repartir el mérito de una venta entre los canales que tocó el cliente antes de comprar (vio un anuncio, buscó en Google, abrió un correo) | — |
| **Bootstrap** | Remuestrear los datos con reemplazo muchas veces para ver cuánto varía una cifra; de ahí sale el intervalo de confianza | El mismo que usamos para la diferencia de AUC |
| **CAC** (costo de adquisición de cliente) | Cuánto cuesta conseguir un cliente: gasto en marketing / clientes nuevos | Costo por desembolso |
| **Canal** | Por dónde llegó el lead: búsqueda pagada (Google Ads), orgánica (SEO), social, e-mail, display (banners), referidos, directo | Canal de originación |
| **Cohorte** | Grupo que empezó en el mismo período (mes de primera compra) y se sigue en el tiempo | Cosecha |
| **Comisión** | Lo que Olist se queda de cada venta de un vendedor. No es pública; suponemos 15% | Tasa / margen financiero |
| **Contrato maduro** | Un contrato con al menos 90 días observables después de firmar; solo esos entran al valor por lead | Préstamo con ventana de desempeño completa |
| **CPL** (costo por lead) | Cuánto se paga por cada lead | Costo por solicitud |
| **CPL de equilibrio** | El máximo que puedes pagar por un lead sin perder plata = ingreso esperado por lead | Tasa de equilibrio (ROI = 0) |
| **Display** | Anuncios gráficos (banners) en sitios web | — |
| **Embudo** (*funnel*) | Las etapas que recorre un prospecto: lead → contrato → vendedor que vende | Solicitud → aprobación → desembolso → buen pago |
| **η² (eta cuadrado)** | Qué parte de la variación de una cifra (el gasto futuro) explican los grupos (los segmentos). 0,47 = los segmentos explican el 47% | Como el R² de una regresión con una variable categórica |
| **GMM** (mezcla de gaussianas) | Clustering "blando": supone que cada grupo es una nube con forma de campana y da la probabilidad de pertenecer a cada uno | — |
| **GMV** (*gross merchandise value*) | Valor total vendido en la plataforma, antes de comisiones | Monto colocado |
| **Importancia por permutación** | Cuánto empeora el modelo (cae el AUC) si se desordena una variable al azar. Funciona para cualquier modelo | Alternativa a SHAP o al IV |
| **K-means** | Clustering que agrupa a los clientes en k grupos, cada uno alrededor de un "centro" (el cliente promedio del grupo) | — |
| **Landing page** | La página a la que llega quien hace clic en un anuncio; ahí deja sus datos y se vuelve lead | El formulario de solicitud |
| **Lead** | Un prospecto que dejó sus datos y podría volverse cliente | Una solicitud |
| **Lead scoring** | Modelo que da a cada lead una probabilidad de cerrar, para que ventas atienda primero a los mejores | Scorecard de originación |
| **Lift** | Cuántas veces mejor convierte un grupo (el decil 1 del score) que el promedio. Lift 2 = el doble | Lift por deciles (mismo concepto) |
| **Calidad del lead** | Probabilidad de cerrar × valor esperado si cierra (propensión × valor) | PD × pérdida, pero al revés: aquí se suma valor |
| **MMM** (*marketing mix modeling*) | Regresión sobre series semanales de ventas y gasto por canal para estimar cuánto aporta cada canal y dónde poner el siguiente peso | — |
| **MQL** (*marketing qualified lead*) | Lead que marketing considera calificado y pasa a ventas | Solicitud que pasa los filtros mínimos |
| **Optuna** | Librería que prueba combinaciones de hiperparámetros de forma inteligente (TPE: aprende de los intentos anteriores) | La misma del lab de crédito |
| **RFM** | Segmentación clásica de clientes por **R**ecencia (días desde la última compra), **F**recuencia (cuántas compras) y **M**onto (cuánto gastó) | Comportamiento de pago reciente |
| **ROI incremental** | Valor que trajo la acción **y que no habría ocurrido sin ella**, dividido por lo que costó | ROI ajustado por riesgo |
| **SEO / orgánica** | Tráfico que llega desde Google sin pagar por clic; cuesta contenido y equipo (costo fijo) | — |
| **Shapley (atribución)** | Reparte el mérito de una conversión según cuánto aporta cada canal en promedio en todas las combinaciones posibles; viene de la teoría de juegos | — |
| **Silhouette** | Qué tan bien separados están los clusters: cerca de 1 = grupos nítidos, cerca de 0 = se traslapan | — |
| **Uplift** | El efecto de contactar a una persona: cuánto sube su probabilidad de comprar **por** el contacto. Sirve para no gastar en quien iba a comprar igual | Efecto de una oferta de retención |
| **Validación temporal** | Ajustar con meses pasados y evaluar con meses posteriores | *Out-of-time* (OOT) |
