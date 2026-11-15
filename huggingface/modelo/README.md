---
license: cc-by-nc-sa-4.0
language:
- es
library_name: sklearn
tags:
- marketing
- lead-scoring
- marketing-mix-modeling
- tabular-classification
pipeline_tag: tabular-classification
---

# marketing-agent-lab · modelos

Modelos del [marketing-agent-lab](https://github.com/neiderolaya26/marketing-agent-lab), un laboratorio de growth marketing de punta a punta construido solo con datos públicos. La documentación completa, en español, está en el repositorio. Este repo lo usa la [demo en Streamlit](https://github.com/neiderolaya26/marketing-agent-lab/tree/main/demo).

| Archivo | Qué es |
|---|---|
| `lead_scoring_logistica.joblib` | Lead scoring campeón: regresión logística (pipeline de scikit-learn) que da la probabilidad de que un lead de Olist cierre contrato |
| `mmm_con_control.json` | Parámetros del Marketing Mix Model elegido (arrastre θ, saturación λ y efecto β por canal), ajustado sobre **datos sintéticos** con un indicador de demanda |

## Lead scoring (`lead_scoring_logistica.joblib`)

- **Datos:** leads calificados (MQL) de Olist + Marketing Funnel. Se entrenó con los 3.343 leads de enero a marzo de 2018 y se probó con los 2.655 de abril y mayo. La tasa de cierre en la prueba es 11,8%.
- **Variables (8), todas conocidas al llegar el lead:**
  - Categóricas: canal, landing page (las frecuentes; el resto como «otra») y día de la semana.
  - Historia previa de la landing y del canal: leads previos (log), tasa de cierre previa suavizada, edad de la landing y si es nueva.
  - No hay fuga: la historia se calcula solo con lo ocurrido antes de cada lead.
- **Por qué una logística:** entre 6 modelos con 40 intentos de Optuna cada uno y validación temporal, empata con LightGBM y es la más simple (decisión 005).

| Métrica (abr–may 2018, fuera de tiempo) | Valor |
|---|---|
| AUC | 0,680 (IC 90% [0,655; 0,706]) |
| Brier | 0,101 |
| Lift del decil 1 | 1,88 |
| Cierres capturados por el 20% de leads con mejor score | 38,7% |

```python
import joblib, pandas as pd
from huggingface_hub import hf_hub_download

b = joblib.load(hf_hub_download("neiderolaya26/marketing-agent-lab", "lead_scoring_logistica.joblib"))
x = pd.DataFrame([...])[b["variables"]]                     # las 8 variables
for c, cats in b["categorias"].items():
    x[c] = pd.Categorical(x[c].astype(str), categories=cats)  # mismas categorías que en el entrenamiento
p = b["modelo"].predict_proba(x)[:, 1]
```

**Límites:** el AUC de 0,68 es modesto: sirve para ordenar el trabajo de ventas, no para descartar leads. El modelo depende de tasas previas que cambian con el tiempo, así que conviene monitorear su estabilidad (PSI). La licencia es CC BY-NC-SA 4.0, la misma de Olist: solo para uso no comercial.

## MMM (`mmm_con_control.json`)

- Las ventas semanales se modelan como base + Σ β·(1 − exp(−adstock/λ)), con arrastre geométrico θ por canal, para 5 canales: TV, búsqueda pagada, social, display y e-mail.
- Es una estimación MAP con *priors* débiles sobre el ROI, calibrada con un experimento de e-mail y con un indicador de demanda para corregir el gasto endógeno (decisiones 014–015).
- **Los datos son sintéticos:** el JSON trae también los parámetros verdaderos del generador. Muestra el método, no el ROI real de ningún canal.
- En 10 simulaciones, el plan de presupuesto de este modelo captura en mediana el 83% de la utilidad óptima.
