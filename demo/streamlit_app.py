"""Demo del marketing-agent-lab en Streamlit Community Cloud (decisión 019).

Corre solo código: valor por canal, score de leads, atribución contra la verdad, simulador de presupuesto
con las reglas del agente y los evals. No llama a ningún LLM; el agente con Claude vive en el repositorio.
El modelo de lead scoring y el MMM se descargan la primera vez de Hugging Face.

En local, desde la raíz del repo:  streamlit run demo/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

DEMO = Path(__file__).resolve().parent
RAIZ = DEMO.parent
sys.path.insert(0, str(RAIZ / "src"))

from growth_lab import mmm  # noqa: E402
from growth_lab.agente import CONGELADOS, validar_reparto  # noqa: E402

REPO_MODELO = "neiderolaya26/marketing-agent-lab"
PROCESADOS = RAIZ / "data" / "processed"
REPO = "https://github.com/neiderolaya26/marketing-agent-lab"
NOMBRES = {"tv": "TV", "paid_search": "Búsqueda pagada", "social": "Social", "display": "Display", "email": "E-mail"}

st.set_page_config(page_title="Lab de marketing y growth", page_icon="📈", layout="wide")


@st.cache_resource
def descargar(archivo: str) -> Path:
    local = RAIZ / "modelos" / archivo
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_MODELO, archivo))


@st.cache_data
def tabla(nombre: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESADOS / f"{nombre}.parquet")


@st.cache_resource
def modelo_leads() -> dict:
    return joblib.load(descargar("lead_scoring_logistica.joblib"))


@st.cache_resource
def parametros_mmm() -> dict:
    p = json.loads(descargar("mmm_con_control.json").read_text(encoding="utf-8"))
    p["gasto"] = np.array(p["gasto_semanal"])
    for k in ("theta", "lam", "beta"):
        p[k] = np.array(p[k])
    p["verdad"] = {k: np.array(v) for k, v in p["verdad"].items()}
    return p


st.title("Lab de marketing y growth")
st.caption(f"Datos públicos (Olist, Hillstrom, benchmark de atribución) y un MMM sintético con verdad conocida. "
           f"Sin LLM: el agente con Claude vive en el [repositorio]({REPO}). Cifras en BRL (Olist) y miles de USD (MMM).")

t_canales, t_leads, t_atr, t_pres, t_agente = st.tabs(
    ["Valor por canal", "Score de leads", "Atribución", "Simulador de presupuesto", "Agente y evals"])

with t_canales:
    st.subheader("¿Cuánto puedo pagar por un lead?")
    st.write("Valor de un lead = P(cierre) × GMV a 90 días × comisión supuesta del 15%. "
             "Es el **CPL de equilibrio**: pagar más destruye valor (decisiones 002 y 004).")
    r = tabla("canales_rentabilidad").merge(tabla("embudo_canal")[["canal", "tasa_cierre"]], on="canal")
    r = r[r.leads >= 50].sort_values("cpl_equilibrio", ascending=False)
    st.bar_chart(r.set_index("canal").cpl_equilibrio, horizontal=True, x_label="CPL de equilibrio (BRL)", y_label="")
    st.dataframe(r[["canal", "leads", "tasa_cierre", "gmv_por_lead", "cpl_equilibrio", "tipo_costo"]]
                 .rename(columns={"tasa_cierre": "tasa de cierre", "gmv_por_lead": "GMV por lead",
                                  "cpl_equilibrio": "CPL de equilibrio", "tipo_costo": "costo"}),
                 hide_index=True, column_config={"tasa de cierre": st.column_config.NumberColumn(format="percent")})

with t_leads:
    st.subheader("Score de un lead que el modelo no vio")
    st.write("Regresión logística entrenada con los leads de ene–mar 2018 (decisión 005; AUC 0,68 en abr–may). "
             "Usa solo lo que se sabe al llegar el lead: canal, landing, día y la historia previa de la landing.")
    ej = pd.read_csv(DEMO / "ejemplos_leads.csv")
    b = modelo_leads()
    x = ej[b["variables"]].copy()
    for c, cats in b["categorias"].items():
        x[c] = pd.Categorical(x[c].astype(str), categories=cats)
    ej["p_cierre"] = b["modelo"].predict_proba(x)[:, 1]
    ej["decil"] = 10 - pd.qcut(ej.p_cierre.rank(method="first"), 10, labels=False)
    i = st.selectbox("Lead de ejemplo (abr–may 2018)", ej.ejemplo,
                     format_func=lambda k: f"#{k} · {ej.canal[k]} · {ej.fecha[k]}")
    f = ej.loc[i]
    c1, c2, c3 = st.columns(3)
    c1.metric("P(cierre)", f"{f.p_cierre:.1%}", f"{f.p_cierre / b['tasa_base_entreno']:.1f}× la tasa base",
              delta_color="off")
    c2.metric("Decil entre los 200 ejemplos", f"{int(f.decil)}", "1 = los mejores", delta_color="off")
    c3.metric("¿Cerró de verdad?", "Sí" if f.cerro else "No")
    st.caption(f"Landing {'nueva' if f.landing_nueva else f'con {np.expm1(f.landing_leads_previos_log):.0f} leads previos'}"
               f" · tasa previa de la landing {f.landing_tasa_previa:.1%} · del canal {f.canal_tasa_previa:.1%}")
    d = ej.groupby("decil").agg(leads=("cerro", "size"), tasa_de_cierre=("cerro", "mean"))
    st.bar_chart(d.tasa_de_cierre, x_label="decil del score (1 = mejor)", y_label="tasa de cierre real")

with t_atr:
    st.subheader("¿Qué canal causa las ventas?")
    st.write("500 mil recorridos sintéticos con la atribución **verdadera** conocida. Las reglas reparten según quién "
             "aparece en el recorrido; Shapley sobre un modelo de conversión mide cuánto sube la probabilidad cada canal "
             "(decisión 013).")
    part = tabla("atr_participacion").set_index("canal")
    metodos = st.multiselect("Métodos", ["ultimo_clic", "primer_clic", "lineal", "markov", "shapley_datos", "shapley_modelo"],
                             default=["ultimo_clic", "markov", "shapley_modelo"])
    st.bar_chart(part[["verdad", *metodos]] * 100, stack=False, y_label="% de las conversiones")
    st.dataframe(tabla("atr_error")[["metodo", "mae_pp"]].rename(columns={"mae_pp": "error medio (puntos)"}),
                 hide_index=True)

with t_pres:
    st.subheader("Repartir la pauta por utilidad")
    p = parametros_mmm()
    st.write(f"MMM con control de demanda (decisión 015). Utilidad = ventas extra × {p['margen']:.0%} − gasto extra. "
             f"Reglas del agente (decisión 016): cada canal ±{p['limite']:.0%}, el total no sube y **búsqueda pagada "
             "no sube** hasta la prueba geo.")
    planes = {"Hoy": {}, "Aprobado (v3)": {"tv": 10, "paid_search": 0, "social": -30, "display": -19, "email": 30}}
    elegido = st.radio("Partir de", list(planes), horizontal=True)
    cols = st.columns(len(p["canales"]))
    cambios = []
    for col, c in zip(cols, p["canales"]):
        tope = 0 if c in CONGELADOS else int(p["limite"] * 100)
        v = col.slider(NOMBRES[c], -int(p["limite"] * 100), tope, planes[elegido].get(c, 0), step=1,
                       key=f"{elegido}-{c}", help="Congelado hasta la prueba geo" if c in CONGELADOS else None)
        cambios.append({"canal": c, "cambio_pct": float(v)})
    mult = np.array([1 + c["cambio_pct"] / 100 for c in cambios])
    gasto = p["gasto"]
    total = gasto[-52:].sum(axis=0)
    u_m = mmm.utilidad(gasto, mult, p["theta"], p["lam"], p["beta"])
    u_v = mmm.utilidad(gasto, mult, *(p["verdad"][k] for k in ("theta", "lam", "beta")))
    fallas = validar_reparto(cambios, gasto)
    c1, c2, c3 = st.columns(3)
    c1.metric("Gasto anual (miles USD)", f"{(total * mult).sum():,.0f}", f"{(total * mult).sum() - total.sum():+,.0f}",
              delta_color="inverse")
    c2.metric("Utilidad extra según el MMM", f"{u_m:+,.0f}")
    c3.metric("Utilidad extra real (simulador)", f"{u_v:+,.0f}", help="Solo existe porque los datos son sintéticos")
    if fallas:
        st.error("No cumple las reglas: " + "; ".join(fallas))
    else:
        st.success("Cumple las reglas. En el agente, esta propuesta igual esperaría la firma de una persona.")
    roi = tabla("mmm_roi")
    roi = roi[(roi.escenario == "endogeno") & (roi.version == p["version"])].set_index("canal")
    st.caption("ROI marginal según el MMM (equilibrio = " f"{1 / p['margen']:.2f}): "
               + " · ".join(f"{NOMBRES[c]} {roi.roi_marginal_est[c]:.2f}" for c in p["canales"]))

with t_agente:
    st.subheader("El agente analista de growth")
    st.write("Enrutador → especialistas de captación, retención y presupuesto → coordinador → reglas en código → "
             "firma humana. Corre con Claude Sonnet 5.5 en el repositorio; aquí solo se muestran sus evals "
             "(20 casos dorados, juez Opus 5.5).")
    r = tabla("evl_resumen").set_index("version")
    st.dataframe(r[["fallas_seguridad_o_reglas", "cifras_esperadas", "propuesta_ok", "juez_aprueba", "pasa_umbral"]]
                 .rename(columns={"fallas_seguridad_o_reglas": "fallas de seguridad", "cifras_esperadas": "cifras",
                                  "propuesta_ok": "propuesta correcta", "juez_aprueba": "juez aprueba",
                                  "pasa_umbral": "pasa"}),
                 column_config={k: st.column_config.NumberColumn(format="percent")
                                for k in ["cifras", "propuesta correcta", "juez aprueba"]})
    seg = tabla("evl_resultados")
    seg = seg[(seg.grupo == "seguridad") & (seg.version == seg.version.max())]
    for f in seg.itertuples():
        with st.expander(f"{'✅' if f.seguridad_ok else '❌'} {f.pregunta}"):
            st.write(f.respuesta)
