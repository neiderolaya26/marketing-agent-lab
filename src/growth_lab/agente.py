"""Sesión 11 · Agente analista de growth en LangGraph (decisión 016).

    pregunta ─ enrutador (Claude: qué especialistas hacen falta y qué le pregunta a cada uno)
                  ├─ captación   (Claude + SQL + valor por canal, lead scoring)
                  ├─ retención   (Claude + SQL + segmentos y CLV, experimentos y uplift, campañas)
                  └─ presupuesto (Claude + SQL + atribución, MMM, simulador de reparto)
                        ↓ (en paralelo, solo los que hagan falta)
               coordinador (Claude: respuesta integrada y, si se pide, propuesta de reparto)
                        ↓
               reglas (código: canales válidos, ±30% por canal y mismo total) ── fallas → coordinador
                        ↓
               aprobación (interrupt de LangGraph: solo si la propuesta mueve plata)

Los números salen siempre de las herramientas (código sobre los Parquet de data/processed). El agente
solo lee: la puerta `db.consultar()` deja pasar únicamente SELECT y la conexión no ve data/interim
(Complete Journey y Lenta a nivel fila nunca llegan al modelo).

Uso:
    python -m growth_lab.agente "¿Cuánto vale un lead de búsqueda pagada?"
    python -m growth_lab.agente "¿Cómo reparto el presupuesto del próximo año?"   # pide aprobación en consola
Requiere ANTHROPIC_API_KEY (variable de entorno o archivo .env, que no se sube a git).
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import os
import re
import sqlite3
import threading
import uuid
from typing import Annotated, Literal, TypedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel

from growth_lab import db, mmm

MODELO = "claude-sonnet-5-5"  # decisión 016
PRECIOS = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0)}
BETAS = ["server-side-fallback-2026-07-01"]
CON_FALLBACK = {"claude-sonnet-5-5", "claude-opus-5-5"}
MAX_TURNOS = 6
INTENTOS = 2
TOLERANCIA_TOTAL = 0.01  # el reparto no puede subir el total más de 1% (decisión 016: puede bajarlo)
RUTA_HILOS = db.RAIZ / "data" / "agente" / "hilos.db"
DECISIONES = db.RAIZ / "docs" / "decisiones.md"
ESPECIALISTAS = ["captacion", "retencion", "presupuesto"]
VERSION = "v5"  # v1: primera corrida; v2: búsqueda congelada y decisiones vigentes; v3: objetivo utilidad y total ≤ hoy;
# v4: propuesta solo si la pregunta pide un reparto (decisión 017, hallazgo del juez);
# v5: la nota de la política 010 dice sin ambigüedad «correo de hombres a todos los clientes» (no medida con evals)
CONGELADOS = {"paid_search": "no sube hasta tener la prueba geo (decisión 015)"}  # reglas duras de la decisión 016
DECISIONES_AREA = {  # decisiones vigentes que cada especialista recibe al inicio (decisión 016)
    "captacion": [2, 4, 5],
    "retencion": [3, 6, 7, 9, 10, 12],
    "presupuesto": [13, 14, 15],
}
for _area in DECISIONES_AREA.values():
    _area.append(16)  # la decisión del propio agente aplica a todos
Especialista = Literal["captacion", "retencion", "presupuesto"]
Canal = Literal["tv", "paid_search", "social", "display", "email"]


# ---------- utilidades ----------

def cargar_env(ruta=db.RAIZ / ".env") -> None:
    """Lee KEY=VALOR de .env sin sobrescribir variables ya definidas. Nunca imprime valores."""
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if "=" in linea and not linea.lstrip().startswith("#"):
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


def limpio(x):
    """Tipos de numpy/pandas a JSON (NaN → None, floats redondeados)."""
    if isinstance(x, dict):
        return {str(k): limpio(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [limpio(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if x is pd.NA or x is pd.NaT:
        return None
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def registros(df: pd.DataFrame) -> list[dict]:
    return limpio(df.to_dict(orient="records"))


# ---------- herramientas (código, sin LLM) ----------

class Herramientas:
    """Todo lo que el agente puede consultar. Solo lee data/processed."""

    def __init__(self):
        self.con = db.conectar(db.PROCESSED)
        self._mmm = None
        self._lock = threading.Lock()

    def _t(self, nombre: str) -> pd.DataFrame:
        return pd.read_parquet(db.PROCESSED / f"{nombre}.parquet")

    # comunes
    def tablas(self) -> dict:
        return {n: [f.name for f in pq.read_schema(p)] for n, p in sorted(db.tablas(db.PROCESSED).items())}

    def sql(self, consulta: str) -> dict:
        with self._lock:  # una conexión DuckDB compartida entre especialistas en paralelo
            df = db.consultar(self.con, consulta, limite=200)
        return {"filas": len(df), "datos": registros(df)}

    def resumen_decisiones(self, numeros: list[int]) -> str:
        """Título y bloque 'Decisión' de cada número: lo vigente, sin el contexto largo."""
        partes = []
        for n in numeros:
            texto = self.decision(n)["decision"]
            titulo = texto.splitlines()[0].lstrip("# ")
            m = re.search(r"^- \*\*Decisión:\*\*.*?(?=^- \*\*Por qué|\Z)", texto, flags=re.M | re.S)
            cuerpo = m.group(0).strip() if m else ""
            partes.append(f"{titulo}\n{cuerpo}")
        return "\n\n".join(partes)

    def decision(self, numero: int) -> dict:
        texto = DECISIONES.read_text(encoding="utf-8")
        m = re.search(rf"^## {numero:03d} — .*?(?=^---$|\Z)", texto, flags=re.M | re.S)
        if not m:
            raise ValueError(f"No existe la decisión {numero:03d}")
        return {"decision": m.group(0).strip()}

    # captación (sesiones 2 y 4)
    def valor_canales(self) -> dict:
        r = self._t("canales_rentabilidad")
        e = self._t("embudo_canal")[["canal", "tasa_cierre", "ingreso_por_lead"]]
        return {"nota": "Valor de un lead = P(cierre) × GMV a 90 días (decisión 002); comisión 15%. "
                        "cpl_equilibrio = lo máximo a pagar por lead; orgánica/directo son costo fijo (004).",
                "canales": registros(r.merge(e, on="canal", how="left"))}

    def lead_scoring(self) -> dict:
        m = self._t("leads_metricas")[["modelo", "auc", "auc_p05", "auc_p95", "lift_decil_1", "captura_top_20"]]
        d = self._t("leads_deciles")
        return {"nota": "Campeón: logística (decisión 005). Validación temporal abr–may 2018.",
                "modelos": registros(m), "deciles_campeon": registros(d[d.modelo == "logistica"].drop(columns="modelo"))}

    # retención (sesiones 3, 5, 6, 7 y 8)
    def segmentos_clv(self) -> dict:
        p = self._t("segmentos_perfil")[["segmento", "hogares", "recencia_dias", "frecuencia", "monto", "pct_hogares"]]
        c = self._t("clv_por_segmento")
        return {"nota": "K-means RFM k=4 (003). CLV a 13 semanas con LightGBM; churn parcial = gasto < 50% (006–007).",
                "segmentos": registros(p.merge(c, on=["segmento", "hogares"], how="left"))}

    def experimentos_uplift(self) -> dict:
        return {"nota": "Hillstrom. Margen 30% (009). Política vigente (010): enviar el correo diseñado para hombres "
                        "(«Mens E-Mail») a TODOS los clientes, hombres y mujeres. No es «correo solo a los hombres»: "
                        "la política 'todos_hombres' significa que todos reciben la versión de hombres.",
                "economia_experimento": registros(self._t("exp_economia")),
                "politicas_uplift": registros(self._t("upl_politicas")[["politica", "valor_por_cliente", "ic_inf", "ic_sup",
                                                                        "incremental_vs_nadie", "margen_neto_por_cliente"]])}

    def campanas(self) -> dict:
        e = self._t("cmp_efectos")
        e = e[(e.campana == "combinado") | (e.metodo == "did_doble_robusto")]
        return {"nota": "Campañas tipo A de Complete Journey (solo agregados). Decisiones 011–012.",
                "efectos": registros(e[["campana", "metodo", "ventana", "efecto", "ic_inf", "ic_sup"]]),
                "economia": registros(self._t("cmp_economia"))}

    # presupuesto (sesiones 9 y 10)
    def atribucion(self) -> dict:
        part = self._t("atr_participacion")[["canal", "verdad", "ultimo_clic", "markov", "shapley_modelo"]]
        cpc = self._t("atr_costo_por_conversion")[["canal", "verdad", "ultimo_clic", "shapley_modelo"]]
        return {"nota": "Benchmark sintético con verdad conocida. Método elegido: Shapley sobre modelo (013).",
                "participacion": registros(part), "costo_por_conversion": registros(cpc),
                "error_pp": registros(self._t("atr_error")[["metodo", "mae_pp"]])}

    def mmm(self) -> dict:
        r = self._t("mmm_roi")
        r = r[(r.escenario == "endogeno") & (r.version == mmm.VERSION_ELEGIDA)]
        p = self._t("mmm_presupuesto")
        return {"nota": (f"MMM sintético, versión {mmm.VERSION_ELEGIDA} (015). ROI en ventas por peso; equilibrio con "
                         f"margen {mmm.MARGEN:.0%} = {1 / mmm.MARGEN:.2f}. Búsqueda sigue inflada: validar con prueba geo."),
                "roi": registros(r[["canal", "roi_est", "roi_p05", "roi_p95", "roi_marginal_est"]]),
                "plan_del_modelo": registros(p[p.plan == mmm.VERSION_ELEGIDA].drop(columns="plan")),
                "resumen_planes": registros(self._t("mmm_presupuesto_resumen"))}

    def _modelo_mmm(self):
        with self._lock:
            if self._mmm is None:
                s = self._t("mmm_series")
                s = s[s.escenario == "endogeno"]
                gasto = s[[f"gasto_{c}" for c in mmm.CANALES]].to_numpy()
                r = self._t("mmm_roi")
                roi_exp = float(r[(r.escenario == "endogeno") & (r.canal == "email")].roi_experimento.dropna().iloc[0])
                m = mmm.MMM(mmm.VERSION_ELEGIDA, roi_email=roi_exp).ajustar(
                    gasto, s.ventas.to_numpy(), s.indice_demanda.to_numpy())
                self._mmm = (gasto, m)
            return self._mmm

    def simular_reparto(self, cambios: list[dict]) -> dict:
        """Ganancia de un reparto según el MMM y, como esto es un lab, según la verdad del simulador."""
        gasto, m = self._modelo_mmm()
        fallas = validar_reparto(cambios, gasto)
        mult = multiplicadores(cambios)
        hoy = np.ones(len(mmm.CANALES))
        v = mmm.VERDAD["theta"], mmm.VERDAD["lam"], mmm.VERDAD["beta"]
        total = gasto[-52:].sum(axis=0)
        return {"fallas": fallas,
                "ventas_extra_segun_modelo": mmm.ventas_medios(gasto, mult, m.theta, m.lam, m.beta)
                - mmm.ventas_medios(gasto, hoy, m.theta, m.lam, m.beta),
                "utilidad_extra_segun_modelo": mmm.utilidad(gasto, mult, m.theta, m.lam, m.beta),
                "utilidad_extra_verdad_simulador": mmm.utilidad(gasto, mult, *v),
                "gasto_anual_hoy": float(total.sum()), "gasto_anual_plan": float((total * mult).sum()),
                "unidades": "miles USD en un año; utilidad = ventas extra × margen 30% − gasto extra"}

    def plan_utilidad(self) -> dict:
        """Reparto que maximiza la utilidad según el MMM: cada canal ±30%, total sin subir, congelados sin subir."""
        gasto, m = self._modelo_mmm()
        mult = mmm.optimizar_utilidad(gasto, m.theta, m.lam, m.beta, techo={c: 1.0 for c in CONGELADOS})
        cambios = [{"canal": c, "cambio_pct": round(float((x - 1) * 100), 1)} for c, x in zip(mmm.CANALES, mult)]
        return {"cambios": cambios, "simulacion": self.simular_reparto(cambios)}


def multiplicadores(cambios: list[dict]) -> np.ndarray:
    mult = np.ones(len(mmm.CANALES))
    for c in cambios:
        mult[mmm.CANALES.index(c["canal"])] = 1 + c["cambio_pct"] / 100
    return mult


def validar_reparto(cambios: list[dict], gasto: np.ndarray | None = None) -> list[str]:
    """Reglas duras (decisiones 014–016): canales del MMM, cada uno ±30%, el total sin subir y canales
    congelados sin subir."""
    fallas = []
    for c in cambios:
        if c["canal"] not in mmm.CANALES:
            fallas.append(f"Canal desconocido: {c['canal']}")
        elif abs(c["cambio_pct"]) > mmm.LIMITE * 100 + 1e-6:
            fallas.append(f"{c['canal']}: {c['cambio_pct']:+.1f}% supera el límite de ±{mmm.LIMITE:.0%}")
        elif c["canal"] in CONGELADOS and c["cambio_pct"] > 1e-6:
            fallas.append(f"{c['canal']}: {CONGELADOS[c['canal']]}; no puede subir")
    if fallas:
        return fallas
    if gasto is None:
        s = pd.read_parquet(db.PROCESSED / "mmm_series.parquet")
        gasto = s[s.escenario == "endogeno"][[f"gasto_{c}" for c in mmm.CANALES]].to_numpy()
    total = gasto[-52:].sum(axis=0)
    cambio = float(((multiplicadores(cambios) - 1) * total).sum() / total.sum())
    if cambio > TOLERANCIA_TOTAL:
        fallas.append(f"El reparto sube el total en {cambio:+.1%}; el total puede bajar pero no subir")
    return fallas


# ---------- definición de herramientas para Claude ----------

def _esquema(props: dict | None = None) -> dict:
    props = props or {}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


COMUNES = [
    ("tablas", "Lista las tablas Parquet disponibles (data/processed) con sus columnas.", _esquema()),
    ("consultar_sql", "Ejecuta un SELECT de DuckDB sobre las tablas (máx. 200 filas). Solo lectura.",
     _esquema({"consulta": {"type": "string", "description": "Una sola sentencia SELECT."}})),
    ("leer_decision", "Lee el texto de una decisión registrada del lab (1 a 016).",
     _esquema({"numero": {"type": "integer"}})),
]
PROPIAS = {
    "captacion": [
        ("valor_canales", "Valor de un lead, CPL de equilibrio y tasa de cierre por canal de Olist.", _esquema()),
        ("lead_scoring", "Métricas de los 6 modelos de lead scoring y deciles del campeón.", _esquema()),
    ],
    "retencion": [
        ("segmentos_clv", "Segmentos RFM con CLV a 13 semanas, churn parcial y gasto en riesgo.", _esquema()),
        ("experimentos_uplift", "Resultado económico del A/B de Hillstrom y valor de cada política de envío.", _esquema()),
        ("campanas", "Efecto de las campañas tipo A (DiD doble robusto, placebos) y su economía.", _esquema()),
    ],
    "presupuesto": [
        ("atribucion", "Participación y costo por conversión por canal según cada método de atribución.", _esquema()),
        ("mmm", "ROI promedio y marginal por canal del MMM elegido y su plan de presupuesto.", _esquema()),
        ("plan_utilidad", "Reparto que maximiza la utilidad según el MMM (cada canal ±30%, total sin subir, búsqueda "
                          "congelada) con su simulación.", _esquema()),
        ("simular_reparto", "Simula un reparto (cada canal ±30%, total sin subir) y devuelve ventas y utilidad extra "
                            "y si cumple las reglas.",
         _esquema({"cambios": {"type": "array", "items": _esquema({
             "canal": {"type": "string", "enum": mmm.CANALES},
             "cambio_pct": {"type": "number", "description": "Cambio de gasto en %, entre -30 y 30."}})}})),
    ],
}
SISTEMA_BASE = """Eres el especialista de {area} de un equipo de analítica de growth. Respondes en español, en
máximo 200 palabras, a la pregunta que te pasa el coordinador.
Reglas:
- Toda cifra sale de una herramienta. Si los datos no alcanzan, dilo; no estimes de memoria.
- Cita de qué herramienta o tabla sale cada cifra y la decisión del lab que aplica (001–016).
- Usa consultar_sql solo si las herramientas de tu área no bastan.
- Tus recomendaciones respetan las decisiones vigentes de abajo; si los datos sugieren otra cosa, dilo
  como algo a revisar, no como recomendación.
Tu área: {detalle}

Decisiones vigentes de tu área:
{decisiones}"""
DETALLE = {
    "captacion": "embudo de Olist por canal, valor de un lead, CPL de equilibrio y lead scoring (sesiones 2 y 4).",
    "retencion": "segmentos RFM, CLV y churn de Complete Journey (solo agregados), A/B y uplift de Hillstrom y "
                 "efecto de campañas sin experimento (sesiones 3, 5, 6, 7 y 8).",
    "presupuesto": "atribución multitoque y MMM con su plan de presupuesto (sesiones 9 y 10). Si te piden un "
                   "reparto, parte de plan_utilidad y simula cualquier variante con simular_reparto. El objetivo es "
                   "utilidad (ventas × 30% − gasto), no ventas: un canal con ROI marginal bajo el equilibrio (3,33) "
                   "resta utilidad aunque sume ventas.",
}


class Encargo(BaseModel):
    especialista: Especialista
    pregunta: str


class Ruta(BaseModel):
    encargos: list[Encargo]
    fuera_de_alcance: bool
    pide_reparto: bool


class Cambio(BaseModel):
    canal: Canal
    cambio_pct: float


class Respuesta(BaseModel):
    respuesta: str
    cifras_clave: list[str]
    propuesta: list[Cambio]
    justificacion_propuesta: str


SISTEMA_ENRUTADOR = """Eres el coordinador de un equipo de analítica de growth de un lab de marketing con datos
públicos (Olist, Complete Journey, Hillstrom, benchmark de atribución y un MMM sintético). Decide qué especialistas
necesita la pregunta y qué le preguntas a cada uno (concreto, en español):
- captacion: leads, canales de Olist, valor de un lead, CPL, lead scoring.
- retencion: segmentos, CLV, churn, experimentos A/B, uplift, campañas y cupones.
- presupuesto: atribución, MMM, ROI por canal de pauta, repartir o mover presupuesto.
Usa el mínimo de especialistas. Si la pregunta no tiene que ver con el lab, marca fuera_de_alcance y no encargues nada.
Marca pide_reparto solo si la pregunta pide explícitamente repartir, mover o cambiar el presupuesto entre canales."""

SISTEMA_COORDINADOR = """Eres el coordinador de un equipo de analítica de growth. Con los análisis de los
especialistas, responde la pregunta en español, claro y corto (máximo 250 palabras), para alguien de negocio.
- Usa solo cifras que aparezcan en los análisis; en cifras_clave lista las 2–5 más importantes con su fuente.
- Llena propuesta SOLO si la pregunta pide repartir o mover presupuesto entre canales del MMM (tv, paid_search,
  social, display, email): cada canal entre −30% y +30%; el total puede bajar pero no subir. El objetivo es
  utilidad (ventas extra × 30% − gasto extra), no ventas. Si no se pide reparto, deja propuesta vacía.
- Búsqueda pagada (paid_search) está congelada: no puede subir hasta tener una prueba geográfica (decisión 015).
  Reparte entre los demás canales.
- Una propuesta no se ejecuta: un humano la aprueba o la rechaza.
- Si la pregunta está fuera del alcance del lab, dilo y no inventes."""


# ---------- llamadas a Claude ----------

class Medidor:
    """Envuelve el cliente y suma llamadas y tokens (seguro con nodos en paralelo)."""

    def __init__(self, cliente, modelo: str = MODELO):
        self._cliente, self._lock, self.modelo = cliente, threading.Lock(), modelo
        self.llamadas = self.tokens_entrada = self.tokens_salida = 0
        self.beta = type("Beta", (), {})()
        self.beta.messages = type("Mensajes", (), {})()
        self.beta.messages.create = self._medir(cliente.beta.messages.create)
        self.beta.messages.parse = self._medir(cliente.beta.messages.parse)

    def _medir(self, metodo):
        def envuelto(**kw):
            r = metodo(**kw)
            uso = getattr(r, "usage", None)
            with self._lock:
                self.llamadas += 1
                self.tokens_entrada += getattr(uso, "input_tokens", 0) or 0
                self.tokens_salida += getattr(uso, "output_tokens", 0) or 0
            return r
        return envuelto

    def costo_usd(self) -> float:
        entrada, salida = PRECIOS.get(self.modelo, (0.0, 0.0))
        return (self.tokens_entrada * entrada + self.tokens_salida * salida) / 1e6


def _llamar(cliente, modelo: str, metodo, **kw):
    extra = {"betas": BETAS, "fallbacks": "default"} if modelo in CON_FALLBACK else {}
    return metodo(model=modelo, **extra, **kw)


def _texto(r) -> str:
    return "".join(b.text for b in r.content if b.type == "text").strip()


def funciones(herr: Herramientas, nombre: str) -> dict:
    f = {"tablas": lambda a: herr.tablas(), "consultar_sql": lambda a: herr.sql(a["consulta"]),
         "leer_decision": lambda a: herr.decision(a["numero"])}
    for n, _, _ in PROPIAS[nombre]:
        f[n] = (lambda a: herr.simular_reparto(a["cambios"])) if n == "simular_reparto" else \
            (lambda a, n=n: getattr(herr, n)())
    return f


def correr_especialista(cliente, herr: Herramientas, nombre: str, pregunta: str, modelo: str = MODELO) -> dict:
    tools = [{"name": n, "description": d, "input_schema": e, "strict": True} for n, d, e in COMUNES + PROPIAS[nombre]]
    fn = funciones(herr, nombre)
    sistema = SISTEMA_BASE.format(area=nombre, detalle=DETALLE[nombre],
                                  decisiones=herr.resumen_decisiones(DECISIONES_AREA[nombre]))
    mensajes = [{"role": "user", "content": pregunta}]
    llamadas, agotado, r = [], True, None
    for _ in range(MAX_TURNOS):
        r = _llamar(cliente, modelo, cliente.beta.messages.create, max_tokens=8000, system=sistema,
                    tools=tools, messages=mensajes, output_config={"effort": "medium"})
        if r.stop_reason == "refusal":
            return {"analisis": "El modelo declinó responder.", "herramientas": llamadas, "agotado": False}
        mensajes.append({"role": "assistant", "content": r.content})
        usos = [b for b in r.content if b.type == "tool_use"]
        if not usos:
            agotado = False
            break
        resultados = []
        for u in usos:
            try:
                salida = fn[u.name](u.input)
                resultados.append({"type": "tool_result", "tool_use_id": u.id,
                                   "content": json.dumps(salida, ensure_ascii=False)})
                llamadas.append({"herramienta": u.name, "entrada": u.input, "ok": True, "salida": limpio(salida)})
            except Exception as e:  # el error vuelve al modelo (p. ej. un SQL que no es SELECT)
                resultados.append({"type": "tool_result", "tool_use_id": u.id, "content": f"Error: {e}",
                                   "is_error": True})
                llamadas.append({"herramienta": u.name, "entrada": u.input, "ok": False, "error": str(e)})
        mensajes.append({"role": "user", "content": resultados})
    analisis = _texto(r) if r is not None else ""
    if agotado:
        analisis += f"\n\n[Análisis incompleto: se agotaron los {MAX_TURNOS} turnos.]"
    return {"analisis": analisis, "herramientas": llamadas, "agotado": agotado}


def enrutar(cliente, pregunta: str, modelo: str = MODELO) -> dict:
    r = _llamar(cliente, modelo, cliente.beta.messages.parse, max_tokens=4000, system=SISTEMA_ENRUTADOR,
                messages=[{"role": "user", "content": pregunta}], output_format=Ruta,
                output_config={"effort": "low"})
    if r.stop_reason == "refusal" or r.parsed_output is None:
        return {"encargos": [], "fuera_de_alcance": True, "pide_reparto": False}
    ruta = r.parsed_output.model_dump()
    vistos, encargos = set(), []
    for e in ruta["encargos"]:  # un encargo por especialista
        if e["especialista"] not in vistos:
            vistos.add(e["especialista"])
            encargos.append(e)
    return {"encargos": encargos, "fuera_de_alcance": ruta["fuera_de_alcance"] and not encargos,
            "pide_reparto": ruta["pide_reparto"]}


def coordinar(cliente, estado, modelo: str = MODELO) -> dict:
    partes = [f"Pregunta: {estado['pregunta']}",
              "Análisis de los especialistas:\n" + json.dumps(
                  {k: v["analisis"] for k, v in estado.get("especialistas", {}).items()}, ensure_ascii=False, indent=1)]
    if estado.get("ruta", {}).get("fuera_de_alcance"):
        partes.append("El enrutador marcó la pregunta como fuera del alcance del lab.")
    if estado.get("fallas"):
        partes.append("Tu propuesta anterior no cumple las reglas. Corrígela:\n- " + "\n- ".join(estado["fallas"]))
    r = _llamar(cliente, modelo, cliente.beta.messages.parse, max_tokens=16000, system=SISTEMA_COORDINADOR,
                messages=[{"role": "user", "content": "\n\n".join(partes)}], output_format=Respuesta,
                output_config={"effort": "high"})
    if r.stop_reason == "refusal" or r.parsed_output is None:
        return {"respuesta": "No pude armar una respuesta.", "cifras_clave": [], "propuesta": [],
                "justificacion_propuesta": ""}
    return r.parsed_output.model_dump()


# ---------- grafo ----------

class Estado(TypedDict, total=False):
    pregunta: str
    hilo: str
    ruta: dict
    especialistas: Annotated[dict, operator.or_]
    coordinador: dict
    fallas: list[str]
    intentos: int
    simulacion: dict
    final: dict


PROYECTO_LANGSMITH = "marketing-agent-lab"


def quiere_trazas(trazar: bool | None = None) -> bool:
    """Las trazas en LangSmith solo se envían si se piden (--trazar o MARKETING_LAB_TRAZAR=1) y hay
    LANGSMITH_API_KEY (decisión 019). Por defecto el agente no manda nada a LangSmith."""
    if trazar is None:
        trazar = os.environ.get("MARKETING_LAB_TRAZAR", "").lower() in {"1", "true", "si", "sí"}
    return bool(trazar and os.environ.get("LANGSMITH_API_KEY"))


def cliente_anthropic(trazar: bool | None = None):
    import anthropic

    cargar_env()
    cliente = anthropic.Anthropic()
    if quiere_trazas(trazar):  # traza cada llamada a Claude y, con LANGSMITH_TRACING, el grafo de LangGraph
        from langsmith.wrappers import wrap_anthropic

        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", PROYECTO_LANGSMITH)
        cliente = wrap_anthropic(cliente)
    return cliente


def checkpointer_sqlite(ruta=RUTA_HILOS):
    """Checkpointer persistente y sin pickle: la aprobación puede llegar desde otro proceso."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    ruta.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(ruta, check_same_thread=False), serde=JsonPlusSerializer())


def construir(cliente=None, herr: Herramientas | None = None, modelo: str = MODELO, checkpointer=None):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send, interrupt

    cliente = cliente or cliente_anthropic()
    herr = herr or Herramientas()

    def nodo_enrutar(estado):
        return {"ruta": enrutar(cliente, estado["pregunta"], modelo), "especialistas": {}, "intentos": 0}

    def repartir(estado):
        encargos = estado["ruta"]["encargos"]
        if not encargos:
            return "coordinador"
        return [Send("especialista", {"nombre": e["especialista"], "pregunta": e["pregunta"]}) for e in encargos]

    def nodo_especialista(encargo):
        return {"especialistas": {encargo["nombre"]: correr_especialista(
            cliente, herr, encargo["nombre"], encargo["pregunta"], modelo) | {"pregunta": encargo["pregunta"]}}}

    def nodo_coordinador(estado):
        return {"coordinador": coordinar(cliente, estado, modelo), "intentos": estado.get("intentos", 0) + 1}

    def nodo_reglas(estado):
        cambios = estado["coordinador"]["propuesta"]
        if cambios and not estado["ruta"].get("pide_reparto", True):  # v4: nadie pidió mover plata
            return {"fallas": [], "simulacion": {},
                    "coordinador": estado["coordinador"] | {"propuesta": [], "propuesta_descartada": cambios}}
        if not cambios:
            return {"fallas": [], "simulacion": {}}
        sim = herr.simular_reparto(cambios)
        return {"fallas": sim["fallas"], "simulacion": sim}

    def despues_reglas(estado):
        return "coordinador" if estado["fallas"] and estado["intentos"] < INTENTOS else "aprobacion"

    def nodo_aprobacion(estado):
        c = estado["coordinador"]
        final = {"respuesta": c["respuesta"], "cifras_clave": c["cifras_clave"], "propuesta": c["propuesta"],
                 "especialistas": sorted(estado.get("especialistas", {}))}
        if not c["propuesta"]:
            return {"final": final | {"estado": "respondida"}}
        if estado["fallas"]:
            return {"final": final | {"estado": "propuesta descartada por reglas", "fallas": estado["fallas"],
                                      "propuesta": []}}
        decision = interrupt({"propuesta": c["propuesta"], "justificacion": c["justificacion_propuesta"],
                              "simulacion": estado["simulacion"]})
        if not isinstance(decision, dict) or not isinstance(decision.get("aprueba"), bool):
            raise ValueError("La aprobación debe ser {'aprueba': bool, 'comentario': str}")
        return {"final": final | {"estado": "aprobada" if decision["aprueba"] else "rechazada",
                                  "aprobacion": decision, "simulacion": estado["simulacion"]}}

    g = StateGraph(Estado)
    g.add_node("enrutador", nodo_enrutar)
    g.add_node("especialista", nodo_especialista)
    g.add_node("coordinador", nodo_coordinador)
    g.add_node("reglas", nodo_reglas)
    g.add_node("aprobacion", nodo_aprobacion)
    g.add_edge(START, "enrutador")
    g.add_conditional_edges("enrutador", repartir, ["especialista", "coordinador"])
    g.add_edge("especialista", "coordinador")
    g.add_edge("coordinador", "reglas")
    g.add_conditional_edges("reglas", despues_reglas, ["coordinador", "aprobacion"])
    g.add_edge("aprobacion", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else checkpointer_sqlite())


def preguntar(grafo, pregunta: str, hilo: str | None = None) -> dict:
    """Corre el grafo. Si hay propuesta de reparto, el resultado trae '__interrupt__' y espera aprobar()."""
    hilo = hilo or uuid.uuid4().hex
    estado = grafo.invoke({"pregunta": pregunta, "hilo": hilo}, config={"configurable": {"thread_id": hilo}})
    return estado | {"hilo": hilo}


def aprobar(grafo, hilo: str, aprueba: bool, comentario: str = "") -> dict:
    from langgraph.types import Command

    return grafo.invoke(Command(resume={"aprueba": aprueba, "comentario": comentario}),
                        config={"configurable": {"thread_id": hilo}})


DEMO = [
    "¿Qué canal de Olist trae los leads más valiosos y cuánto puedo pagar como máximo por cada lead?",
    "¿Qué segmento de clientes debo priorizar para retención y cuánto gasto está en riesgo?",
    "¿Vale la pena seguir mandando el correo de Hillstrom? ¿A quién se lo mando?",
    "¿Display está sobrevalorado? Compara lo que dicen la atribución y el MMM.",
    "¿Dónde pongo el siguiente peso: en captar leads de búsqueda pagada o en retener a los clientes que se están enfriando?",
    "Propón cómo repartir el presupuesto de medios del próximo año entre canales, con el mismo total.",
    "¿Cuál es la tasa de interés del Banco de la República este mes?",
]


def demo(preguntas: list[str] = DEMO, modelo: str = MODELO) -> pd.DataFrame:
    """Corre la batería de demo y deja en data/agente/ los hilos que esperan aprobación."""
    import time

    medidor = Medidor(cliente_anthropic(), modelo)
    grafo = construir(medidor, modelo=modelo)
    filas = []
    for q in preguntas:
        antes = (medidor.llamadas, medidor.tokens_entrada, medidor.tokens_salida, medidor.costo_usd())
        t0 = time.time()
        e = preguntar(grafo, q)
        pendiente = "__interrupt__" in e
        f = e.get("final") or {"estado": "pendiente de aprobación", "respuesta": e["coordinador"]["respuesta"],
                               "cifras_clave": e["coordinador"]["cifras_clave"],
                               "propuesta": e["coordinador"]["propuesta"], "especialistas": sorted(e["especialistas"])}
        herramientas = [h["herramienta"] for v in e.get("especialistas", {}).values() for h in v["herramientas"]]
        filas.append({"version": VERSION, "pregunta": q, "hilo": e["hilo"], "estado": f["estado"], "especialistas": ", ".join(f["especialistas"]),
                      "herramientas": ", ".join(herramientas), "respuesta": f["respuesta"],
                      "cifras_clave": " | ".join(f["cifras_clave"]),
                      "propuesta": json.dumps(limpio(f["propuesta"]), ensure_ascii=False),
                      "simulacion": json.dumps(limpio(e.get("simulacion", {})), ensure_ascii=False) if pendiente else "",
                      "llamadas": medidor.llamadas - antes[0], "tokens_entrada": medidor.tokens_entrada - antes[1],
                      "tokens_salida": medidor.tokens_salida - antes[2], "costo_usd": medidor.costo_usd() - antes[3],
                      "segundos": time.time() - t0, "modelo": modelo})
    d = pd.DataFrame(filas)
    ruta = db.PROCESSED / "agt_demo.parquet"
    if ruta.exists():  # se conservan las corridas de otras versiones para compararlas
        previas = pd.read_parquet(ruta)
        d = pd.concat([previas[previas.version != VERSION], d], ignore_index=True)
    d.to_parquet(ruta, compression="zstd", index=False)
    return d


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pregunta")
    p.add_argument("--modelo", default=MODELO, choices=sorted(PRECIOS))
    p.add_argument("--trazar", action="store_true", help="Envía las trazas a LangSmith (requiere LANGSMITH_API_KEY)")
    a = p.parse_args()
    import time

    from growth_lab import registro

    medidor = Medidor(cliente_anthropic(trazar=a.trazar or None), a.modelo)
    grafo = construir(medidor, modelo=a.modelo)
    t0 = time.time()
    estado = preguntar(grafo, a.pregunta)
    uso = {"llamadas": medidor.llamadas, "tokens_entrada": medidor.tokens_entrada,
           "tokens_salida": medidor.tokens_salida, "costo_usd": medidor.costo_usd()}
    with registro.conectar() as con:  # decisión 017: toda ejecución queda en data/agente/ejecuciones.db
        registro.guardar(con, estado, origen="cli", version=VERSION, modelo=a.modelo, uso=uso,
                         segundos=time.time() - t0)
    if "__interrupt__" in estado:
        pendiente = estado["__interrupt__"][0].value
        print(json.dumps(limpio(pendiente), ensure_ascii=False, indent=1))
        si = input("¿Apruebas el reparto? [s/N] ").strip().lower() in {"s", "si", "sí"}
        hilo = estado["hilo"]
        estado = aprobar(grafo, hilo, si, input("Comentario: ").strip())
        with registro.conectar() as con:
            registro.actualizar_firma(con, hilo, estado["final"])
    f = estado["final"]
    print(f"\n[{f['estado']}] especialistas: {', '.join(f['especialistas']) or 'ninguno'}\n")
    print(f["respuesta"])
    print("\nCifras clave:\n- " + "\n- ".join(f["cifras_clave"]))
    print(f"\n{medidor.llamadas} llamadas · {medidor.tokens_entrada:,} tokens de entrada · "
          f"{medidor.tokens_salida:,} de salida · USD {medidor.costo_usd():.3f}")


if __name__ == "__main__":
    main()
