"""Modelos y datos del lab en Hugging Face, y ejemplos para la demo (decisión 019).

    python -m growth_lab.publicar_hf guardar    # modelos/ (lead scoring + MMM) y demo/ejemplos_leads.csv
    python -m growth_lab.publicar_hf usuario    # la cuenta del token (HF_TOKEN en .env)
    python -m growth_lab.publicar_hf publicar   # sube el repo de modelo y el de datos al Hub

Qué se publica:
- **Modelo** (neiderolaya26/marketing-agent-lab): el lead scoring campeón (regresión logística entrenada
  con los leads de Olist de ene–mar 2018, decisión 005) y los parámetros del MMM elegido (con control de
  demanda, decisión 015), ajustado sobre datos sintéticos.
- **Datos** (neiderolaya26/marketing-agent-lab-data): los Parquet de data/processed, que ya son públicos
  en GitHub.
- **No se publica** el modelo de CLV: está entrenado con The Complete Journey, cuya licencia solo permite
  publicar agregados.

La demo (demo/streamlit_app.py) corre en Streamlit Community Cloud y descarga los modelos del Hub.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from growth_lab import agente, db, leads, mmm

REPO_MODELO = "neiderolaya26/marketing-agent-lab"
REPO_DATOS = "neiderolaya26/marketing-agent-lab-data"
MODELOS = db.RAIZ / "modelos"
DEMO = db.RAIZ / "demo"
TARJETAS = db.RAIZ / "huggingface"
ARCHIVO_LEADS = "lead_scoring_logistica.joblib"
ARCHIVO_MMM = "mmm_con_control.json"


def guardar_lead_scoring(carpeta: Path = MODELOS) -> Path:
    from growth_lab import leads_modelos as lm

    entreno, prueba = leads.particion(leads.poblacion(db.conectar(db.PROCESSED)))
    modelo = lm.ajustar_final(lm.CAMPEON, entreno)
    carpeta.mkdir(exist_ok=True)
    ruta = carpeta / ARCHIVO_LEADS
    joblib.dump({"modelo": modelo, "variables": leads.VARIABLES,
                 "categorias": {c: list(entreno[c].cat.categories) for c in leads.CATEGORICAS},
                 "tasa_base_entreno": float(entreno.cerro.mean())}, ruta)
    return ruta


def guardar_mmm(carpeta: Path = MODELOS) -> Path:
    herr = agente.Herramientas()
    gasto, m = herr._modelo_mmm()
    s = pd.read_parquet(db.PROCESSED / "mmm_series.parquet")
    s = s[s.escenario == "endogeno"]
    carpeta.mkdir(exist_ok=True)
    ruta = carpeta / ARCHIVO_MMM
    ruta.write_text(json.dumps({
        "version": mmm.VERSION_ELEGIDA, "canales": mmm.CANALES, "margen": mmm.MARGEN, "limite": mmm.LIMITE,
        "theta": m.theta.tolist(), "lam": m.lam.tolist(), "beta": m.beta.tolist(),
        "verdad": {k: v.tolist() for k, v in mmm.VERDAD.items()},
        "gasto_semanal": gasto.tolist(), "semanas": int(len(s)),
        "nota": "MMM ajustado sobre 156 semanas sintéticas (escenario con gasto endógeno). Los parámetros "
                "'verdad' son los del generador: solo existen porque los datos son sintéticos.",
    }, ensure_ascii=False), encoding="utf-8")
    return ruta


def guardar_ejemplos(n: int = 200, semilla: int = 0, destino: Path = DEMO / "ejemplos_leads.csv") -> pd.DataFrame:
    """Leads de abr–may 2018 (el modelo no los vio), sin identificadores, con el resultado real."""
    entreno, prueba = leads.particion(leads.poblacion(db.conectar(db.PROCESSED)))
    m = prueba.sample(min(n, len(prueba)), random_state=semilla)
    ej = m[leads.VARIABLES + ["cerro"]].copy()
    for c in leads.CATEGORICAS:
        ej[c] = ej[c].astype(str)
    ej["fecha"] = m.fecha.dt.date.astype(str)
    ej.insert(0, "ejemplo", np.arange(len(ej)))
    destino.parent.mkdir(exist_ok=True)
    ej.to_csv(destino, index=False)
    return ej


def _token() -> str:
    agente.cargar_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Falta HF_TOKEN en .env")
    return token


def usuario() -> str:
    from huggingface_hub import HfApi

    return HfApi(token=_token()).whoami()["name"]


def publicar() -> dict:
    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    api.create_repo(REPO_MODELO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=TARJETAS / "modelo" / "README.md", path_in_repo="README.md", repo_id=REPO_MODELO)
    for archivo in (ARCHIVO_LEADS, ARCHIVO_MMM):
        api.upload_file(path_or_fileobj=MODELOS / archivo, path_in_repo=archivo, repo_id=REPO_MODELO)
    api.create_repo(REPO_DATOS, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=TARJETAS / "datos" / "README.md", path_in_repo="README.md",
                    repo_id=REPO_DATOS, repo_type="dataset")
    api.upload_folder(folder_path=db.PROCESSED, path_in_repo="processed", repo_id=REPO_DATOS, repo_type="dataset",
                      allow_patterns=["*.parquet"])
    return {"modelo": f"https://huggingface.co/{REPO_MODELO}",
            "datos": f"https://huggingface.co/datasets/{REPO_DATOS}"}


def main() -> None:
    p = argparse.ArgumentParser(description="Modelos y datos del lab en Hugging Face")
    p.add_argument("accion", choices=["guardar", "usuario", "publicar"])
    a = p.parse_args()
    if a.accion == "guardar":
        print(guardar_lead_scoring(), guardar_mmm(), len(guardar_ejemplos()), "ejemplos")
    elif a.accion == "usuario":
        print(usuario())
    else:
        print(json.dumps(publicar(), indent=1))


if __name__ == "__main__":
    main()
