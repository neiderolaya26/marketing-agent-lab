"""Sesión 4 · Búsqueda del mejor modelo de lead scoring (Optuna + validación temporal).

Uso:
    python -m growth_lab.leads_modelos 40     # 40 intentos de Optuna por modelo + prueba final
    python -m growth_lab.leads_modelos 0      # solo la prueba final, con los parámetros ya guardados

Cómo se elige sin hacer trampa:
1. **Ajuste** (Optuna, TPE): cada combinación de hiperparámetros se evalúa con dos cortes
   temporales DENTRO del entrenamiento (ene → feb y ene–feb → mar). La prueba no se toca.
2. **Prueba** (abr–may 2018): cada modelo, con sus mejores hiperparámetros y reentrenado con
   ene–mar, se mide una sola vez: AUC con IC 90%, diferencia contra la logística con bootstrap
   pareado, Brier, lift y captura del top 20%.

Modelos: logística, Random Forest, XGBoost (GPU), LightGBM, CatBoost (GPU si está disponible)
y una red neuronal (MLP). Salidas en data/processed/: leads_optuna, leads_metricas,
leads_deciles, leads_importancia. Bitácora por intento en modelos/leads_ajuste.log.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from growth_lab import db, leads
from growth_lab.leads import CATEGORICAS, NUMERICAS, SEMILLA, VARIABLES

RAIZ = Path(__file__).resolve().parents[2]
MODELOS_DIR = RAIZ / "modelos"
CORTES_AJUSTE = ("2018-02-01", "2018-03-01")  # validación temporal dentro del entrenamiento
CAMPEON = "logistica"  # decisión 005: empata con LightGBM en la prueba y es la más simple
# Con ~3.300 filas la GPU es más lenta que la CPU (medido en `medir_dispositivos`): XGBoost 2,6 s vs 0,3 s,
# CatBoost 53 s vs 28 s. La búsqueda corre en CPU; la medición queda en leads_cpu_vs_gpu.parquet.
DISPOSITIVO_XGB = "cpu"
GPU_CATBOOST = False

log = logging.getLogger("leads_modelos")


def _preproceso(escalar: bool = True) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATEGORICAS),
        ("num", StandardScaler() if escalar else "passthrough", NUMERICAS),
    ])


class CatBoostModelo(ClassifierMixin, BaseEstimator):  # el mixin va primero para que sklearn lo reconozca como clasificador
    """CatBoost con categóricas como texto; intenta GPU y cae a CPU si no está disponible."""

    def __init__(self, depth=4, learning_rate=0.05, iterations=400, l2_leaf_reg=3.0, gpu=GPU_CATBOOST):
        self.depth, self.learning_rate, self.iterations, self.l2_leaf_reg, self.gpu = depth, learning_rate, iterations, l2_leaf_reg, gpu

    def _x(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x[VARIABLES].copy()
        for c in CATEGORICAS:
            x[c] = x[c].astype(str)
        return x

    def fit(self, x, y):
        from catboost import CatBoostClassifier

        base = dict(depth=self.depth, learning_rate=self.learning_rate, iterations=self.iterations,
                    l2_leaf_reg=self.l2_leaf_reg, random_seed=SEMILLA, verbose=False, cat_features=CATEGORICAS)
        try:
            self.m_ = CatBoostClassifier(**base, task_type="GPU" if self.gpu else "CPU").fit(self._x(x), y)
            self.dispositivo_ = "GPU" if self.gpu else "CPU"
        except Exception:  # GPU no soportada por esta compilación
            self.m_ = CatBoostClassifier(**base, task_type="CPU").fit(self._x(x), y)
            self.dispositivo_ = "CPU"
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        return self.m_.predict_proba(self._x(x))


def construir(nombre: str, p: dict):
    """Modelo con hiperparámetros `p`; todos reciben el DataFrame con VARIABLES."""
    if nombre == "logistica":
        return make_pipeline(_preproceso(), LogisticRegression(C=p["C"], max_iter=3000))
    if nombre == "random_forest":
        return make_pipeline(_preproceso(escalar=False), RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"], n_jobs=-1, random_state=SEMILLA))
    if nombre == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            device=DISPOSITIVO_XGB, tree_method="hist", enable_categorical=True, max_cat_to_onehot=p["max_cat_to_onehot"],
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
            min_child_weight=p["min_child_weight"], subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
            reg_lambda=p["reg_lambda"], random_state=SEMILLA, eval_metric="auc")
    if nombre == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
            min_child_samples=p["min_child_samples"], subsample=p["subsample"], subsample_freq=1,
            colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], cat_smooth=p["cat_smooth"],
            random_state=SEMILLA, verbose=-1)
    if nombre == "catboost":
        return CatBoostModelo(depth=p["depth"], learning_rate=p["learning_rate"], iterations=p["iterations"],
                              l2_leaf_reg=p["l2_leaf_reg"])
    if nombre == "red_neuronal":
        return make_pipeline(_preproceso(), MLPClassifier(
            hidden_layer_sizes=tuple([p["neuronas"]] * p["capas"]), alpha=p["alpha"],
            learning_rate_init=p["learning_rate_init"], early_stopping=True, validation_fraction=0.2,
            max_iter=500, random_state=SEMILLA))
    raise ValueError(nombre)


def espacio(nombre: str, t: optuna.Trial) -> dict:
    """Espacio de búsqueda de cada modelo."""
    if nombre == "logistica":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True)}
    if nombre == "random_forest":
        return {"n_estimators": t.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": t.suggest_int("max_depth", 3, 12),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 100, log=True),
                "max_features": t.suggest_float("max_features", 0.2, 1.0)}
    if nombre == "xgboost":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": t.suggest_int("max_depth", 2, 6),
                "min_child_weight": t.suggest_float("min_child_weight", 1, 50, log=True),
                "subsample": t.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 0.1, 50, log=True),
                "max_cat_to_onehot": t.suggest_int("max_cat_to_onehot", 1, 16)}
    if nombre == "lightgbm":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": t.suggest_int("num_leaves", 4, 32, log=True),
                "min_child_samples": t.suggest_int("min_child_samples", 10, 200, log=True),
                "subsample": t.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 0.1, 50, log=True),
                "cat_smooth": t.suggest_float("cat_smooth", 1, 100, log=True)}
    if nombre == "catboost":
        return {"depth": t.suggest_int("depth", 2, 8),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "iterations": t.suggest_int("iterations", 100, 800, step=50),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 30, log=True)}
    if nombre == "red_neuronal":
        return {"capas": t.suggest_int("capas", 1, 3), "neuronas": t.suggest_int("neuronas", 8, 128, log=True),
                "alpha": t.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}
    raise ValueError(nombre)


NOMBRES = ["logistica", "random_forest", "xgboost", "lightgbm", "catboost", "red_neuronal"]


def folds(entreno: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Cortes temporales dentro del entrenamiento; las landings frecuentes se recalculan en cada corte."""
    return [leads.particion(entreno, corte) for corte in CORTES_AJUSTE]


def auc_temporal(nombre: str, p: dict, cortes: list[tuple[pd.DataFrame, pd.DataFrame]]) -> float:
    aucs = []
    for tr, va in cortes:
        m = construir(nombre, p).fit(tr[VARIABLES], tr.cerro.astype(int))
        aucs.append(roc_auc_score(va.cerro.astype(int), m.predict_proba(va[VARIABLES])[:, 1]))
    return float(np.mean(aucs))


def buscar(nombre: str, cortes, n_trials: int) -> optuna.Study:
    estudio = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMILLA),
                                  study_name=nombre)

    def objetivo(t: optuna.Trial) -> float:
        inicio = time.perf_counter()
        auc = auc_temporal(nombre, espacio(nombre, t), cortes)
        log.info("%s intento=%d auc=%.4f seg=%.1f params=%s", nombre, t.number, auc, time.perf_counter() - inicio, t.params)
        return auc

    estudio.optimize(objetivo, n_trials=n_trials)
    return estudio


def mejores_parametros() -> dict[str, dict]:
    """Mejores hiperparámetros guardados por la última búsqueda (data/processed/leads_optuna.parquet)."""
    t = pd.read_parquet(db.PROCESSED / "leads_optuna.parquet")
    mejor = t.loc[t.groupby("modelo").auc_temporal.idxmax()]
    return {r.modelo: json.loads(r.params) for r in mejor.itertuples()}


def ajustar_final(nombre: str, entreno: pd.DataFrame):
    return construir(nombre, mejores_parametros()[nombre]).fit(entreno[VARIABLES], entreno.cerro.astype(int))


def medir_dispositivos(entreno: pd.DataFrame, repeticiones: int = 3) -> pd.DataFrame:
    """Segundos de entrenamiento en CPU y GPU con la misma configuración (mediana de `repeticiones`)."""
    from xgboost import XGBClassifier

    x, y = entreno[VARIABLES], entreno.cerro.astype(int)
    casos = {
        ("xgboost", "CPU"): lambda: XGBClassifier(device="cpu", tree_method="hist", enable_categorical=True,
                                                  n_estimators=500, max_depth=4).fit(x, y),
        ("xgboost", "GPU"): lambda: XGBClassifier(device="cuda", tree_method="hist", enable_categorical=True,
                                                  n_estimators=500, max_depth=4).fit(x, y),
        ("catboost", "CPU"): lambda: CatBoostModelo(depth=6, iterations=500, gpu=False).fit(x, y),
        ("catboost", "GPU"): lambda: CatBoostModelo(depth=6, iterations=500, gpu=True).fit(x, y),
    }
    filas = []
    for (modelo, disp), f in casos.items():
        tiempos = []
        for _ in range(repeticiones):
            inicio = time.perf_counter()
            f()
            tiempos.append(time.perf_counter() - inicio)
        filas.append({"modelo": modelo, "dispositivo": disp, "segundos": float(np.median(tiempos)), "filas": len(x)})
    return pd.DataFrame(filas)


def correr(n_trials: int = 40) -> dict[str, pd.DataFrame]:
    MODELOS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(filename=MODELOS_DIR / "leads_ajuste.log", level=logging.INFO,
                        format="%(asctime)s %(message)s", force=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    entreno, prueba = leads.particion(leads.poblacion(db.conectar(db.PROCESSED)))
    cortes = folds(entreno)
    if n_trials > 0:
        medir_dispositivos(entreno).to_parquet(db.PROCESSED / "leads_cpu_vs_gpu.parquet", compression="zstd", index=False)
        buscar_todos(cortes, n_trials)
    intentos = pd.read_parquet(db.PROCESSED / "leads_optuna.parquet").to_dict("records")
    mejores = mejores_parametros()
    return evaluar_final(entreno, prueba, intentos, mejores)


def buscar_todos(cortes, n_trials: int) -> None:
    intentos = []
    for nombre in NOMBRES:
        inicio = time.perf_counter()
        est = buscar(nombre, cortes, n_trials)
        for t in est.trials:
            intentos.append({"modelo": nombre, "intento": t.number, "auc_temporal": t.value, "params": json.dumps(t.params)})
        print(f"{nombre:<14} mejor AUC temporal {est.best_value:.4f} ({time.perf_counter() - inicio:.0f} s)", flush=True)
    pd.DataFrame(intentos).to_parquet(db.PROCESSED / "leads_optuna.parquet", compression="zstd", index=False)


def evaluar_final(entreno: pd.DataFrame, prueba: pd.DataFrame, intentos: list[dict], mejores: dict) -> dict[str, pd.DataFrame]:
    """Prueba final (abr–may): una sola vez por modelo, con los mejores hiperparámetros."""
    yte = prueba.cerro.to_numpy(int)
    preds, metricas, tablas = {}, [], []
    for nombre in NOMBRES:
        inicio = time.perf_counter()
        m = construir(nombre, mejores[nombre]).fit(entreno[VARIABLES], entreno.cerro.astype(int))
        seg = time.perf_counter() - inicio
        p = m.predict_proba(prueba[VARIABLES])[:, 1]
        preds[nombre] = p
        dec = leads.deciles(yte, p)
        lo, hi = leads.ic_auc(yte, p)
        metricas.append({"modelo": nombre, "auc_temporal": max(r["auc_temporal"] for r in intentos if r["modelo"] == nombre),
                         "auc": roc_auc_score(yte, p), "auc_p05": lo, "auc_p95": hi, "brier": brier_score_loss(yte, p),
                         "lift_decil_1": dec.lift.iloc[0], "captura_top_20": dec.captura_acumulada.iloc[1],
                         "segundos_entreno": seg, "dispositivo": getattr(m, "dispositivo_", DISPOSITIVO_XGB.upper() if nombre == "xgboost" else "CPU"),
                         "params": json.dumps(mejores[nombre])})
        tablas.append(dec.assign(modelo=nombre))
    met = pd.DataFrame(metricas)
    for i, r in met.iterrows():
        dif, lo, hi = leads.ic_diferencia_auc(yte, preds[r.modelo], preds["logistica"])
        met.loc[i, ["dif_auc_vs_logistica", "dif_p05", "dif_p95"]] = dif, lo, hi
    met = met.sort_values("auc", ascending=False, ignore_index=True)

    # Importancia por permutación (caída del AUC en prueba) para todos los modelos.
    imp = []
    for nombre in NOMBRES:
        m = construir(nombre, mejores[nombre]).fit(entreno[VARIABLES], entreno.cerro.astype(int))
        r = permutation_importance(m, prueba[VARIABLES], yte, scoring="roc_auc", n_repeats=20, random_state=SEMILLA)
        imp.append(pd.DataFrame({"modelo": nombre, "variable": VARIABLES, "caida_auc": r.importances_mean,
                                 "sd": r.importances_std}))
    salidas = {"leads_metricas": met, "leads_deciles": pd.concat(tablas, ignore_index=True),
               "leads_importancia": pd.concat(imp, ignore_index=True)}
    for nombre, df in salidas.items():
        df.to_parquet(db.PROCESSED / f"{nombre}.parquet", compression="zstd", index=False)
    return salidas


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    s = correr(n)
    cols = ["modelo", "auc_temporal", "auc", "auc_p05", "auc_p95", "dif_auc_vs_logistica", "dif_p05", "dif_p95",
            "brier", "lift_decil_1", "captura_top_20", "segundos_entreno", "dispositivo"]
    print(s["leads_metricas"][cols].round(4).to_string(index=False))
