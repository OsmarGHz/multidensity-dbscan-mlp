"""
Etapas 3 y 4: Optimización Jerárquica de Hiperparámetros (MLPClassifier) y Evaluación Final
===========================================================================================
Archivo: entrenamiento.py
Entradas (escala física original, generadas en Etapa 2 por muestreo.py):
  - train.csv      (7,941 registros, 16 atributos + 'clase')
  - validation.csv (1,702 registros, 16 atributos + 'clase')
  - test.csv       (1,702 registros, reservado intacto hasta la Etapa 4)

Salidas Etapa 3 (salidas_etapa3/):
  - malla_gruesa_resultados.csv
  - malla_fina_resultados.csv
  - resumen_poda_jerarquica.csv
  - evolucion_epocas_ganador.csv
  - curvas_entrenamiento_ganador.png   (Train Loss vs. Val Loss + F1-Macro Train vs. Val)
  - comparacion_malla_fina.png         (Mejor F1-Macro por solver y profundidad)
  - modelo_mlp_ganador.joblib
  - scaler_train.joblib

Salidas Etapa 4 (salidas_etapa4/):
  - evolucion_experimentos.csv         (Malla Gruesa -> Malla Fina -> Test final)
  - metricas_globales_val_vs_test.csv  (Train vs. Validation vs. Test)
  - reporte_clasificacion_test.csv     (Precision, Recall, F1, Soporte y AUC OvR por clase)
  - matriz_confusion_test.csv
  - matriz_confusion_test.png          (Conteos absolutos y normalizada por fila)
  - curvas_roc_test.png                (Curvas ROC multiclase OvR + zoom superior izquierdo)

Metodología alineada a los criterios del Dr. Iván Olmos Pineda:
  1. Preprocesamiento anti-fuga: StandardScaler ajustado EXCLUSIVAMENTE sobre train.csv.
  2. Heurística topológica del curso: n_in=16, n_out=4 -> rango [min(16,4), 16+4] = [4, 20],
     punto medio = (16 + 4) / 2 = 10 neuronas.
  3. Búsqueda de rejilla jerárquica REAL (Malla Gruesa -> Poda basada en ganadores -> Malla Fina):
     - Respeta estrictamente los hiperparámetros válidos por solver ('sgd', 'adam', 'lbfgs').
     - Explora #neuronas, #capas ocultas (1 vs 2), #epocas (max_iter), LR, Momentum,
       regularización L2 (alpha) y función de activación ('relu', 'tanh').
     - Documenta parámetros propios de MLPClassifier: activation, alpha, tol, n_iter_no_change,
       batch_size, early_stopping=False (usamos validation.csv explícito) y random_state=42.
  4. Desempate inteligente (Navaja de Ockham): prioriza F1-Macro en validación (por desbalance
     de la Clase 3 con 3.84%), mínima brecha |F1_Train - F1_Val|, menor #parámetros y mayor AUC-ROC.
  5. Sin validación cruzada ni variación estadística estimada (confirmado en sesión de dudas).
"""

import os
import time
import warnings
import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    log_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %% ==========================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
CARPETA_ETAPA3 = "salidas_etapa3"
CARPETA_ETAPA4 = "salidas_etapa4"
SEMILLA = 42
N_JOBS = -1  # Aprovecha todos los núcleos disponibles en Colab / local

os.makedirs(CARPETA_ETAPA3, exist_ok=True)
os.makedirs(CARPETA_ETAPA4, exist_ok=True)


# %% ==========================================================================
# 1. CARGA Y PREPROCESAMIENTO ANTI-FUGA (FIT SOLO EN TRAIN)
# =============================================================================
def cargar_y_escalar_datos():
    print("=" * 78)
    print("1. CARGA Y PREPROCESAMIENTO ANTI-FUGA DE INFORMACIÓN")
    print("=" * 78)

    df_train = pd.read_csv("train.csv")
    df_val = pd.read_csv("validation.csv")
    df_test = pd.read_csv("test.csv")

    cols_x = [c for c in df_train.columns if c != "clase"]
    X_train_raw, y_train = df_train[cols_x].values, df_train["clase"].values
    X_val_raw, y_val = df_val[cols_x].values, df_val["clase"].values
    X_test_raw, y_test = df_test[cols_x].values, df_test["clase"].values

    # Ajuste exclusivo en train.csv (cero contaminación desde validation o test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    n_in = X_train.shape[1]
    clases = sorted(np.unique(y_train).tolist())
    n_out = len(clases)

    print(f"Atributos de entrada (n_in)  : {n_in}")
    print(f"Clases de salida (n_out)     : {n_out} -> {clases}")
    print(f"Rango heurístico de neuronas : [min({n_in},{n_out}), {n_in}+{n_out}] = [{min(n_in, n_out)}, {n_in + n_out}]")
    print(f"Punto medio heurístico       : ({n_in} + {n_out}) / 2 = {(n_in + n_out) // 2} neuronas")
    print(f"StandardScaler ajustado SOLO en train ({X_train.shape[0]:,} filas).")
    print(f"Media residual post-escalado -> Train: {X_train.mean():.4f} | Val: {X_val.mean():.4f}")

    return X_train, y_train, X_val, y_val, X_test, y_test, cols_x, clases, scaler


# %% ==========================================================================
# FUNCIONES AUXILIARES DE EVALUACIÓN, CONTEO DE PARÁMETROS Y DESEMPATE
# =============================================================================
def contar_parametros(capas_ocultas, n_in=16, n_out=4) -> int:
    arquitectura = [n_in] + list(capas_ocultas) + [n_out]
    total = 0
    for i in range(len(arquitectura) - 1):
        pesos = arquitectura[i] * arquitectura[i + 1]
        sesgos = arquitectura[i + 1]
        total += pesos + sesgos
    return int(total)


def evaluar_configuracion(params_mlp: dict, X_train, y_train, X_val, y_val):
    t0 = time.perf_counter()
    clf = MLPClassifier(
        **params_mlp,
        early_stopping=False,  # Desactivado: usamos validation.csv explícito sin robar 10% a train
        random_state=SEMILLA,
    )
    clf.fit(X_train, y_train)
    tiempo_seg = time.perf_counter() - t0

    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)
    proba_train = clf.predict_proba(X_train)
    proba_val = clf.predict_proba(X_val)

    f1_macro_tr = f1_score(y_train, pred_train, average="macro")
    f1_macro_va = f1_score(y_val, pred_val, average="macro")
    f1_micro_va = f1_score(y_val, pred_val, average="micro")
    acc_va = accuracy_score(y_val, pred_val)
    auc_ovr_va = roc_auc_score(y_val, proba_val, multi_class="ovr", average="macro")
    loss_tr = log_loss(y_train, proba_train)
    loss_va = log_loss(y_val, proba_val)
    brecha = abs(f1_macro_tr - f1_macro_va)
    n_params = contar_parametros(params_mlp["hidden_layer_sizes"], X_train.shape[1], len(np.unique(y_train)))

    registro = {
        "solver": params_mlp["solver"],
        "hidden_layer_sizes": str(params_mlp["hidden_layer_sizes"]),
        "n_capas_ocultas": len(params_mlp["hidden_layer_sizes"]),
        "n_parametros": n_params,
        "activation": params_mlp["activation"],
        "alpha": params_mlp["alpha"],
        "max_iter": params_mlp["max_iter"],
        "learning_rate_init": params_mlp.get("learning_rate_init", np.nan),
        "learning_rate": params_mlp.get("learning_rate", "N/A"),
        "momentum": params_mlp.get("momentum", np.nan),
        "nesterovs_momentum": params_mlp.get("nesterovs_momentum", "N/A"),
        "batch_size": params_mlp.get("batch_size", "N/A"),
        "tol": params_mlp.get("tol", 1e-4),
        "n_iter_no_change": params_mlp.get("n_iter_no_change", "N/A"),
        "epocas_reales": int(clf.n_iter_),
        "convergio": bool(clf.n_iter_ < params_mlp["max_iter"]),
        "Loss_Train": round(float(loss_tr), 5),
        "Loss_Val": round(float(loss_va), 5),
        "F1_Macro_Train": round(float(f1_macro_tr), 5),
        "F1_Macro_Val": round(float(f1_macro_va), 5),
        "Brecha_Train_Val": round(float(brecha), 5),
        "F1_Micro_Val": round(float(f1_micro_va), 5),
        "Accuracy_Val": round(float(acc_va), 5),
        "AUC_ROC_OvR_Val": round(float(auc_ovr_va), 5),
        "Tiempo_s": round(float(tiempo_seg), 3),
    }
    return registro, params_mlp


def ordenar_con_desempate_inteligente(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    Criterio de desempate inteligente (Navaja de Ockham):
      1. Mayor F1_Macro_Val redondeado a 4 decimales (empates técnicos dentro de 0.0001).
      2. Menor Brecha_Train_Val (mejor generalización / menor riesgo de overfitting).
      3. Menor Loss_Val (calibración probabilística más sólida en validación).
      4. Menor n_parametros (topología más compacta).
      5. Mayor AUC_ROC_OvR_Val.
    """
    df_ord = df_res.copy()
    df_ord["_f1_tier"] = df_ord["F1_Macro_Val"].round(4)
    df_ord = (
        df_ord.sort_values(
            by=["_f1_tier", "Brecha_Train_Val", "Loss_Val", "n_parametros", "AUC_ROC_OvR_Val"],
            ascending=[False, True, True, True, False],
        )
        .drop(columns=["_f1_tier"])
        .reset_index(drop=True)
    )
    return df_ord


# %% ==========================================================================
# 2. ETAPA 3 - FASE 1: MALLA GRUESA (EXPLORACIÓN AMPLIA POR SOLVER)
# =============================================================================
def ejecutar_malla_gruesa(X_train, y_train, X_val, y_val) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("2. ETAPA 3 - FASE 1: BÚSQUEDA EN MALLA GRUESA (EXPLORACIÓN POR SOLVER)")
    print("=" * 78)

    # Topologías base cubriendo extremos y centro del rango heurístico [4, 20]
    topologias_gruesas = [(4,), (10,), (20,)]

    # Rejillas separadas respetando estrictamente los hiperparámetros válidos por solver
    rejilla_gruesa = [
        # --- Solver SGD: usa LR, learning_rate, momentum, nesterovs_momentum, batch_size ---
        {
            "solver": ["sgd"],
            "hidden_layer_sizes": topologias_gruesas,
            "activation": ["relu", "tanh"],
            "learning_rate_init": [0.01, 0.05],
            "learning_rate": ["adaptive"],
            "momentum": [0.85, 0.95],
            "nesterovs_momentum": [True],
            "alpha": [1e-4],
            "max_iter": [100, 300],
            "batch_size": [64],
            "tol": [1e-4],
            "n_iter_no_change": [15],
        },
        # --- Solver ADAM: usa LR, batch_size, alpha, max_iter (ignora momentum) ---
        {
            "solver": ["adam"],
            "hidden_layer_sizes": topologias_gruesas,
            "activation": ["relu", "tanh"],
            "learning_rate_init": [0.001, 0.01],
            "alpha": [1e-4, 1e-2],
            "max_iter": [100, 300],
            "batch_size": [64],
            "tol": [1e-4],
            "n_iter_no_change": [15],
        },
        # --- Solver LBFGS: cuasi-Newton ideal para ~8k filas (ignora LR, momentum, batch_size) ---
        {
            "solver": ["lbfgs"],
            "hidden_layer_sizes": topologias_gruesas,
            "activation": ["relu", "tanh"],
            "alpha": [1e-4, 1e-2, 1e-1],
            "max_iter": [100, 300],
            "tol": [1e-4],
        },
    ]

    combinaciones = list(ParameterGrid(rejilla_gruesa))
    print(f"Total de configuraciones en Malla Gruesa: {len(combinaciones)}")
    t0 = time.perf_counter()

    salidas = Parallel(n_jobs=N_JOBS)(
        delayed(evaluar_configuracion)(params, X_train, y_train, X_val, y_val)
        for params in combinaciones
    )
    registros = [reg for reg, _ in salidas]
    elapsed = time.perf_counter() - t0

    df_gruesa = ordenar_con_desempate_inteligente(pd.DataFrame(registros))
    df_gruesa.to_csv(os.path.join(CARPETA_ETAPA3, "malla_gruesa_resultados.csv"), index=False)

    print(f"Malla Gruesa completada en {elapsed:.1f} s.")
    print("\nMejor configuración por solver en Malla Gruesa:")
    cols_vista = [
        "solver", "hidden_layer_sizes", "activation", "alpha", "max_iter",
        "learning_rate_init", "momentum", "epocas_reales",
        "F1_Macro_Train", "F1_Macro_Val", "Brecha_Train_Val", "Loss_Val", "n_parametros"
    ]
    print(df_gruesa.groupby("solver", as_index=False).first()[cols_vista].to_string(index=False))

    print("\nTop 8 global de la Malla Gruesa (tras desempate inteligente):")
    print(df_gruesa[cols_vista].head(8).to_string(index=False))
    return df_gruesa


# %% ==========================================================================
# 3. ETAPA 3 - FASE 2: PODA JERÁRQUICA REAL Y MALLA FINA (1 VS 2 CAPAS)
# =============================================================================
def construir_rejilla_fina_desde_gruesa(df_gruesa: pd.DataFrame):
    """
    Conecta explícitamente Fase 1 -> Fase 2:
    Poda el espacio de búsqueda a partir de las mejores regiones de df_gruesa,
    acotando neuronas alrededor de la topología ganadora y contrastando 1 vs 2 capas ocultas.
    """
    # Ranking de solvers por su mejor resultado en Malla Gruesa
    mejor_por_solver = (
        df_gruesa.groupby("solver", as_index=False)
        .first()
        .pipe(ordenar_con_desempate_inteligente)
    )
    # Seleccionar los 2 mejores solvers para refinamiento (y siempre mantener comparación entre 2)
    solvers_seleccionados = mejor_por_solver["solver"].head(2).tolist()

    # Detectar centro de neuronas ganador en 1 capa y contrastar 1 vs 2 capas compactas
    top_str = df_gruesa.loc[0, "hidden_layer_sizes"]
    n_ganador = int(top_str.strip("()").split(",")[0])
    if n_ganador <= 6:
        topologias_1capa = [(4,), (6,), (8,)]
        topologias_2capas = [(8, 4), (10, 6)]
    elif n_ganador >= 18:
        topologias_1capa = [(14,), (16,), (20,)]
        topologias_2capas = [(16, 8), (20, 10)]
    else:
        topologias_1capa = [(8,), (10,), (12,)]
        topologias_2capas = [(10, 6), (16, 8)]
    topologias_finas = topologias_1capa + topologias_2capas

    rejilla_fina = []
    resumen_poda = []

    for sol in solvers_seleccionados:
        sub = df_gruesa[df_gruesa["solver"] == sol].reset_index(drop=True)
        fila_top = sub.iloc[0]
        act_top = fila_top["activation"]

        # Conservar solo la activación ganadora del solver (poda real desde Fase 1)
        activaciones = [act_top]

        if sol == "sgd":
            lr_top = float(fila_top["learning_rate_init"])
            mom_top = float(fila_top["momentum"])
            lrs = sorted(set([round(lr_top * 0.5, 4), lr_top]))
            moms = sorted(set([round(max(0.80, mom_top - 0.05), 2), mom_top]))
            bloque = {
                "solver": ["sgd"],
                "hidden_layer_sizes": topologias_finas,
                "activation": activaciones,
                "learning_rate_init": lrs,
                "learning_rate": ["adaptive"],
                "momentum": moms,
                "nesterovs_momentum": [True],
                "alpha": [1e-4, 1e-2],
                "max_iter": [150, 300],
                "batch_size": [64],
                "tol": [1e-4],
                "n_iter_no_change": [12],
            }
            detalle = f"act={activaciones}, lr={lrs}, mom={moms}, alpha=[1e-4,1e-2], max_iter=[150,300]"
        elif sol == "adam":
            lr_top = float(fila_top["learning_rate_init"])
            lrs = sorted(set([round(lr_top * 0.5, 4), lr_top]))
            bloque = {
                "solver": ["adam"],
                "hidden_layer_sizes": topologias_finas,
                "activation": activaciones,
                "learning_rate_init": lrs,
                "alpha": [1e-4, 1e-3, 1e-2],
                "max_iter": [150, 300],
                "batch_size": [64],
                "tol": [1e-4],
                "n_iter_no_change": [12],
            }
            detalle = f"act={activaciones}, lr={lrs}, alpha=[1e-4,1e-3,1e-2], max_iter=[150,300]"
        else:  # lbfgs
            alpha_top = float(fila_top["alpha"])
            alphas = sorted(set([round(alpha_top * 0.5, 5), alpha_top, round(alpha_top * 2.0, 5)]))
            bloque = {
                "solver": ["lbfgs"],
                "hidden_layer_sizes": topologias_finas,
                "activation": activaciones,
                "alpha": alphas,
                "max_iter": [150, 300],
                "tol": [1e-4],
            }
            detalle = f"act={activaciones}, alpha={alphas}, max_iter=[150,300]"

        rejilla_fina.append(bloque)
        resumen_poda.append({
            "Solver_Conservado": sol,
            "Mejor_F1_Val_Gruesa": fila_top["F1_Macro_Val"],
            "Topologias_Finas": str(topologias_finas),
            "Región_Acotada": detalle,
            "Modelos_En_Malla_Fina": len(list(ParameterGrid([bloque]))),
        })

    df_poda = pd.DataFrame(resumen_poda)
    df_poda.to_csv(os.path.join(CARPETA_ETAPA3, "resumen_poda_jerarquica.csv"), index=False)
    return rejilla_fina, df_poda


def ejecutar_malla_fina(df_gruesa: pd.DataFrame, X_train, y_train, X_val, y_val):
    print("\n" + "=" * 78)
    print("3. ETAPA 3 - FASE 2: MALLA FINA (CON PODA DESDE FASE 1 Y 1 vs 2 CAPAS)")
    print("=" * 78)

    rejilla_fina, df_poda = construir_rejilla_fina_desde_gruesa(df_gruesa)
    print("Resumen de poda jerárquica (Fase 1 -> Fase 2):")
    print(df_poda.to_string(index=False))

    combinaciones = list(ParameterGrid(rejilla_fina))
    print(f"\nTotal de configuraciones en Malla Fina (tras poda): {len(combinaciones)}")
    t0 = time.perf_counter()

    salidas = Parallel(n_jobs=N_JOBS)(
        delayed(evaluar_configuracion)(params, X_train, y_train, X_val, y_val)
        for params in combinaciones
    )
    elapsed = time.perf_counter() - t0

    registros = []
    params_lista = []
    for idx, (reg, p_dict) in enumerate(salidas):
        reg["_idx_param"] = idx
        registros.append(reg)
        params_lista.append(p_dict)

    df_fina = ordenar_con_desempate_inteligente(pd.DataFrame(registros))
    idx_ganador = int(df_fina.loc[0, "_idx_param"])
    params_ganador = params_lista[idx_ganador]

    # Re-entrenar el modelo ganador con exactitud determinista en memoria principal
    modelo_ganador = MLPClassifier(
        **params_ganador,
        early_stopping=False,
        random_state=SEMILLA,
    )
    modelo_ganador.fit(X_train, y_train)

    df_fina_export = df_fina.drop(columns=["_idx_param"])
    df_fina_export.to_csv(os.path.join(CARPETA_ETAPA3, "malla_fina_resultados.csv"), index=False)

    print(f"Malla Fina completada en {elapsed:.1f} s.")
    print("\nTop 10 configuraciones de la Malla Fina (tras desempate inteligente):")
    cols_vista = [
        "solver", "hidden_layer_sizes", "activation", "alpha", "max_iter",
        "learning_rate_init", "momentum", "epocas_reales",
        "F1_Macro_Train", "F1_Macro_Val", "Brecha_Train_Val", "Loss_Val", "n_parametros"
    ]
    print(df_fina_export[cols_vista].head(10).to_string(index=False))

    return df_fina_export, params_ganador, modelo_ganador


# %% ==========================================================================
# 4. DIAGNÓSTICO DE CONVERGENCIA POR ÉPOCA (TRAIN LOSS vs VAL LOSS)
# =============================================================================
def diagnosticar_y_exportar_etapa3(
    params_ganador: dict,
    modelo_ganador: MLPClassifier,
    df_fina: pd.DataFrame,
    X_train, y_train, X_val, y_val,
    clases: list,
    scaler: StandardScaler,
):
    print("\n" + "=" * 78)
    print("4. DIAGNÓSTICO DE CONVERGENCIA Y EXPORTACIÓN DEL MODELO GANADOR")
    print("=" * 78)

    fila_top = df_fina.iloc[0]
    print("Hiperparámetros y métricas en validación del modelo ganador:")
    for k, v in fila_top.items():
        print(f"  {k:<22}: {v}")

    # Traza por época SIN el sesgo de warm_start=True con semilla fija repetida
    epocas_reales = int(fila_top["epocas_reales"])
    params_traza = {
        k: v for k, v in params_ganador.items()
        if k not in ("max_iter", "warm_start", "early_stopping", "random_state")
    }

    traza = []
    if params_traza["solver"] == "lbfgs":
        # Para L-BFGS evaluamos iteraciones progresivas hasta la convergencia real
        ep_max = max(epocas_reales, 25)
        epocas_eval = sorted(set(list(range(1, ep_max + 1, max(1, ep_max // 30))) + [ep_max]))
        for ep in epocas_eval:
            m_ep = MLPClassifier(**params_traza, max_iter=ep, early_stopping=False, random_state=SEMILLA)
            m_ep.fit(X_train, y_train)
            p_tr = m_ep.predict_proba(X_train)
            p_va = m_ep.predict_proba(X_val)
            traza.append({
                "epoca": ep,
                "loss_train": float(log_loss(y_train, p_tr)),
                "loss_val": float(log_loss(y_val, p_va)),
                "f1_macro_train": float(f1_score(y_train, m_ep.predict(X_train), average="macro")),
                "f1_macro_val": float(f1_score(y_val, m_ep.predict(X_val), average="macro")),
            })
    else:
        # Para SGD / Adam usamos partial_fit con barajado distinto en cada época (sin sesgo)
        ep_max = max(epocas_reales, 40)
        paso = max(1, ep_max // 40)
        epocas_eval = set(list(range(1, ep_max + 1, paso)) + [ep_max])
        clases_arr = np.array(clases)
        m_ep = MLPClassifier(**params_traza, max_iter=1, early_stopping=False, random_state=SEMILLA)
        rng = np.random.RandomState(SEMILLA)
        for ep in range(1, ep_max + 1):
            perm = rng.permutation(len(X_train))
            m_ep.partial_fit(X_train[perm], y_train[perm], classes=clases_arr)
            if ep in epocas_eval:
                p_tr = m_ep.predict_proba(X_train)
                p_va = m_ep.predict_proba(X_val)
                traza.append({
                    "epoca": ep,
                    "loss_train": float(log_loss(y_train, p_tr)),
                    "loss_val": float(log_loss(y_val, p_va)),
                    "f1_macro_train": float(f1_score(y_train, m_ep.predict(X_train), average="macro")),
                    "f1_macro_val": float(f1_score(y_val, m_ep.predict(X_val), average="macro")),
                })

    df_traza = pd.DataFrame(traza)
    df_traza.to_csv(os.path.join(CARPETA_ETAPA3, "evolucion_epocas_ganador.csv"), index=False)

    # Gráfica 1: Curvas de entrenamiento (Train Loss vs Val Loss + F1-Macro Train vs Val)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    axes[0].plot(df_traza["epoca"], df_traza["loss_train"], label="Train Loss (70%)", color="#1f77b4", lw=2)
    axes[0].plot(df_traza["epoca"], df_traza["loss_val"], label="Validation Loss (15%)", color="#ff7f0e", lw=2, ls="--")
    axes[0].set_title("Curva de Pérdida (Cross-Entropy): Train vs Validation", fontweight="bold")
    axes[0].set_xlabel("Época / Iteración")
    axes[0].set_ylabel("Log-Loss (Entropía Cruzada)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df_traza["epoca"], df_traza["f1_macro_train"], label="F1-Macro Train (70%)", color="#1f77b4", lw=2)
    axes[1].plot(df_traza["epoca"], df_traza["f1_macro_val"], label="F1-Macro Validation (15%)", color="#ff7f0e", lw=2, ls="--")
    axes[1].set_title("Generalización por Época: F1-Macro (Train vs Validation)", fontweight="bold")
    axes[1].set_xlabel("Época / Iteración")
    axes[1].set_ylabel("F1-Macro")
    y_min = max(0.75, min(df_traza["f1_macro_val"].min(), df_traza["f1_macro_train"].min()) - 0.02)
    axes[1].set_ylim(y_min, 1.003)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_ETAPA3, "curvas_entrenamiento_ganador.png"), dpi=150)
    plt.close()

    # Gráfica 2: Comparación de mejores modelos por solver y profundidad en Malla Fina
    resumen_solver = (
        df_fina.groupby(["solver", "n_capas_ocultas"], as_index=False)
        .first()
        .pipe(ordenar_con_desempate_inteligente)
    )
    etiquetas = [
        f"{r['solver'].upper()}\n{r['hidden_layer_sizes']} ({r['n_capas_ocultas']} capa{'s' if r['n_capas_ocultas'] > 1 else ''})"
        for _, r in resumen_solver.iterrows()
    ]
    x = np.arange(len(etiquetas))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - ancho / 2, resumen_solver["F1_Macro_Train"], width=ancho, label="F1-Macro Train", color="#1f77b4")
    ax.bar(x + ancho / 2, resumen_solver["F1_Macro_Val"], width=ancho, label="F1-Macro Validation", color="#2ca02c")
    for i in range(len(etiquetas)):
        ax.text(x[i] - ancho / 2, resumen_solver["F1_Macro_Train"].iloc[i] + 0.0003,
                f"{resumen_solver['F1_Macro_Train'].iloc[i]:.4f}", ha="center", fontsize=8)
        ax.text(x[i] + ancho / 2, resumen_solver["F1_Macro_Val"].iloc[i] + 0.0003,
                f"{resumen_solver['F1_Macro_Val'].iloc[i]:.4f}", ha="center", fontsize=8)

    ax.set_title("Mejor F1-Macro por Solver y Profundidad (Malla Fina)", fontweight="bold")
    ax.set_ylabel("F1-Macro")
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylim(max(0.95, resumen_solver["F1_Macro_Val"].min() - 0.004), 1.003)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_ETAPA3, "comparacion_malla_fina.png"), dpi=150)
    plt.close()

    # Exportar modelo ganador y escalador (nombres alineados al encabezado)
    joblib.dump(modelo_ganador, os.path.join(CARPETA_ETAPA3, "modelo_mlp_ganador.joblib"))
    joblib.dump(scaler, os.path.join(CARPETA_ETAPA3, "scaler_train.joblib"))
    print(f"\nArtefactos de Etapa 3 guardados en '{CARPETA_ETAPA3}/'.")


# %% ==========================================================================
# 5. ETAPA 4: EVALUACIÓN FINAL SOBRE TEST.CSV (MATRIZ DE CONFUSIÓN Y ROC/AUC)
# =============================================================================
def evaluar_etapa4_en_test(
    modelo_ganador: MLPClassifier,
    df_gruesa: pd.DataFrame,
    df_fina: pd.DataFrame,
    X_train, y_train, X_val, y_val, X_test, y_test,
    clases: list,
):
    print("\n" + "=" * 78)
    print("5. ETAPA 4: EVALUACIÓN FINAL DEL MODELO GANADOR SOBRE TEST.CSV (15%)")
    print("=" * 78)

    def resumen_split(nombre: str, X_s, y_s) -> dict:
        pred = modelo_ganador.predict(X_s)
        proba = modelo_ganador.predict_proba(X_s)
        return {
            "Particion": nombre,
            "N_Instancias": int(len(y_s)),
            "F1_Macro": round(float(f1_score(y_s, pred, average="macro")), 5),
            "F1_Micro": round(float(f1_score(y_s, pred, average="micro")), 5),
            "Accuracy": round(float(accuracy_score(y_s, pred)), 5),
            "Precision_Macro": round(float(precision_score(y_s, pred, average="macro")), 5),
            "Recall_Macro": round(float(recall_score(y_s, pred, average="macro")), 5),
            "AUC_ROC_OvR_Macro": round(float(roc_auc_score(y_s, proba, multi_class="ovr", average="macro")), 5),
            "AUC_ROC_OvR_Micro": round(float(roc_auc_score(y_s, proba, multi_class="ovr", average="micro")), 5),
            "Log_Loss": round(float(log_loss(y_s, proba)), 5),
        }

    df_splits = pd.DataFrame([
        resumen_split("Train (70%)", X_train, y_train),
        resumen_split("Validation (15%)", X_val, y_val),
        resumen_split("Test (15%)", X_test, y_test),
    ])
    df_splits.to_csv(os.path.join(CARPETA_ETAPA4, "metricas_globales_val_vs_test.csv"), index=False)
    print("Comparativa global del modelo ganador (Train vs. Validation vs. Test):")
    print(df_splits.to_string(index=False))

    # Tabla resumen de evolución de experimentos (Malla Gruesa -> Malla Fina -> Test final)
    top_g = df_gruesa.iloc[0]
    top_f = df_fina.iloc[0]
    fila_test = df_splits.iloc[2]
    df_evol = pd.DataFrame([
        {
            "Fase": "1. Malla Gruesa (mejor en Val)",
            "Solver": top_g["solver"],
            "Arquitectura": top_g["hidden_layer_sizes"],
            "Activacion": top_g["activation"],
            "Alpha": top_g["alpha"],
            "Parametros": int(top_g["n_parametros"]),
            "F1_Macro": top_g["F1_Macro_Val"],
            "F1_Micro": top_g["F1_Micro_Val"],
            "AUC_ROC_OvR": top_g["AUC_ROC_OvR_Val"],
            "Log_Loss": top_g["Loss_Val"],
        },
        {
            "Fase": "2. Malla Fina (ganador en Val)",
            "Solver": top_f["solver"],
            "Arquitectura": top_f["hidden_layer_sizes"],
            "Activacion": top_f["activation"],
            "Alpha": top_f["alpha"],
            "Parametros": int(top_f["n_parametros"]),
            "F1_Macro": top_f["F1_Macro_Val"],
            "F1_Micro": top_f["F1_Micro_Val"],
            "AUC_ROC_OvR": top_f["AUC_ROC_OvR_Val"],
            "Log_Loss": top_f["Loss_Val"],
        },
        {
            "Fase": "3. Evaluación Final (Test 15%)",
            "Solver": top_f["solver"],
            "Arquitectura": top_f["hidden_layer_sizes"],
            "Activacion": top_f["activation"],
            "Alpha": top_f["alpha"],
            "Parametros": int(top_f["n_parametros"]),
            "F1_Macro": fila_test["F1_Macro"],
            "F1_Micro": fila_test["F1_Micro"],
            "AUC_ROC_OvR": fila_test["AUC_ROC_OvR_Macro"],
            "Log_Loss": fila_test["Log_Loss"],
        },
    ])
    df_evol.to_csv(os.path.join(CARPETA_ETAPA4, "evolucion_experimentos.csv"), index=False)

    # Predicciones sobre test.csv
    y_pred_test = modelo_ganador.predict(X_test)
    y_proba_test = modelo_ganador.predict_proba(X_test)

    # Reporte por clase + AUC-ROC OvR por clase
    y_test_bin = label_binarize(y_test, classes=clases)
    rep_dict = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    filas_rep = []
    for idx_c, c in enumerate(clases):
        fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, idx_c], y_proba_test[:, idx_c])
        auc_c = auc(fpr_c, tpr_c)
        d_c = rep_dict[str(c)]
        filas_rep.append({
            "Clase": f"Clase {c}",
            "Precision": round(float(d_c["precision"]), 5),
            "Recall": round(float(d_c["recall"]), 5),
            "F1_Score": round(float(d_c["f1-score"]), 5),
            "AUC_ROC_OvR": round(float(auc_c), 5),
            "Soporte": int(d_c["support"]),
        })
    for clave, etiqueta in [("macro avg", "Macro Avg"), ("weighted avg", "Weighted Avg")]:
        d_a = rep_dict[clave]
        filas_rep.append({
            "Clase": etiqueta,
            "Precision": round(float(d_a["precision"]), 5),
            "Recall": round(float(d_a["recall"]), 5),
            "F1_Score": round(float(d_a["f1-score"]), 5),
            "AUC_ROC_OvR": round(float(roc_auc_score(
                y_test, y_proba_test, multi_class="ovr",
                average="macro" if clave == "macro avg" else "weighted"
            )), 5),
            "Soporte": int(d_a["support"]),
        })

    df_rep = pd.DataFrame(filas_rep)
    df_rep.to_csv(os.path.join(CARPETA_ETAPA4, "reporte_clasificacion_test.csv"), index=False)
    print("\nReporte de clasificación por clase en Test:")
    print(df_rep.to_string(index=False))

    # Matriz de confusión (conteos y normalizada por fila)
    cm = confusion_matrix(y_test, y_pred_test, labels=clases)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
    df_cm = pd.DataFrame(
        cm,
        index=[f"Real_C{c}" for c in clases],
        columns=[f"Pred_C{c}" for c in clases],
    )
    df_cm.to_csv(os.path.join(CARPETA_ETAPA4, "matriz_confusion_test.csv"))
    print("\nMatriz de confusión en Test (conteos):")
    print(df_cm)
    errores_test = int((y_test != y_pred_test).sum())
    print(f"Errores totales en Test: {errores_test} de {len(y_test)} ({errores_test/len(y_test)*100:.2f}%)")

    # Gráfica 1 Etapa 4: Matriz de confusión (conteos + normalizada)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    nombres_clases = [f"Clase {c}" for c in clases]

    for ax, matriz, titulo, fmt in [
        (axes[0], cm, "Matriz de Confusión en Test (Conteos)", "d"),
        (axes[1], cm_norm, "Matriz de Confusión en Test (% por Clase Real)", ".2f"),
    ]:
        im = ax.imshow(matriz, interpolation="nearest", cmap="Blues")
        ax.set_title(titulo, fontweight="bold")
        ax.set_xticks(np.arange(len(clases)))
        ax.set_yticks(np.arange(len(clases)))
        ax.set_xticklabels(nombres_clases)
        ax.set_yticklabels(nombres_clases)
        ax.set_xlabel("Clase Predicha")
        ax.set_ylabel("Clase Real")
        umbral = matriz.max() / 2.0
        for i in range(len(clases)):
            for j in range(len(clases)):
                val = matriz[i, j]
                txt = f"{val:d}" if fmt == "d" else f"{val:.2f}%"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val > umbral else "black", fontweight="bold", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_ETAPA4, "matriz_confusion_test.png"), dpi=150)
    plt.close()

    # Gráfica 2 Etapa 4: Curvas ROC multiclase One-vs-Rest (completa + zoom esquina superior izq.)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba_test.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    # Macro-average ROC
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for idx_c, c in enumerate(clases):
        fpr_dict[c], tpr_dict[c], _ = roc_curve(y_test_bin[:, idx_c], y_proba_test[:, idx_c])
        auc_dict[c] = auc(fpr_dict[c], tpr_dict[c])

    all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in clases]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in clases:
        mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
    mean_tpr /= len(clases)
    auc_macro = auc(all_fpr, mean_tpr)

    for ax_idx, ax in enumerate(axes):
        for idx_c, c in enumerate(clases):
            ax.plot(
                fpr_dict[c], tpr_dict[c], color=colores[idx_c], lw=1.8,
                label=f"Clase {c} (AUC = {auc_dict[c]:.5f})"
            )
        ax.plot(fpr_micro, tpr_micro, color="navy", ls=":", lw=2.0,
                label=f"Micro-promedio (AUC = {auc_micro:.5f})")
        ax.plot(all_fpr, mean_tpr, color="black", ls="--", lw=1.8,
                label=f"Macro-promedio (AUC = {auc_macro:.5f})")
        ax.plot([0, 1], [0, 1], color="gray", ls="-.", lw=1.0, label="Azar (AUC = 0.50)")
        ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
        ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.set_title("Curvas ROC Multiclase OvR — Vista Completa (Test)", fontweight="bold")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.legend(loc="lower right", fontsize=8)
        else:
            ax.set_title("Zoom Esquina Superior Izquierda (FPR ≤ 0.05)", fontweight="bold")
            ax.set_xlim(-0.002, 0.05)
            ax.set_ylim(0.95, 1.003)
            ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_ETAPA4, "curvas_roc_test.png"), dpi=150)
    plt.close()

    print(f"\nEvidencias de Etapa 4 guardadas en '{CARPETA_ETAPA4}/'.")


if __name__ == "__main__":
    X_tr, y_tr, X_va, y_va, X_te, y_te, cols, clases_lista, sc = cargar_y_escalar_datos()
    df_gruesa_res = ejecutar_malla_gruesa(X_tr, y_tr, X_va, y_va)
    df_fina_res, mejor_params, mejor_modelo = ejecutar_malla_fina(df_gruesa_res, X_tr, y_tr, X_va, y_va)
    diagnosticar_y_exportar_etapa3(
        mejor_params, mejor_modelo, df_fina_res,
        X_tr, y_tr, X_va, y_va, clases_lista, sc
    )
    evaluar_etapa4_en_test(
        mejor_modelo, df_gruesa_res, df_fina_res,
        X_tr, y_tr, X_va, y_va, X_te, y_te, clases_lista
    )
