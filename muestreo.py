"""
Etapa 2: Muestreo Estratificado, Muestreo Progresivo y Validación de Representatividad
======================================================================================
Archivo: muestreo.py
Entrada: dataset_preprocesado.csv (generado en Etapa 1: 11,345 instancias, 16 atributos + 'clase')
Salidas:
  - train.csv      (70% -> 7,941 registros)
  - validation.csv (15% -> 1,702 registros)
  - test.csv       (15% -> 1,702 registros)
  - salidas_etapa2/muestreo_progresivo_metricas.csv
  - salidas_etapa2/verificacion_estratos.csv
  - salidas_etapa2/verificacion_estadistica.csv
  - salidas_etapa2/pruebas_ks_equivalencia.csv
  - salidas_etapa2/curvas_muestreo_progresivo.png
  - salidas_etapa2/distribucion_muestreo_estratificado.png

Criterios implementados para justificar la suficiencia de los tamaños (70% / 15% / 15%):
  1. Muestreo Progresivo (Incremental) estratificado: mide la estabilización de las medias
     de PC1/PC2 por clase y del error RMS del centroide en el espacio 16D estandarizado
     (promediado en 20 réplicas) conforme crece n. Demuestra que a partir de n ~ 1,600
     las métricas convergen a la meseta de estabilidad -> Val (1,702) y Test (1,702)
     son suficientes sin requerir aumentación sintética.
  2. Muestreo Aleatorio Estratificado sin reemplazo con verificación formal de independencia
     (intersección vacía entre Train, Validation y Test -> cero fuga de información).
  3. Prueba no paramétrica de Kolmogorov-Smirnov (2 muestras) en las 16 variables
     (global y en la Clase 3 minoritaria) + error relativo de media y desviación estándar.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# %% ==========================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
ARCHIVO_ENTRADA = "dataset_preprocesado.csv"
CARPETA_SALIDA = "salidas_etapa2"
SEMILLA = 42

os.makedirs(CARPETA_SALIDA, exist_ok=True)


# %% ==========================================================================
# 1. CARGA DEL DATASET PREPROCESADO (ETAPA 1)
# =============================================================================
def cargar_dataset(ruta_csv: str) -> pd.DataFrame:
    df = pd.read_csv(ruta_csv)
    print("=" * 74)
    print("1. CARGA DE DATASET PREPROCESADO (ETAPA 1)")
    print("=" * 74)
    print(f"Archivo cargado     : {ruta_csv}")
    print(f"Dimensiones totales : {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Valores nulos       : {int(df.isna().sum().sum())}")
    print(f"Clases detectadas   : {sorted(df['clase'].unique().tolist())}")
    return df


# %% ==========================================================================
# 2. MUESTREO PROGRESIVO (INCREMENTAL) PARA JUSTIFICAR TAMAÑO DE MUESTRA
# =============================================================================
def analizar_muestreo_progresivo(df: pd.DataFrame, semilla: int = 42, n_replicas: int = 20):
    print("\n" + "=" * 74)
    print("2. ANÁLISIS DE SUFICIENCIA DE MUESTRA: MUESTREO PROGRESIVO ESTRATIFICADO")
    print("=" * 74)

    cols_num = [c for c in df.columns if c != "clase"]
    clases = sorted(df["clase"].unique())
    n_total = len(df)

    # Estandarización SOLO para diagnóstico de estabilidad geométrica (no altera el dataset exportado)
    X_std = StandardScaler().fit_transform(df[cols_num])
    pca2 = PCA(n_components=2, random_state=semilla)
    X_pca2 = pca2.fit_transform(X_std)

    df_diag = pd.DataFrame(X_std, columns=cols_num)
    df_diag["clase"] = df["clase"].values
    df_diag["PC1"] = X_pca2[:, 0]
    df_diag["PC2"] = X_pca2[:, 1]

    # Centroides reales (referencia de la muestra limpia completa) por clase en 16D estandarizado
    medias_ref = df_diag.groupby("clase")[cols_num].mean()
    proporciones = df_diag["clase"].value_counts(normalize=True).sort_index()

    # 2A) Traza acumulativa única (estilo Ejemplo1_ProgressiveSampling.ipynb) para PC1 y PC2 por clase
    barajado_por_clase = {
        c: df_diag[df_diag["clase"] == c].sample(frac=1.0, random_state=semilla).reset_index(drop=True)
        for c in clases
    }
    tamanos_traza = list(range(200, n_total, 200)) + [n_total]
    traza_pc = []
    for n in tamanos_traza:
        partes = [
            barajado_por_clase[c].iloc[: max(2, min(len(barajado_por_clase[c]), int(round(n * proporciones[c]))))]
            for c in clases
        ]
        sub = pd.concat(partes, ignore_index=True)
        fila = {"n": n}
        for c in clases:
            sc = sub[sub["clase"] == c]
            fila[f"PC1_c{c}"] = float(sc["PC1"].mean())
            fila[f"PC2_c{c}"] = float(sc["PC2"].mean())
        traza_pc.append(fila)
    df_traza = pd.DataFrame(traza_pc)

    # 2B) Estabilidad robusta (promedio de n_replicas semillas) del error RMS del centroide 16D
    tamanos_eval = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1702, 2000, 2269, 3000, 4000, 6000, 7941]
    filas_prog = []
    for n in tamanos_eval:
        errores_por_clase = {c: [] for c in clases}
        for r in range(n_replicas):
            sub, _ = train_test_split(
                df_diag, train_size=n, stratify=df_diag["clase"], random_state=semilla + r
            )
            for c in clases:
                sc = sub[sub["clase"] == c]
                rms = float(np.sqrt(np.mean((sc[cols_num].mean() - medias_ref.loc[c]) ** 2)))
                errores_por_clase[c].append(rms)

        fila = {
            "n_muestra": n,
            "porcentaje_%": round(n / n_total * 100, 2),
            "n_clase3_min": int(round(n * proporciones[clases[-1]])),
        }
        for c in clases:
            fila[f"RMS_c{c}"] = round(float(np.mean(errores_por_clase[c])), 4)
        fila["RMS_max_clase"] = round(max(fila[f"RMS_c{c}"] for c in clases), 4)
        filas_prog.append(fila)

    df_prog = pd.DataFrame(filas_prog)
    print(f"Error RMS promedio del centroide 16D (en unidades de desviación estándar, {n_replicas} réplicas):")
    print(df_prog.to_string(index=False))
    df_prog.to_csv(os.path.join(CARPETA_SALIDA, "muestreo_progresivo_metricas.csv"), index=False)

    # Gráfica combinada: (1) Media PC1 por clase, (2) Media PC2 por clase, (3) Error RMS 16D por clase
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Panel 1: Media de PC1 por clase (igual que el notebook del profesor)
    for c in clases:
        axes[0].plot(df_traza["n"], df_traza[f"PC1_c{c}"], marker="o", ms=3, lw=1.3, color=colores[c], label=f"Clase {c}")
    axes[0].axvline(1702, color="black", ls="--", lw=1.2, label="Val/Test (1,702)")
    axes[0].axvline(7941, color="purple", ls=":", lw=1.4, label="Train (7,941)")
    axes[0].set_title("Estabilidad de Media en PC1 por Clase", fontweight="bold")
    axes[0].set_xlabel("Tamaño de muestra estratificada (n)")
    axes[0].set_ylabel("Media de PC1")
    axes[0].legend(fontsize=8, loc="center right")
    axes[0].grid(alpha=0.3)

    # Panel 2: Media de PC2 por clase
    for c in clases:
        axes[1].plot(df_traza["n"], df_traza[f"PC2_c{c}"], marker="o", ms=3, lw=1.3, color=colores[c], label=f"Clase {c}")
    axes[1].axvline(1702, color="black", ls="--", lw=1.2, label="Val/Test (1,702)")
    axes[1].axvline(7941, color="purple", ls=":", lw=1.4, label="Train (7,941)")
    axes[1].set_title("Estabilidad de Media en PC2 por Clase", fontweight="bold")
    axes[1].set_xlabel("Tamaño de muestra estratificada (n)")
    axes[1].set_ylabel("Media de PC2")
    axes[1].legend(fontsize=8, loc="center right")
    axes[1].grid(alpha=0.3)

    # Panel 3: Convergencia del Error RMS del centroide 16D (20 réplicas)
    for c in clases:
        axes[2].plot(df_prog["n_muestra"], df_prog[f"RMS_c{c}"], marker="s", ms=4, lw=1.6, color=colores[c], label=f"Clase {c}")
    axes[2].axvline(1702, color="black", ls="--", lw=1.2, label="Val/Test = 1,702 (15%)")
    axes[2].axvline(7941, color="purple", ls=":", lw=1.4, label="Train = 7,941 (70%)")
    axes[2].axhline(0.10, color="gray", ls="-.", lw=1.0, label="Tolerancia 0.10 sigma")
    axes[2].set_title(f"Error RMS del Centroide 16D ({n_replicas} réplicas)", fontweight="bold")
    axes[2].set_xlabel("Tamaño de muestra estratificada (n)")
    axes[2].set_ylabel("Error RMS (en unidades sigma)")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "curvas_muestreo_progresivo.png"), dpi=150)
    plt.close()

    return df_prog


# %% ==========================================================================
# 3. PARTICIÓN ESTRATIFICADA (70% TRAIN / 15% VALIDATION / 15% TEST)
# =============================================================================
def dividir_estratificado(df: pd.DataFrame, semilla: int = 42):
    print("\n" + "=" * 74)
    print("3. MUESTREO ALEATORIO ESTRATIFICADO SIN REEMPLAZO (70% / 15% / 15%)")
    print("=" * 74)

    df_trabajo = df.copy()
    df_trabajo["_id_original"] = np.arange(len(df_trabajo))

    # Primer corte: 70% Train vs 30% Temporal (Validation + Test)
    df_train, df_temp = train_test_split(
        df_trabajo,
        test_size=0.30,
        random_state=semilla,
        stratify=df_trabajo["clase"],
    )

    # Segundo corte: 50% de 30% -> 15% Validation y 15% Test
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=semilla,
        stratify=df_temp["clase"],
    )

    # Verificación formal de independencia (intersección vacía -> cero fuga de datos)
    ids_train = set(df_train["_id_original"])
    ids_val = set(df_val["_id_original"])
    ids_test = set(df_test["_id_original"])

    solapamientos = (
        len(ids_train & ids_val)
        + len(ids_train & ids_test)
        + len(ids_val & ids_test)
    )
    assert solapamientos == 0, "Error: existen registros compartidos entre conjuntos."
    print(f"Verificación de independencia (registros compartidos): {solapamientos} (OK)")

    df_train = df_train.drop(columns=["_id_original"]).reset_index(drop=True)
    df_val = df_val.drop(columns=["_id_original"]).reset_index(drop=True)
    df_test = df_test.drop(columns=["_id_original"]).reset_index(drop=True)

    n_total = len(df)
    print(f"  Train      : {len(df_train):5,d} instancias ({len(df_train)/n_total*100:.2f}%)")
    print(f"  Validation : {len(df_val):5,d} instancias ({len(df_val)/n_total*100:.2f}%)")
    print(f"  Test       : {len(df_test):5,d} instancias ({len(df_test)/n_total*100:.2f}%)")

    return df_train, df_val, df_test


# %% ==========================================================================
# 4. VALIDACIÓN DE REPRESENTATIVIDAD (ESTRATOS, MEDIAS Y PRUEBA KS)
# =============================================================================
def validar_representatividad(
    df_total: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
):
    print("\n" + "=" * 74)
    print("4. COMPROBACIÓN DE ESTRATOS POR CLASE")
    print("=" * 74)

    clases = sorted(df_total["clase"].unique())
    cols_num = [c for c in df_total.columns if c != "clase"]

    filas_estratos = []
    for c in clases:
        n_tot = int((df_total["clase"] == c).sum())
        n_tr = int((df_train["clase"] == c).sum())
        n_va = int((df_val["clase"] == c).sum())
        n_te = int((df_test["clase"] == c).sum())
        filas_estratos.append({
            "Clase": c,
            "Total_N": n_tot,
            "Total_%": round(n_tot / len(df_total) * 100, 2),
            "Train_N": n_tr,
            "Train_%": round(n_tr / len(df_train) * 100, 2),
            "Val_N": n_va,
            "Val_%": round(n_va / len(df_val) * 100, 2),
            "Test_N": n_te,
            "Test_%": round(n_te / len(df_test) * 100, 2),
        })

    tabla_estratos = pd.DataFrame(filas_estratos)
    print(tabla_estratos.to_string(index=False))
    tabla_estratos.to_csv(os.path.join(CARPETA_SALIDA, "verificacion_estratos.csv"), index=False)

    # Equivalencia descriptiva en variables morfológicas clave
    print("\n" + "=" * 74)
    print("5. EQUIVALENCIA ESTADÍSTICA (MEDIA ± STD) Y PRUEBA KOLMOGOROV-SMIRNOV")
    print("=" * 74)
    vars_clave = ["Area", "Perimeter", "AspectRation", "roundness", "Solidity"]
    filas_stats = []
    for v in vars_clave:
        filas_stats.append({
            "Variable": v,
            "Original": f"{df_total[v].mean():.3f} ± {df_total[v].std():.3f}",
            "Train (70%)": f"{df_train[v].mean():.3f} ± {df_train[v].std():.3f}",
            "Validation (15%)": f"{df_val[v].mean():.3f} ± {df_val[v].std():.3f}",
            "Test (15%)": f"{df_test[v].mean():.3f} ± {df_test[v].std():.3f}",
        })
    tabla_stats = pd.DataFrame(filas_stats)
    print(tabla_stats.to_string(index=False))
    tabla_stats.to_csv(os.path.join(CARPETA_SALIDA, "verificacion_estadistica.csv"), index=False)

    # Pruebas KS de 2 muestras en las 16 variables (H0: misma distribución)
    filas_ks = []
    for v in cols_num:
        p_tv = ks_2samp(df_train[v], df_val[v]).pvalue
        p_tt = ks_2samp(df_train[v], df_test[v]).pvalue
        p_vt = ks_2samp(df_val[v], df_test[v]).pvalue
        err_val = abs(df_val[v].mean() - df_total[v].mean()) / abs(df_total[v].mean()) * 100
        err_test = abs(df_test[v].mean() - df_total[v].mean()) / abs(df_total[v].mean()) * 100
        filas_ks.append({
            "Variable": v,
            "KS_p_Train_Val": round(p_tv, 4),
            "KS_p_Train_Test": round(p_tt, 4),
            "KS_p_Val_Test": round(p_vt, 4),
            "ErrRel_Val_%": round(err_val, 3),
            "ErrRel_Test_%": round(err_test, 3),
            "Equivalente_alpha05": "Sí" if min(p_tv, p_tt, p_vt) >= 0.05 else "No",
        })
    tabla_ks = pd.DataFrame(filas_ks)
    tabla_ks.to_csv(os.path.join(CARPETA_SALIDA, "pruebas_ks_equivalencia.csv"), index=False)

    n_ok = int((tabla_ks["Equivalente_alpha05"] == "Sí").sum())
    print(f"\nPrueba KS Global (alpha=0.05): {n_ok}/{len(cols_num)} variables mantienen distribución equivalente.")
    print(f"  p-valor mínimo Train vs Val : {tabla_ks['KS_p_Train_Val'].min():.4f}")
    print(f"  p-valor mínimo Train vs Test: {tabla_ks['KS_p_Train_Test'].min():.4f}")
    print(f"  p-valor mínimo Val vs Test  : {tabla_ks['KS_p_Val_Test'].min():.4f}")
    print(f"  Error relativo máximo de media -> Val: {tabla_ks['ErrRel_Val_%'].max():.2f}%, Test: {tabla_ks['ErrRel_Test_%'].max():.2f}%")

    # Verificación específica sobre la clase minoritaria (Clase 3: 436 instancias)
    c_min = clases[-1]
    tr3, va3, te3 = df_train[df_train["clase"] == c_min], df_val[df_val["clase"] == c_min], df_test[df_test["clase"] == c_min]
    p_min_tv = min(ks_2samp(tr3[v], va3[v]).pvalue for v in cols_num)
    p_min_tt = min(ks_2samp(tr3[v], te3[v]).pvalue for v in cols_num)
    p_min_vt = min(ks_2samp(va3[v], te3[v]).pvalue for v in cols_num)
    print(f"\nPrueba KS en Clase minoritaria (Clase {c_min}: Train={len(tr3)}, Val={len(va3)}, Test={len(te3)}):")
    print(f"  p-valor mínimo en las 16 variables -> Train-Val: {p_min_tv:.4f} | Train-Test: {p_min_tt:.4f} | Val-Test: {p_min_vt:.4f}")

    # Gráfica comparativa de proporciones por clase
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(clases))
    ancho = 0.25

    ax.bar(x - ancho, tabla_estratos["Train_%"], width=ancho, label="Train (70%)", color="#1f77b4")
    ax.bar(x, tabla_estratos["Val_%"], width=ancho, label="Validation (15%)", color="#ff7f0e")
    ax.bar(x + ancho, tabla_estratos["Test_%"], width=ancho, label="Test (15%)", color="#2ca02c")

    for i in range(len(clases)):
        ax.text(x[i] - ancho, tabla_estratos["Train_%"][i] + 1.0, f"{tabla_estratos['Train_%'][i]:.1f}%", ha="center", fontsize=8)
        ax.text(x[i], tabla_estratos["Val_%"][i] + 1.0, f"{tabla_estratos['Val_%'][i]:.1f}%", ha="center", fontsize=8)
        ax.text(x[i] + ancho, tabla_estratos["Test_%"][i] + 1.0, f"{tabla_estratos['Test_%'][i]:.1f}%", ha="center", fontsize=8)

    ax.set_title("Verificación de Muestreo Estratificado por Clase (70% / 15% / 15%)", fontweight="bold")
    ax.set_xlabel("Clase Morfológica (descubierta en Etapa 1)")
    ax.set_ylabel("Porcentaje de registros (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Clase {c}" for c in clases])
    ax.set_ylim(0, 78)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "distribucion_muestreo_estratificado.png"), dpi=150)
    plt.close()


# %% ==========================================================================
# 5. EXPORTACIÓN DE CONJUNTOS (SIN ESCALAR PARA EVITAR DATA LEAKAGE)
# =============================================================================
def exportar_particiones(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
):
    print("\n" + "=" * 74)
    print("6. EXPORTACIÓN DE ARCHIVOS PARA ETAPA 3 (SIN FUGA DE INFORMACIÓN)")
    print("=" * 74)

    df_train.to_csv("train.csv", index=False)
    df_val.to_csv("validation.csv", index=False)
    df_test.to_csv("test.csv", index=False)

    print("Archivos exportados en escala física original:")
    print(f"  -> train.csv      ({df_train.shape[0]:,d} filas x {df_train.shape[1]} columnas)")
    print(f"  -> validation.csv ({df_val.shape[0]:,d} filas x {df_val.shape[1]} columnas)")
    print(f"  -> test.csv       ({df_test.shape[0]:,d} filas x {df_test.shape[1]} columnas)")
    print(f"  -> Evidencias guardadas en carpeta: {CARPETA_SALIDA}/")


if __name__ == "__main__":
    df_pre = cargar_dataset(ARCHIVO_ENTRADA)
    analizar_muestreo_progresivo(df_pre, semilla=SEMILLA, n_replicas=20)
    train_df, val_df, test_df = dividir_estratificado(df_pre, semilla=SEMILLA)
    validar_representatividad(df_pre, train_df, val_df, test_df)
    exportar_particiones(train_df, val_df, test_df)
