"""
Etapa 1: Preprocesamiento, Limpieza y Clustering con DBSCAN (2.º Parcial - Minería de Datos)
============================================================================================
Archivo: preprocesameinto.py (respetando el nombre exacto indicado en el PDF del parcial)
Compatible con ejecución directa en Python y copiado por bloques en Google Colab (limpieza.ipynb).

Resumen del pipeline:
1. Carga de Dataset.csv (encoding utf-8-sig para limpiar BOM) y auditoría inicial (EDA).
2. Limpieza estructural: eliminación de columna vacía (Unnamed: 16), 68 duplicados exactos,
   verificación de valores faltantes y validación de consistencia geométrica.
3. Análisis de colinealidad y escalado (StandardScaler) + diagnóstico k-distance (método del codo).
4. Evaluación comparativa de 3 escenarios DBSCAN:
   - Escenario A (1 etapa, 16D StandardScaler): eps=1.50, min_samples=32 -> K=2, 2.97% ruido.
     Evidencia la limitación clásica de DBSCAN ante clústeres de densidades heterogéneas.
   - Escenario B (1 etapa, Std -> Yeo-Johnson -> PCA-3 [91.33% var]): eps=0.52, min_samples=30 -> K=4, 20.09% ruido.
   - Escenario C (2 etapas multi-densidad, 100% DBSCAN — PROPUESTA PRINCIPAL):
       * Paso C1 (global, 16D Std): eps=1.50, min_samples=32 -> elimina 402 puntos de ruido global
         y aísla completo el clúster poco denso de granos gigantes (436 registros).
       * Paso C2 (local, nube principal re-escalada + PCA-3 [90.35% var]): eps=0.60, min_samples=30
         -> separa la zona de alta densidad en 3 clases morfológicas nítidas.
       * Resultado integrado: K=4 clases bien definidas (11,345 registros válidos, 16.23% ruido).
5. Exportación de datasets etiquetados (con las 16 variables originales SIN escalar + columna 'clase',
   para evitar fuga de información al escalar únicamente con train en la Etapa 3) y gráficas diagnósticas.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")

# %% ==========================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
ARCHIVO_ENTRADA = "Dataset.csv"
CARPETA_SALIDA = "salidas_etapa1"
SEMILLA = 42

os.makedirs(CARPETA_SALIDA, exist_ok=True)


# %% ==========================================================================
# 1. CARGA Y AUDITORÍA INICIAL (EDA)
# =============================================================================
def cargar_y_limpiar_estructura(ruta_csv: str) -> pd.DataFrame:
    df_raw = pd.read_csv(ruta_csv, encoding="utf-8-sig")
    print("=" * 72)
    print("1. CARGA Y LIMPIEZA ESTRUCTURAL")
    print("=" * 72)
    print(f"Dimensiones originales : {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas")

    # Eliminar columnas vacías producto de comas finales (ej. 'Unnamed: 16')
    cols_vacias = [c for c in df_raw.columns if "Unnamed" in c or df_raw[c].isna().all()]
    df = df_raw.drop(columns=cols_vacias)
    print(f"Columnas vacías eliminadas: {cols_vacias}")

    # Verificar valores faltantes en las 16 variables reales
    nulos_totales = int(df.isna().sum().sum())
    print(f"Valores faltantes en atributos reales: {nulos_totales}")

    # Eliminar duplicados exactos (evita sesgo de densidad en DBSCAN y data leakage)
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicados exactos eliminados: {n_dup}")
    print(f"Dimensiones tras limpieza estructural: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # Verificación de consistencia geométrica (diccionario de datos)
    err_ar = np.max(np.abs(df["AspectRation"] - df["MajorAxisLength"] / df["MinorAxisLength"]))
    err_ed = np.max(np.abs(df["EquivDiameter"] - np.sqrt(4 * df["Area"] / np.pi)))
    err_sol = np.max(np.abs(df["Solidity"] - df["Area"] / df["ConvexArea"]))
    err_rnd = np.max(np.abs(df["roundness"] - 4 * np.pi * df["Area"] / (df["Perimeter"] ** 2)))
    err_cmp = np.max(np.abs(df["Compactness"] - df["EquivDiameter"] / df["MajorAxisLength"]))
    err_sf3 = np.max(np.abs(df["ShapeFactor3"] - df["Compactness"] ** 2))
    inconsistencias = int((df["Area"] > df["ConvexArea"]).sum() + (df["MinorAxisLength"] > df["MajorAxisLength"]).sum())

    print("\nValidación de consistencia matemática (error absoluto máximo):")
    print(f"  AspectRation = L/l           : {err_ar:.2e}")
    print(f"  EquivDiameter = sqrt(4A/pi)  : {err_ed:.2e}")
    print(f"  Solidity = A/C               : {err_sol:.2e}")
    print(f"  roundness = 4*pi*A/P^2       : {err_rnd:.2e}")
    print(f"  Compactness = Ed/L           : {err_cmp:.2e}")
    print(f"  ShapeFactor3 = Compactness^2 : {err_sf3:.2e}")
    print(f"  Violaciones geométricas (A>C o l>L): {inconsistencias}")

    # Reportar multicolinealidad alta (|r| >= 0.90)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pares_colineales = int((upper >= 0.90).sum().sum())
    print(f"\nPares de variables con |r| >= 0.90: {pares_colineales} (justifica PCA para DBSCAN local)")

    return df


# %% ==========================================================================
# 2. FUNCIONES AUXILIARES DE REINDEXADO Y MÉTRICAS
# =============================================================================
def reindexar_por_tamano(etiquetas: np.ndarray) -> np.ndarray:
    """Reasigna los IDs de clúster válidos (0..K-1) ordenados de mayor a menor tamaño."""
    salida = np.full_like(etiquetas, -1)
    validos = etiquetas != -1
    clases, conteos = np.unique(etiquetas[validos], return_counts=True)
    orden = clases[np.argsort(conteos)[::-1]]
    for nuevo_id, viejo_id in enumerate(orden):
        salida[etiquetas == viejo_id] = nuevo_id
    return salida


def evaluar_clustering(nombre: str, X_ref: np.ndarray, etiquetas: np.ndarray) -> dict:
    """Calcula métricas internas de calidad excluyendo ruido (-1), como indica la teoría."""
    validos = etiquetas != -1
    n_total = len(etiquetas)
    n_validos = int(validos.sum())
    n_ruido = int((~validos).sum())
    ruido_pct = n_ruido / n_total * 100.0
    clases, conteos = np.unique(etiquetas[validos], return_counts=True)

    sil = silhouette_score(X_ref[validos], etiquetas[validos])
    dbi = davies_bouldin_score(X_ref[validos], etiquetas[validos])
    chi = calinski_harabasz_score(X_ref[validos], etiquetas[validos])

    return {
        "Escenario": nombre,
        "K": len(clases),
        "Validos": n_validos,
        "Ruido": n_ruido,
        "Ruido_%": round(ruido_pct, 2),
        "Silhouette_16D": round(sil, 4),
        "Davies_Bouldin_16D": round(dbi, 4),
        "Calinski_Harabasz_16D": round(chi, 1),
        "Tamanos_por_clase": list(conteos),
    }


# %% ==========================================================================
# 3. EJECUCIÓN DE ESCENARIOS DBSCAN
# =============================================================================
def ejecutar_escenarios_dbscan(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("2. ESCALADO Y EVALUACIÓN DE ESCENARIOS DBSCAN")
    print("=" * 72)

    X_std = StandardScaler().fit_transform(df)
    X_pca2_vis = PCA(n_components=2, random_state=SEMILLA).fit_transform(X_std)

    # --- Escenario A: DBSCAN 1 etapa en 16D estandarizado ---
    # min_samples = 2 * d = 32; eps = 1.50 (zona del codo en curva k-distance)
    lab_A = reindexar_por_tamano(DBSCAN(eps=1.50, min_samples=32).fit_predict(X_std))

    # --- Escenario B: DBSCAN 1 etapa tras Yeo-Johnson + PCA-3 ---
    X_yj = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(X_std)
    pca_yj = PCA(n_components=3, random_state=SEMILLA)
    X_yj_pca3 = pca_yj.fit_transform(X_yj)
    print(f"Varianza explicada YJ + PCA-3: {pca_yj.explained_variance_ratio_.sum() * 100:.2f}%")
    lab_B = reindexar_por_tamano(DBSCAN(eps=0.52, min_samples=30).fit_predict(X_yj_pca3))

    # --- Escenario C: DBSCAN Multi-densidad en 2 etapas (Propuesta principal) ---
    # Paso C1: DBSCAN global en 16D Std (elimina 402 ruido global y aísla clúster gigante de baja densidad)
    idx_main = np.where(lab_A == 0)[0]   # Nube principal densa (12,705 granos)
    idx_giant = np.where(lab_A == 1)[0]  # Clúster aislado de granos gigantes (436 granos)

    # Paso C2: Re-escalado de nube principal + PCA-3 + DBSCAN de mayor resolución
    X_main_std = StandardScaler().fit_transform(df.iloc[idx_main])
    pca_main = PCA(n_components=3, random_state=SEMILLA)
    X_main_pca3 = pca_main.fit_transform(X_main_std)
    print(f"Varianza explicada nube principal PCA-3: {pca_main.explained_variance_ratio_.sum() * 100:.2f}%")

    lab_main = DBSCAN(eps=0.60, min_samples=30).fit_predict(X_main_pca3)
    lab_C_raw = np.full(len(df), -1)
    for cid in np.unique(lab_main[lab_main != -1]):
        lab_C_raw[idx_main[lab_main == cid]] = cid
    # Asignar ID temporal único al clúster gigante aislado en el Paso C1
    lab_C_raw[idx_giant] = 99
    lab_C = reindexar_por_tamano(lab_C_raw)

    # Tabla comparativa evaluada en el espacio común 16D estandarizado
    metricas = pd.DataFrame([
        evaluar_clustering("A) 16D Std (1 etapa: eps=1.50, ms=32)", X_std, lab_A),
        evaluar_clustering("B) YJ + PCA-3 (1 etapa: eps=0.52, ms=30)", X_std, lab_B),
        evaluar_clustering("C) Multi-densidad (2 etapas: eps1=1.50, eps2=0.60)", X_std, lab_C),
    ])
    print("\nComparativa de escenarios DBSCAN (métricas en espacio común 16D StandardScaler):")
    print(metricas.to_string(index=False))

    # Guardar tabla de métricas
    metricas.to_csv(os.path.join(CARPETA_SALIDA, "comparativa_metricas_dbscan.csv"), index=False)

    # Perfiles morfológicos del Escenario C (principal)
    df_perfil = df.copy()
    df_perfil["clase"] = lab_C
    perfil_C = (
        df_perfil[df_perfil["clase"] != -1]
        .groupby("clase")[["Area", "Perimeter", "AspectRation", "Eccentricity", "roundness", "Solidity"]]
        .agg(["count", "mean", "std"])
        .round(3)
    )
    print("\nPerfil morfológico de las 4 clases descubiertas (Escenario C):")
    print(perfil_C)

    # Generar gráficas diagnósticas
    generar_graficas(X_std, X_main_pca3, X_pca2_vis, lab_A, lab_B, lab_C)

    return lab_A, lab_B, lab_C, metricas


# %% ==========================================================================
# 4. GRÁFICAS DIAGNÓSTICAS (CURVAS K-DISTANCE + COMPARATIVA PCA 2D)
# =============================================================================
def generar_graficas(X_std, X_main_pca3, X_pca2_vis, lab_A, lab_B, lab_C):
    # Gráfica 1: Curvas k-distance para justificar eps en Paso C1 y Paso C2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    d_c1 = np.sort(NearestNeighbors(n_neighbors=32).fit(X_std).kneighbors(X_std)[0][:, -1])
    axes[0].plot(d_c1, color="steelblue", lw=1.6)
    axes[0].axhline(1.50, color="crimson", ls="--", label="eps = 1.50 (Paso C1 / Escenario A)")
    axes[0].set_title("Curva k-distance Global (16D Std, k=32)")
    axes[0].set_xlabel("Instancias ordenadas")
    axes[0].set_ylabel("Distancia al 32.º vecino")
    axes[0].set_ylim(0, 4.0)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    d_c2 = np.sort(NearestNeighbors(n_neighbors=30).fit(X_main_pca3).kneighbors(X_main_pca3)[0][:, -1])
    axes[1].plot(d_c2, color="darkorange", lw=1.6)
    axes[1].axhline(0.60, color="crimson", ls="--", label="eps = 0.60 (Paso C2 en nube principal)")
    axes[1].set_title("Curva k-distance Local (Nube principal PCA-3, k=30)")
    axes[1].set_xlabel("Instancias ordenadas")
    axes[1].set_ylabel("Distancia al 30.º vecino")
    axes[1].set_ylim(0, 2.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "curvas_k_distance.png"), dpi=150)
    plt.close()

    # Gráfica 2: Comparativa visual en proyección PCA 2D
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    paneles = [
        ("A) DBSCAN 16D Std (1 etapa)\nK=2 | Ruido=2.97%", lab_A),
        ("B) DBSCAN YJ + PCA-3 (1 etapa)\nK=4 | Ruido=20.09%", lab_B),
        ("C) DBSCAN Multi-densidad (2 etapas)\nK=4 | Ruido=16.23%", lab_C),
    ]

    for ax, (titulo, lab) in zip(axes, paneles):
        m_ruido = lab == -1
        ax.scatter(
            X_pca2_vis[m_ruido, 0], X_pca2_vis[m_ruido, 1],
            s=4, alpha=0.28, c="lightgray", label=f"Ruido (-1): {m_ruido.sum()}"
        )
        for cid in sorted(np.unique(lab[lab != -1])):
            m = lab == cid
            ax.scatter(
                X_pca2_vis[m, 0], X_pca2_vis[m, 1],
                s=5, alpha=0.55, c=colores[cid], label=f"Clase {cid}: {m.sum()}"
            )
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.set_xlabel("PC1 (55.6%)")
        ax.set_ylabel("PC2 (26.4%)")
        ax.legend(markerscale=3, fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "comparativa_escenarios_dbscan.png"), dpi=150)
    plt.close()


# %% ==========================================================================
# 5. EXPORTACIÓN DE DATASETS PREPROCESADOS (SIN FUGA DE INFORMACIÓN)
# =============================================================================
def exportar_datasets(df: pd.DataFrame, lab_A: np.ndarray, lab_C: np.ndarray):
    print("\n" + "=" * 72)
    print("3. EXPORTACIÓN DE DATASETS LIMPIOS Y ETIQUETADOS")
    print("=" * 72)

    # Se exportan las 16 variables ORIGINALES (sin escalar) + columna 'clase'.
    # El escalador para la RNA se ajustará exclusivamente en train (Etapa 3).
    df_A = df.copy()
    df_A["clase"] = lab_A
    df_A_limpio = df_A[df_A["clase"] != -1].reset_index(drop=True)

    df_C = df.copy()
    df_C["clase"] = lab_C
    df_C_limpio = df_C[df_C["clase"] != -1].reset_index(drop=True)

    ruta_A = os.path.join(CARPETA_SALIDA, "dataset_preprocesado_2clases_escA.csv")
    ruta_C = os.path.join(CARPETA_SALIDA, "dataset_preprocesado_4clases_escC.csv")
    ruta_principal = "dataset_preprocesado.csv"  # Entrada directa para muestreo.py

    df_A_limpio.to_csv(ruta_A, index=False)
    df_C_limpio.to_csv(ruta_C, index=False)
    df_C_limpio.to_csv(ruta_principal, index=False)

    print(f"Escenario A (baseline 2 clases) : {ruta_A} -> {df_A_limpio.shape}")
    print(f"Escenario C (principal 4 clases): {ruta_C} -> {df_C_limpio.shape}")
    print(f"Dataset activo para muestreo.py : {ruta_principal} -> {df_C_limpio.shape}")
    print("\nDistribución de clases en dataset_preprocesado.csv:")
    dist = df_C_limpio["clase"].value_counts().sort_index()
    for cid, cnt in dist.items():
        print(f"  Clase {cid}: {cnt:5d} registros ({cnt / len(df_C_limpio) * 100:5.2f}%)")


if __name__ == "__main__":
    df_limpio = cargar_y_limpiar_estructura(ARCHIVO_ENTRADA)
    lab_A, lab_B, lab_C, _ = ejecutar_escenarios_dbscan(df_limpio)
    exportar_datasets(df_limpio, lab_A, lab_C)
