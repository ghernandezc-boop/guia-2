import time
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

def generar_dataset_simulado(n_filas: int = 4_000_000) -> pd.DataFrame:
    """Genera un DataFrame sintético con un volumen significativo de datos."""
    np.random.seed(42)
    categorias = ['Alfa', 'Beta', 'Gamma', 'Delta']
    return pd.DataFrame({
        'categoria': np.random.choice(categorias, size=n_filas),
        'valor_a': np.random.randn(n_filas),
        'valor_b': np.random.exponential(scale=2.0, size=n_filas)
    })

def benchmark_pandas(df: pd.DataFrame) -> float:
    """Ejecución secuencial con Pandas."""
    t0 = time.perf_counter()
    _ = df.groupby('categoria').agg({'valor_a': ['mean', 'std'], 'valor_b': 'sum'})
    return time.perf_counter() - t0

def benchmark_dask(df: pd.DataFrame, n_particiones: int) -> float:
    """Ejecución paralela/lazy con Dask distribuyendo en N particiones."""
    t0 = time.perf_counter()
    ddf = dd.from_pandas(df, npartitions=n_particiones)
    operacion = ddf.groupby('categoria').agg({'valor_a': ['mean', 'std'], 'valor_b': 'sum'})
    _ = operacion.compute()  # Ejecuta el grafo de cómputo Dask
    return time.perf_counter() - t0

if __name__ == '__main__':
    print("==================================================")
    print("  EXPERIMENTO C: PANDAS VS DASK DATAFRAMES        ")
    print("==================================================")

    total_cores = os.cpu_count() or 4
    N_FILAS = 3_500_000
    print(f"[+] Generando DataFrame sintético ({N_FILAS:,} filas)...")
    df_datos = generar_dataset_simulado(N_FILAS)
    print("[+] Dataset preparado exitosamente.\n")

    # 1. Medición Pandas Secuencial
    t_pandas = benchmark_pandas(df_datos)
    print(f"Pandas (Secuencial)  : {t_pandas:.4f} seg")

    # 2. Medición Dask con diferentes particiones
    particiones_evaluar = [1, 2, 4, total_cores, total_cores * 2]
    tiempos_dask = []

    print("\n--- Evaluando Dask con distintas particiones ---")
    for part in particiones_evaluar:
        t_dask = benchmark_dask(df_datos, part)
        tiempos_dask.append(t_dask)
        speedup_vs_pandas = t_pandas / t_dask
        print(f"Particiones: {part:02d} | Tiempo: {t_dask:.4f} seg | Speedup vs Pandas: {speedup_vs_pandas:.2f}x")

    # Guardar evidencia CSV
    os.makedirs("../evidence", exist_ok=True)
    os.makedirs("../graphics", exist_ok=True)

    ruta_csv = "../evidence/exp_c_resultados.csv"
    with open(ruta_csv, "w", encoding="utf-8") as f:
        f.write("Motor,Particiones,Tiempo_Seg,Speedup_vs_Pandas\n")
        f.write(f"Pandas,1,{t_pandas:.4f},1.0000\n")
        for p, t in zip(particiones_evaluar, tiempos_dask):
            sp = t_pandas / t
            f.write(f"Dask,{p},{t:.4f},{sp:.4f}\n")
    print(f"\n[+] Evidencia guardada en: {ruta_csv}")

    # Generar Gráfico Comparativo
    plt.figure(figsize=(9, 5))
    labels = ['Pandas'] + [f'Dask ({p}p)' for p in particiones_evaluar]
    tiempos_todos = [t_pandas] + tiempos_dask
    colores = ['#4A90E2'] + ['#50E3C2' if t < t_pandas else '#E74C3C' for t in tiempos_dask]

    bars = plt.bar(labels, tiempos_todos, color=colores, edgecolor='black', alpha=0.85)
    plt.axhline(y=t_pandas, color='#4A90E2', linestyle='--', linewidth=1.5, label='Línea base Pandas')

    plt.xlabel('Motor / Configuración de Particiones', fontsize=11, fontweight='bold')
    plt.ylabel('Tiempo Total de Ejecución (s)', fontsize=11, fontweight='bold')
    plt.title('Experimento C: Rendimiento de Agregaciones (Pandas vs Dask)', fontsize=12, pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    ruta_png = "../graphics/exp_c_rendimiento.png"
    plt.savefig(ruta_png, dpi=300)
    plt.close()
    print(f"[+] Gráfico guardado en: {ruta_png}\n")