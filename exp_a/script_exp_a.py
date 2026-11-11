import time
import os
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
def kernel_procesamiento(dimension: int) -> float:
    X = np.random.uniform(0.1, 1.0, size=(dimension, dimension))
    Y = np.random.uniform(0.1, 1.0, size=(dimension, dimension))
    matriz_resultante = np.sin(X) @ np.cos(Y)
    return float(np.linalg.norm(matriz_resultante))
def ejecutar_benchmark(num_tareas: int, dim_matriz: int, max_workers: int) -> float:
    tiempo_inicio = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(kernel_procesamiento, dim_matriz) for _ in range(num_tareas)]
        concurrent.futures.wait(futures)
    tiempo_fin = time.perf_counter()
    return tiempo_fin - tiempo_inicio
def generar_reporte_grafico(workers_list: list, tiempos: list, speedups: list, ruta_salida: str):
    fig, ax_tiempo = plt.subplots(figsize=(9, 5))
    color_tiempo = '#007ACC'
    ax_tiempo.set_xlabel('Número de Procesos Activos', fontsize=11, fontweight='bold')
    ax_tiempo.set_ylabel('Tiempo de Ejecución', color=color_tiempo, fontsize=11, fontweight='bold')
    ax_tiempo.plot(workers_list, tiempos, marker='o', linewidth=2.5, color=color_tiempo, label='Tiempo (s)')
    ax_tiempo.tick_params(axis='y', labelcolor=color_tiempo)
    ax_tiempo.grid(True, linestyle='--', alpha=0.5)
    ax_speedup = ax_tiempo.twinx()
    color_speedup = '#D9534F'
    ax_speedup.set_ylabel('Aceleración (Speedup X)', color=color_speedup, fontsize=11, fontweight='bold')
    ax_speedup.plot(workers_list, speedups, marker='s', linestyle='-.', linewidth=2.5, color=color_speedup, label='Speedup')
    ax_speedup.tick_params(axis='y', labelcolor=color_speedup)
    plt.title('Experimento A: Evaluación de Paralelismo Local (Multiprocessing)', fontsize=12, pad=15)
    fig.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
if __name__ == '__main__':
    print("BENCHMARK EXPERIMENTO A: PARALELISMO LOCAL")
    total_cores = os.cpu_count() or 4
    CANTIDAD_TAREAS = 16
    TAMANO_MATRIZ = 650
    print(f"[+] Cores lógicos detectados : {total_cores}")
    print(f"[+] Configuración             : {CANTIDAD_TAREAS} tareas de {TAMANO_MATRIZ}x{TAMANO_MATRIZ}\n")
    lista_workers = list(range(1, total_cores + 1))
    tiempos_ejecucion = []
    lista_speedups = []
    for w in lista_workers:
        duracion = ejecutar_benchmark(CANTIDAD_TAREAS, TAMANO_MATRIZ, w)
        tiempos_ejecucion.append(duracion)
        t_base = tiempos_ejecucion[0]
        speedup = t_base / duracion
        lista_speedups.append(speedup)
        print(f"Workers: {w:02d} | Tiempo: {duracion:6.2f} seg | Speedup: {speedup:5.2f}x")
    os.makedirs("../evidence", exist_ok=True)
    os.makedirs("../graphics", exist_ok=True)
    ruta_csv = "../evidence/exp_a_resultados.csv"
    with open(ruta_csv, "w", encoding="utf-8") as f:
        f.write("Workers,Tiempo_Seg,Speedup\n")
        for w, t, sp in zip(lista_workers, tiempos_ejecucion, lista_speedups):
            f.write(f"{w},{t:.4f},{sp:.4f}\n")
    print(f"\n[+] Datos guardados en: {ruta_csv}")
    ruta_png = "../graphics/exp_a_rendimiento.png"
    generar_reporte_grafico(lista_workers, tiempos_ejecucion, lista_speedups, ruta_png)
    print(f"[+] Gráfico guardado en: {ruta_png}\n")