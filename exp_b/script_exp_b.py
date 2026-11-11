import time
import os
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
def trabajo_cpu_bound(n_iter: int) -> float:
    matriz = np.random.rand(500, 500)
    resultado = 0.0
    for _ in range(n_iter):
        matriz = np.sin(matriz) + np.cos(matriz)
        resultado += np.trace(matriz)
    return resultado
def medir_ejecucion(executor_cls, max_workers: int, num_tareas: int, n_iter: int) -> float:
    inicio = time.perf_counter()
    with executor_cls(max_workers=max_workers) as executor:
        futures = [executor.submit(trabajo_cpu_bound, n_iter) for _ in range(num_tareas)]
        concurrent.futures.wait(futures)
    return time.perf_counter() - inicio
if __name__ == '__main__':
    print("EXPERIMENTO B: MULTITHREADING VS MULTIPROCESSING")
    total_cores = os.cpu_count() or 4
    CANTIDAD_TAREAS = 12
    ITERACIONES = 15
    worker_counts = list(range(1, total_cores + 1))
    print(f"[+] Cores detectados : {total_cores}")
    print(f"[+] Evaluando {CANTIDAD_TAREAS} tareas CPU-bound...\n")
    tiempos_procesos = []
    tiempos_hilos = []
    print(f"{'Workers':<8} | {'Procesos (s)':<14} | {'Hilos (s)':<14} | {'Ganador':<10}")
    print("-" * 52)
    for w in worker_counts:
        t_proc = medir_ejecucion(concurrent.futures.ProcessPoolExecutor, w, CANTIDAD_TAREAS, ITERACIONES)
        t_hilo = medir_ejecucion(concurrent.futures.ThreadPoolExecutor, w, CANTIDAD_TAREAS, ITERACIONES)
        tiempos_procesos.append(t_proc)
        tiempos_hilos.append(t_hilo)
        ganador = "Procesos" if t_proc < t_hilo else "Hilos"
        print(f"{w:<8d} | {t_proc:<14.3f} | {t_hilo:<14.3f} | {ganador:<10}")
    os.makedirs("../evidence", exist_ok=True)
    os.makedirs("../graphics", exist_ok=True)
    ruta_csv = "../evidence/exp_b_resultados.csv"
    with open(ruta_csv, "w", encoding="utf-8") as f:
        f.write("Workers,Tiempo_Procesos_Seg,Tiempo_Hilos_Seg\n")
        for w, tp, th in zip(worker_counts, tiempos_procesos, tiempos_hilos):
            f.write(f"{w},{tp:.4f},{th:.4f}\n")
    print(f"\n[+] Datos guardados en: {ruta_csv}")
    plt.figure(figsize=(9, 5))
    plt.plot(worker_counts, tiempos_procesos, marker='o', color='#2E7D32', linewidth=2.5, label='Multiprocessing (Procesos)')
    plt.plot(worker_counts, tiempos_hilos, marker='s', linestyle='--', color='#C62828', linewidth=2.5, label='Multithreading (Hilos)')
    plt.xlabel('Número de Workers', fontsize=11, fontweight='bold')
    plt.ylabel('Tiempo Total', fontsize=11, fontweight='bold')
    plt.title('Experimento B: Impacto del GIL en Tareas CPU-Bound', fontsize=12, pad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    ruta_png = "../graphics/exp_b_rendimiento.png"
    plt.savefig(ruta_png, dpi=300)
    plt.close()
    print(f"[+] Gráfico guardado: {ruta_png}\n")