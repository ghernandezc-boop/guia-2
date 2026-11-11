import platform
import sys
import psutil

print("=== ESPECIFICACIONES DEL ENTORNO ===")
print(f"Sistema Operativo: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
print(f"Procesador: {platform.processor()}")
print(f"Núcleos Físicos: {psutil.cpu_count(logical=False)}")
print(f"Núcleos Lógicos: {psutil.cpu_count(logical=True)}")
print(f"RAM Total: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
print(f"Versión de Python: {sys.version.split()[0]}")

libs = ["numpy", "pandas", "dask", "matplotlib", "seaborn", "psutil"]
for lib in libs:
    try:
        mod = __import__(lib)
        print(f"Versión {lib}: {getattr(mod, '__version__', 'instalado')}")
    except ImportError:
        print(f"Versión {lib}: NO INSTALADO")
        