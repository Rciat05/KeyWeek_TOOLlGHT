"""
entrenar.py - Entrenamiento del detector YOLO (Ultralytics) en GPU.

Antes de entrenar ejecuta el diagnóstico real de GPU. Si CUDA no funciona, se detiene
y explica cómo arreglarlo (salvo que pidas la CPU con --dispositivo cpu).

Ejemplos:
    python scripts/entrenar.py                          (usa los valores de config.yaml)
    python scripts/entrenar.py --modelo yolo11s.pt --batch 4 --imgsz 512
    python scripts/entrenar.py --epocas 5 --nombre prueba_rapida
    python scripts/entrenar.py --modelo weights/entrenamiento_v1_best.pt --aumentos reforzado --epocas 60 --nombre entrenamiento_v2
    python scripts/entrenar.py --reanudar auto          (continúa el último entrenamiento)
    python scripts/entrenar.py --reanudar runs/entrenamientos/<nombre>/weights/last.pt

Cada entrenamiento va a una carpeta nueva (nunca sobrescribe anteriores) y se guardan:
  weights/best.pt y last.pt, results.csv (métricas por época), gráficas (results.png,
  curvas PR/F1, matriz de confusión), config_usada.yaml y los resultados de validación.
Al terminar, copia best.pt a weights/<nombre>_best.pt (fuera de las carpetas de datos).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import yaml
except ImportError:
    sys.exit("[ERROR] Falta pyyaml. Ejecuta: pip install -r requirements.txt")

from verificar_gpu import MENSAJE_INSTALACION, diagnostico_gpu  # noqa: E402

CONSEJO_MEMORIA = """
--------------------------------------------------------------------
SIN MEMORIA DE GPU (CUDA out of memory). Prueba, en este orden:
  1) Cierra programas que usen la GPU (juegos, Epic Games, navegador con aceleración).
  2) Reduce el batch:        --batch 4   (o --batch 2)
  3) Reduce la resolución:   --imgsz 512 (o --imgsz 416)
  4) Reduce workers:         --workers 0
  5) Usa un modelo menor:    --modelo yolo11n.pt
Ejemplo:
  python scripts/entrenar.py --batch 4 --imgsz 512 --workers 0
--------------------------------------------------------------------
"""

DEFECTOS = {
    "modelo": "yolo11n.pt", "epocas": 100, "imgsz": 640, "batch": 8, "lr0": 0.01,
    "workers": 2, "patience": 30, "amp": True, "dispositivo": "auto", "optimizer": "auto", "aumentos": "normal",
}

# Perfil de aumentos de datos más agresivo: simula cambios de luz, tamaño y ángulo de otra
# cámara. Solo usa argumentos estándar de Ultralytics. "normal" = valores por defecto.
AUMENTOS_REFORZADOS = {
    "hsv_h": 0.02, "hsv_s": 0.8, "hsv_v": 0.5,     # tono, saturación y brillo
    "degrees": 10.0, "translate": 0.15, "scale": 0.6, "shear": 3.0, "perspective": 0.0005,
    "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.1, "erasing": 0.4,
}


def cargar_config(ruta: Path) -> dict:
    if not ruta.exists():
        sys.exit(f"[ERROR] No existe {ruta}")
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ultimo_checkpoint(base: Path) -> Path | None:
    candidatos = list(base.glob("*/weights/last.pt"))
    return max(candidatos, key=lambda p: p.stat().st_mtime) if candidatos else None


def comprobar_dataset(data_yaml: Path, clases: list[str]) -> None:
    if not data_yaml.exists():
        sys.exit(f"[ERROR] No existe {data_yaml}. Ejecuta antes preparar_datasets.py y dividir_datasets.py.")
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    nombres = d.get("names", {})
    n = len(nombres)
    if n != len(clases):
        sys.exit(f"[ERROR] {data_yaml} tiene {n} clases pero config.yaml define {len(clases)}. "
                 f"Vuelve a ejecutar dividir_datasets.py --rehacer.")
    raiz = Path(d["path"])
    for s in ("train", "val"):
        carpeta = raiz / d[s]
        imgs = [p for p in carpeta.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        if not imgs:
            sys.exit(f"[ERROR] No hay imágenes en {carpeta}.")
        print(f"  {s:<6}: {len(imgs)} imágenes")


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento del detector YOLO")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--modelo", help="p. ej. yolo11n.pt, yolo11s.pt, yolo11m.pt")
    ap.add_argument("--epocas", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int, help="-1 = automático según la VRAM")
    ap.add_argument("--lr", type=float, help="learning rate inicial (lr0)")
    ap.add_argument("--dispositivo", help="auto, cpu, 0, 1, ...")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--semilla", type=int)
    ap.add_argument("--nombre", help="nombre de la carpeta del entrenamiento")
    ap.add_argument("--aumentos", choices=["normal", "reforzado"],
                    help="perfil de aumento de datos (por defecto, el de config.yaml o 'normal')")
    ap.add_argument("--sin-amp", action="store_true", help="desactiva la precisión mixta")
    ap.add_argument("--reanudar", help="'auto' o ruta a un last.pt")
    ap.add_argument("--data", help="ruta a data.yaml (por defecto dataset_yolo/data.yaml)")
    args = ap.parse_args()

    cfg = cargar_config(Path(r"C:/Ever/Ciclo 4/detector_herramientas/scripts/config.yaml"))
    ent = {**DEFECTOS, **cfg.get("entrenamiento", {})}
    proy = cfg.get("proyecto", {})
    clases = cfg.get("clases", [])
    if not clases:
        sys.exit("[ERROR] config.yaml no tiene 'clases'.")

    modelo = args.modelo or ent["modelo"]
    epocas = args.epocas or ent["epocas"]
    imgsz = args.imgsz or ent["imgsz"]
    batch = args.batch if args.batch is not None else ent["batch"]
    lr0 = args.lr or ent["lr0"]
    workers = args.workers if args.workers is not None else ent["workers"]
    semilla = args.semilla if args.semilla is not None else proy.get("semilla", 42)
    dispositivo = args.dispositivo or ent["dispositivo"]
    amp = False if args.sin_amp else bool(ent["amp"])

    base_runs = Path(proy.get("directorio_resultados", "runs")) / "entrenamientos"
    dir_pesos = Path(proy.get("directorio_pesos", "weights"))
    dir_pesos.mkdir(parents=True, exist_ok=True)
    base_runs.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(args.data or Path(cfg.get("division", {}).get("directorio_salida", "dataset_yolo")) / "data.yaml")

    # 1) Diagnóstico de GPU (obligatorio antes de entrenar)
    print("Comprobando la GPU antes de entrenar...\n")
    gpu_ok = diagnostico_gpu(dispositivo if dispositivo != "cpu" else "auto", verboso=True) if dispositivo != "cpu" else False
    if dispositivo == "cpu":
        print("[AVISO] Entrenarás en CPU por petición expresa. Será MUY lento.")
        dev = "cpu"
    else:
        if not gpu_ok:
            print("\n[ERROR] La GPU no está operativa; no se inicia el entrenamiento.")
            print("        Arregla CUDA con las instrucciones de arriba, o usa --dispositivo cpu si quieres CPU.")
            return 1
        from verificar_gpu import seleccionar_dispositivo
        dev_str, _ = seleccionar_dispositivo(dispositivo)
        dev = int(dev_str.split(":")[1])

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"[ERROR] Falta {e.name}. Ejecuta: pip install -r requirements.txt\n{MENSAJE_INSTALACION}")

    # 2) Reanudar un entrenamiento
    if args.reanudar:
        ckpt = ultimo_checkpoint(base_runs) if args.reanudar == "auto" else Path(args.reanudar)
        if ckpt is None or not ckpt.exists():
            sys.exit(f"[ERROR] No se encontró el checkpoint a reanudar ({args.reanudar}). "
                     f"Busca los last.pt en {base_runs}\\<nombre>\\weights\\")
        print(f"\nReanudando desde: {ckpt}")
        try:
            YOLO(str(ckpt)).train(resume=True)
        except torch.cuda.OutOfMemoryError:
            print(CONSEJO_MEMORIA)
            return 1
        return 0

    # 3) Entrenamiento nuevo
    print("\nDataset:", data_yaml)
    comprobar_dataset(data_yaml, clases)

    nombre = args.nombre or f"{Path(modelo).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    ruta_modelo = dir_pesos / Path(modelo).name
    try:
        yolo = YOLO(str(ruta_modelo))          # descarga los pesos preentrenados en weights/ si no existen
    except Exception:
        yolo = YOLO(modelo)

    parametros = dict(data=str(data_yaml), epochs=epocas, imgsz=imgsz, batch=batch, lr0=lr0,
                      device=dev, workers=workers, seed=semilla, patience=ent["patience"], amp=amp,
                      project=str(base_runs.resolve()), name=nombre, exist_ok=False, plots=True,
                      pretrained=True, verbose=True, optimizer=ent["optimizer"])
    perfil = args.aumentos or ent["aumentos"]
    if perfil == "reforzado":
        parametros.update(AUMENTOS_REFORZADOS)
        print(f"[INFO] Aumentos de datos REFORZADOS: {AUMENTOS_REFORZADOS}")
    elif perfil != "normal":
        sys.exit(f"[ERROR] 'aumentos' debe ser normal o reforzado (no '{perfil}').")
    if ent["optimizer"] == "auto":
        print("[INFO] optimizer=auto: Ultralytics elige optimizador y learning rate por su cuenta e "
              "IGNORA lr0. Para fijarlo usa, en config.yaml, optimizer: SGD (lr0 0.01) o AdamW (lr0 0.001).")

    def guardar_config(trainer):
        copia = {"modelo": modelo, "parametros": {k: (str(v) if isinstance(v, Path) else v)
                                                   for k, v in parametros.items()},
                 "clases": clases, "fecha": datetime.now().isoformat(timespec="seconds"),
                 "torch": str(torch.__version__), "cuda": str(torch.version.cuda)}
        with open(Path(trainer.save_dir) / "config_usada.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(copia, f, allow_unicode=True, sort_keys=False)

    def mostrar_gpu(trainer):
        if torch.cuda.is_available() and isinstance(dev, int):
            pico = torch.cuda.max_memory_reserved(dev) / 1024**3
            total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
            print(f"   [GPU] pico de VRAM reservada: {pico:.2f} / {total:.2f} GB "
                  f"(época {trainer.epoch + 1}/{trainer.epochs})")

    yolo.add_callback("on_train_start", guardar_config)
    yolo.add_callback("on_fit_epoch_end", mostrar_gpu)

    print(f"\nIniciando entrenamiento '{nombre}': modelo={modelo}, épocas={epocas}, imgsz={imgsz}, "
          f"batch={batch}, AMP={amp}, device={dev}, workers={workers}\n")
    try:
        yolo.train(**parametros)
    except torch.cuda.OutOfMemoryError:
        print(CONSEJO_MEMORIA)
        return 1
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(CONSEJO_MEMORIA)
        else:
            print(f"\n[ERROR] El entrenamiento falló: {e}")
            print("Si el mensaje menciona CUDA, ejecuta: python scripts/verificar_gpu.py")
        return 1
    except KeyboardInterrupt:
        print(f"\nEntrenamiento interrumpido. Continúa luego con:\n  python scripts/entrenar.py --reanudar auto")
        return 1

    # 4) Resultados
    carpeta = Path(yolo.trainer.save_dir)
    mejor = carpeta / "weights" / "best.pt"
    if mejor.exists():
        copia = dir_pesos / f"{nombre}_best.pt"
        shutil.copy2(mejor, copia)
        print(f"\nMejores pesos      : {mejor}")
        print(f"Copia de los pesos : {copia}")
    print(f"Carpeta del entrenamiento (métricas, gráficas, config): {carpeta}")
    print("Siguiente paso: evaluar con el conjunto de PRUEBA (evaluar.py). "
          "Las métricas de validación de arriba no sustituyen esa evaluación.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
