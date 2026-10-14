"""
evaluar.py - Evaluación del detector con el conjunto de PRUEBA (test).

Todas las cifras proceden de ejecutar el modelo sobre las imágenes de test: nada se inventa.
Calcula y guarda:
  * Precision, Recall, F1, mAP@50 y mAP@50-95: globales y por clase
  * Matriz de confusión (la de Ultralytics, en imagen) + lista de confusiones principales
  * Gráfica de métricas por clase
  * Imágenes de ejemplo con cajas, clase y confianza
  * metricas_test.json, metricas_por_clase.csv y resumen.txt
Cada evaluación va a una carpeta nueva en runs/evaluacion/ (no sobrescribe).

Precision, Recall y F1 son los de Ultralytics: se calculan en el umbral de confianza que
maximiza el F1 medio. mAP@50 y mAP@50-95 no dependen de un umbral. La matriz de confusión
usa confianza 0.25.

IMPORTANTE: el conjunto de test sirve para medir al final. No ajustes hiperparámetros
mirando estos resultados (para eso está la validación); si lo haces, el test deja de ser
una medida honesta.

Uso:
    python scripts/evaluar.py
    python scripts/evaluar.py --pesos weights/entrenamiento_v1_best.pt
    python scripts/evaluar.py --batch 4 --ejemplos 40
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

try:
    import yaml
except ImportError:
    sys.exit("[ERROR] Falta pyyaml. Ejecuta: pip install -r requirements.txt")

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MIN_IMAGENES_FIABLE = 5


# ----------------------------------------------------------------------------
# Funciones auxiliares (sin dependencia de Ultralytics)
# ----------------------------------------------------------------------------
def contar_test(raiz: Path, rel_test: str, n_clases: int):
    """Cuenta imágenes e instancias por clase leyendo las etiquetas reales del test."""
    carpeta_img = raiz / rel_test
    carpeta_lab = raiz / "labels" / carpeta_img.name
    imagenes = sorted(p for p in carpeta_img.glob("*") if p.suffix.lower() in EXT_IMG) if carpeta_img.exists() else []
    inst = np.zeros(n_clases, dtype=int)
    imgs = np.zeros(n_clases, dtype=int)
    sin_objetos = 0
    for img in imagenes:
        txt = carpeta_lab / f"{img.stem}.txt"
        ids = []
        if txt.exists():
            for linea in txt.read_text(encoding="utf-8").splitlines():
                partes = linea.split()
                if partes:
                    ids.append(int(partes[0]))
        if not ids:
            sin_objetos += 1
        for c in ids:
            inst[c] += 1
        for c in set(ids):
            imgs[c] += 1
    return imagenes, inst, imgs, sin_objetos


def construir_tabla(names: dict, ap_class_index, p, r, f1, ap50, ap, inst, imgs) -> list[dict]:
    pos = {int(c): k for k, c in enumerate(ap_class_index)}
    filas = []
    for c in sorted(names):
        k = pos.get(c)
        nan = float("nan")
        filas.append({
            "id": c, "clase": names[c], "imagenes_test": int(imgs[c]), "instancias_test": int(inst[c]),
            "precision": float(p[k]) if k is not None else nan,
            "recall": float(r[k]) if k is not None else nan,
            "f1": float(f1[k]) if k is not None else nan,
            "map50": float(ap50[k]) if k is not None else nan,
            "map50_95": float(ap[k]) if k is not None else nan,
            "fiabilidad": ("SIN DATOS EN TEST" if imgs[c] == 0
                           else "POCO FIABLE" if imgs[c] < MIN_IMAGENES_FIABLE else "ok"),
        })
    return filas


def analizar_confusion(matriz: np.ndarray, names: dict, top: int = 8):
    """matriz[predicha, real]; el último índice es 'fondo'. Devuelve confusiones, FN y FP."""
    nc = len(names)
    confusiones, perdidas, falsas = [], {}, {}
    for real in range(nc):
        for pred in range(nc):
            if pred != real and matriz[pred, real] > 0:
                confusiones.append((names[real], names[pred], float(matriz[pred, real])))
        perdidas[names[real]] = (float(matriz[nc, real]), float(matriz[:, real].sum()))
    for pred in range(nc):
        falsas[names[pred]] = (float(matriz[pred, nc]), float(matriz[pred, :].sum()))
    confusiones.sort(key=lambda t: -t[2])
    return confusiones[:top], perdidas, falsas


def fmt(v, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v * 100:5.1f}%" if pct else f"{v:.3f}"


def grafica_por_clase(filas: list[dict], destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    etiquetas = [f"{f['clase']}\n(n={f['imagenes_test']} img)" for f in filas]
    metricas = [("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP@50"), ("map50_95", "mAP@50-95")]
    x = np.arange(len(filas))
    ancho = 0.2
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(filas)), 5.5))
    for i, (clave, nombre) in enumerate(metricas):
        vals = [0 if np.isnan(f[clave]) else f[clave] for f in filas]
        ax.bar(x + (i - 1.5) * ancho, vals, ancho, label=nombre)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor (0-1)")
    ax.set_title("Métricas por clase en el conjunto de prueba\n(n pequeño = resultado poco fiable)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluación con el conjunto de prueba")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pesos", help="ruta a best.pt (por defecto, los más recientes)")
    ap.add_argument("--data", help="ruta a data.yaml (por defecto dataset_yolo/data.yaml)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--conf-ejemplos", type=float, default=0.40, help="umbral para las imágenes de ejemplo")
    ap.add_argument("--ejemplos", type=int, default=30, help="nº de imágenes de ejemplo a guardar (0 = ninguna)")
    ap.add_argument("--nombre", help="nombre de la carpeta de resultados")
    args = ap.parse_args()

    cfg = {}
    if Path(args.config).exists():
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    base = Path(cfg.get("proyecto", {}).get("directorio_resultados", "runs"))
    data_yaml = Path(args.data or Path(cfg.get("division", {}).get("directorio_salida", "dataset_yolo")) / "data.yaml")
    if not data_yaml.exists():
        sys.exit(f"[ERROR] No existe {data_yaml}. Ejecuta antes dividir_datasets.py")
    datos = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if "test" not in datos:
        sys.exit("[ERROR] data.yaml no define el subconjunto 'test'.")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"[ERROR] Falta {e.name}. Ejecuta: pip install -r requirements.txt")

    from detectar_camara import buscar_pesos
    from verificar_gpu import seleccionar_dispositivo

    pesos = Path(args.pesos) if args.pesos else buscar_pesos(Path("weights"), base)
    if pesos is None or not pesos.exists():
        sys.exit("[ERROR] No encontré pesos. Indica --pesos weights/<nombre>_best.pt")
    try:
        dev_str, _ = seleccionar_dispositivo(args.dispositivo)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"[ERROR] {e}")
    dispositivo = int(dev_str.split(":")[1]) if dev_str.startswith("cuda") else "cpu"

    modelo = YOLO(str(pesos))
    names = {int(k): v for k, v in modelo.names.items()}
    esperadas = {int(k): v for k, v in datos["names"].items()}
    if names != esperadas:
        sys.exit(f"[ERROR] Las clases del modelo {list(names.values())} no coinciden con las de "
                 f"data.yaml {list(esperadas.values())}. ¿Pesos de otro entrenamiento?")

    raiz = Path(datos["path"])
    imagenes, inst, imgs, sin_objetos = contar_test(raiz, datos["test"], len(names))
    if not imagenes:
        sys.exit(f"[ERROR] No hay imágenes de prueba en {raiz / datos['test']}.")
    print(f"Pesos          : {pesos}")
    print(f"Imágenes test  : {len(imagenes)}  (sin objetos: {sin_objetos})")

    nombre = args.nombre or f"test_{pesos.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    proyecto = (base / "evaluacion").resolve()

    try:
        metricas = modelo.val(data=str(data_yaml), split="test", imgsz=args.imgsz, batch=args.batch,
                              device=dispositivo, project=str(proyecto), name=nombre, exist_ok=False,
                              plots=True, verbose=False, conf=0.001, iou=0.6)
    except torch.cuda.OutOfMemoryError:
        sys.exit("[ERROR] Sin memoria de GPU. Prueba --batch 2 --imgsz 512 y cierra otras aplicaciones.")
    salida = Path(metricas.save_dir)

    caja = metricas.box
    filas = construir_tabla(names, caja.ap_class_index, caja.p, caja.r, caja.f1, caja.ap50, caja.ap, inst, imgs)
    f1_medio = float(np.mean(caja.f1)) if len(caja.f1) else float("nan")
    global_ = {"precision": float(caja.mp), "recall": float(caja.mr), "f1_medio": f1_medio,
               "map50": float(caja.map50), "map50_95": float(caja.map)}

    # --- Informe en pantalla y resumen.txt ---
    lineas = []
    lineas.append("=" * 92)
    lineas.append("EVALUACIÓN EN EL CONJUNTO DE PRUEBA")
    lineas.append("=" * 92)
    lineas.append(f"{'clase':<18}{'img':>5}{'inst':>6}{'Precision':>11}{'Recall':>9}{'F1':>8}"
                  f"{'mAP50':>9}{'mAP50-95':>10}   fiabilidad")
    for f in filas:
        lineas.append(f"{f['clase']:<18}{f['imagenes_test']:>5}{f['instancias_test']:>6}"
                      f"{fmt(f['precision']):>11}{fmt(f['recall']):>9}{fmt(f['f1']):>8}"
                      f"{fmt(f['map50']):>9}{fmt(f['map50_95']):>10}   {f['fiabilidad']}")
    lineas.append("-" * 92)
    lineas.append(f"{'GLOBAL':<18}{len(imagenes):>5}{int(inst.sum()):>6}{fmt(global_['precision']):>11}"
                  f"{fmt(global_['recall']):>9}{fmt(global_['f1_medio']):>8}{fmt(global_['map50']):>9}"
                  f"{fmt(global_['map50_95']):>10}")

    matriz = np.array(metricas.confusion_matrix.matrix)
    confusiones, perdidas, falsas = analizar_confusion(matriz, names)
    lineas.append("\nCONFUSIONES PRINCIPALES (real -> predicha como, a confianza 0.25):")
    if confusiones:
        for real, pred, n in confusiones:
            lineas.append(f"  {real} -> {pred}: {n:.0f}")
    else:
        lineas.append("  ninguna entre clases")
    lineas.append("\nHERRAMIENTAS NO DETECTADAS (falsos negativos) / FALSAS DETECCIONES (falsos positivos):")
    for c in names.values():
        fn, tot_real = perdidas[c]
        fp, tot_pred = falsas[c]
        lineas.append(f"  {c:<18} no detectadas: {fn:.0f} de {tot_real:.0f} reales | "
                      f"falsas detecciones: {fp:.0f} de {tot_pred:.0f} predichas")

    avisos = [f"'{f['clase']}': {f['fiabilidad']} ({f['imagenes_test']} imágenes de test): "
              f"su métrica no es concluyente." for f in filas if f["fiabilidad"] != "ok"]
    if avisos:
        lineas.append("\n[AVISOS]")
        lineas += [f"  - {a}" for a in avisos]
    texto = "\n".join(lineas)
    print("\n" + texto)
    (salida / "resumen.txt").write_text(texto, encoding="utf-8")

    # --- Archivos ---
    with open(salida / "metricas_por_clase.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    with open(salida / "metricas_test.json", "w", encoding="utf-8") as fh:
        json.dump({"pesos": str(pesos), "fecha": datetime.now().isoformat(timespec="seconds"),
                   "imagenes_test": len(imagenes), "global": global_, "por_clase": filas,
                   "matriz_confusion_pred_x_real": matriz.tolist(),
                   "orden_matriz": list(names.values()) + ["fondo"],
                   "ajustes": {"imgsz": args.imgsz, "batch": args.batch, "iou": 0.6}},
                  fh, ensure_ascii=False, indent=2)
    grafica_por_clase(filas, salida / "metricas_por_clase.png")

    # --- Imágenes de ejemplo con cajas ---
    if args.ejemplos > 0:
        muestra = random.Random(42).sample(imagenes, min(args.ejemplos, len(imagenes)))
        modelo.predict(source=[str(p) for p in muestra], conf=args.conf_ejemplos, imgsz=args.imgsz,
                       device=dispositivo, save=True, project=str(salida), name="ejemplos",
                       exist_ok=True, verbose=False)

    print(f"\nResultados guardados en: {salida}")
    print("  - metricas_por_clase.png   (gráfica por clase)")
    print("  - confusion_matrix.png / confusion_matrix_normalized.png   (matriz de confusión)")
    print("  - ejemplos\\                (imágenes de test con cajas y confianza)")
    print("  - resumen.txt, metricas_por_clase.csv, metricas_test.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
