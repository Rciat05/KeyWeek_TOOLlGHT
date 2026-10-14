"""
sobremuestrear.py - Repite imágenes del subconjunto de ENTRENAMIENTO de las clases con pocos ejemplos.

  * Solo toca dataset_yolo/images|labels/train. Validación y prueba NO se tocan, así que
    las métricas siguen siendo honestas (no hay copias en val/test).
  * Las copias se llaman <nombre>__rep1.jpg, <nombre>__rep2.jpg ... (enlaces duros: no
    ocupan espacio extra; si no es posible, copia).
  * Es repetible: cada ejecución borra primero las copias __rep* anteriores y las regenera.
  * Factor por clase = ceil(objetivo / nº de imágenes de train de esa clase), con un tope.
    Una imagen con varias clases usa el mayor factor de las que contiene.

Importante: repetir imágenes NO añade información nueva. Con el aumento de datos de
Ultralytics (mosaico, color, escala...) cada repetición se ve distinta, y evita que el modelo
ignore las clases pequeñas, pero no sustituye a tener más fotos reales.

Uso:
    python scripts/sobremuestrear.py --simular                 (solo muestra el plan)
    python scripts/sobremuestrear.py                           (objetivo 150 imágenes por clase)
    python scripts/sobremuestrear.py --objetivo 200 --max-factor 8
    python scripts/sobremuestrear.py --factores martillo=8,pinzas=6
    python scripts/sobremuestrear.py --deshacer                (elimina todas las copias)
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("[ERROR] Falta pyyaml. Ejecuta: pip install -r requirements.txt")

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MARCA = "__rep"


def enlazar_o_copiar(origen: Path, destino: Path) -> None:
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)


def borrar_copias(carpeta_img: Path, carpeta_lab: Path) -> int:
    n = 0
    for p in list(carpeta_img.glob(f"*{MARCA}*")) + list(carpeta_lab.glob(f"*{MARCA}*")):
        p.unlink()
        n += 1
    return n


def clases_de_etiqueta(txt: Path) -> set[int]:
    ids = set()
    if txt.exists():
        for linea in txt.read_text(encoding="utf-8").splitlines():
            partes = linea.split()
            if partes:
                ids.add(int(partes[0]))
    return ids


def calcular_factores(imgs_por_clase: dict[int, int], nombres: dict[int, str], objetivo: int,
                      max_factor: int, manuales: dict[str, int]) -> dict[int, int]:
    factores = {}
    for c in nombres:
        n = imgs_por_clase.get(c, 0)
        if nombres[c] in manuales:
            factores[c] = max(1, manuales[nombres[c]])
        elif n == 0 or n >= objetivo:
            factores[c] = 1
        else:
            factores[c] = min(max_factor, math.ceil(objetivo / n))
    return factores


def main() -> int:
    ap = argparse.ArgumentParser(description="Sobremuestreo de clases pequeñas (solo train)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--objetivo", type=int, default=150, help="nº de imágenes de train que se desea por clase")
    ap.add_argument("--max-factor", type=int, default=10, help="máximo de repeticiones por imagen")
    ap.add_argument("--factores", default="", help="manual: clase=factor,clase=factor (p. ej. martillo=8,pinzas=6)")
    ap.add_argument("--simular", action="store_true", help="solo muestra el plan, no escribe nada")
    ap.add_argument("--deshacer", action="store_true", help="elimina todas las copias __rep")
    ap.add_argument("--data", help="ruta a data.yaml (por defecto dataset_yolo/data.yaml)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    data_yaml = Path(args.data or Path((cfg or {}).get("division", {}).get("directorio_salida", "dataset_yolo")) / "data.yaml")
    if not data_yaml.exists():
        sys.exit(f"[ERROR] No existe {data_yaml}. Ejecuta antes dividir_datasets.py")
    datos = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    nombres = {int(k): v for k, v in datos["names"].items()}
    raiz = Path(datos["path"])
    carpeta_img = raiz / "images" / "train"
    carpeta_lab = raiz / "labels" / "train"
    if not carpeta_img.exists():
        sys.exit(f"[ERROR] No existe {carpeta_img}")

    manuales = {}
    for par in filter(None, (x.strip() for x in args.factores.split(","))):
        try:
            clase, valor = par.split("=")
            manuales[clase.strip()] = int(valor)
        except ValueError:
            sys.exit(f"[ERROR] Formato inválido en --factores: '{par}' (usa clase=factor)")
    desconocidas = [c for c in manuales if c not in nombres.values()]
    if desconocidas:
        sys.exit(f"[ERROR] Clases desconocidas en --factores: {desconocidas}. Válidas: {list(nombres.values())}")

    if args.deshacer:
        n = 0 if args.simular else borrar_copias(carpeta_img, carpeta_lab)
        print(f"Copias eliminadas: {n}")
        return 0

    # Imágenes ORIGINALES (se ignoran las copias de ejecuciones anteriores)
    originales = sorted(p for p in carpeta_img.glob("*") if p.suffix.lower() in EXT_IMG and MARCA not in p.stem)
    por_imagen = {p: clases_de_etiqueta(carpeta_lab / f"{p.stem}.txt") for p in originales}
    imgs_por_clase: dict[int, int] = defaultdict(int)
    for ids in por_imagen.values():
        for c in ids:
            imgs_por_clase[c] += 1

    factores = calcular_factores(imgs_por_clase, nombres, args.objetivo, args.max_factor, manuales)
    factor_img = {p: max([factores[c] for c in ids], default=1) for p, ids in por_imagen.items()}

    # Resultado esperado
    despues: dict[int, int] = defaultdict(int)
    for p, ids in por_imagen.items():
        for c in ids:
            despues[c] += factor_img[p]
    total_extra = sum(f - 1 for f in factor_img.values())

    print("=" * 74)
    print(f"SOBREMUESTREO DEL SUBCONJUNTO DE ENTRENAMIENTO  (objetivo {args.objetivo}, tope x{args.max_factor})")
    print("=" * 74)
    print(f"{'clase':<18}{'img train':>11}{'factor':>8}{'img tras repetir':>20}")
    for c, nombre in nombres.items():
        aviso = "   <-- sin imágenes en train" if imgs_por_clase.get(c, 0) == 0 else ""
        print(f"{nombre:<18}{imgs_por_clase.get(c, 0):>11}{'x' + str(factores[c]):>8}{despues.get(c, 0):>20}{aviso}")
    print(f"\nImágenes únicas en train: {len(originales)}   copias a crear: {total_extra}   "
          f"total tras repetir: {len(originales) + total_extra}")

    if args.simular:
        print("\n(Simulación: no se ha escrito nada.)")
        return 0

    borradas = borrar_copias(carpeta_img, carpeta_lab)
    if borradas:
        print(f"Copias anteriores eliminadas: {borradas}")
    for p, f in factor_img.items():
        txt = carpeta_lab / f"{p.stem}.txt"
        for i in range(1, f):
            enlazar_o_copiar(p, carpeta_img / f"{p.stem}{MARCA}{i}{p.suffix}")
            if txt.exists():
                enlazar_o_copiar(txt, carpeta_lab / f"{p.stem}{MARCA}{i}.txt")
    print(f"\nCopias creadas: {total_extra}  (solo en train; val y test intactos)")
    print("Siguiente paso: reentrenar, por ejemplo")
    print("  python scripts/entrenar.py --modelo weights/entrenamiento_v1_best.pt --aumentos reforzado "
          "--epocas 60 --nombre entrenamiento_v2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
