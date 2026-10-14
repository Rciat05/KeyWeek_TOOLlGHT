"""
ver_ids.py - Muestra QUÉ OBJETO es cada ID de clase de cada dataset.

Para cada dataset genera UNA imagen (hoja de contacto) con una fila por ID de clase:
en cada fila, varias imágenes de ejemplo donde las cajas de ese ID están dibujadas en
rojo (las cajas de otros IDs, en gris fino). Así ves con tus ojos qué es el ID 0, el 1...

Solo lectura: no modifica los datasets. Guarda en runs/informes/ids/<dataset>.jpg

Uso:
    python scripts/ver_ids.py
    python scripts/ver_ids.py --muestras 6 --dataset guantes
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspeccionar_datasets import cargar_config  # noqa: E402
from preparar_datasets import buscar_etiqueta, indexar, leer_cajas  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402

LADO = 280          # tamaño de cada miniatura
MARGEN_TEXTO = 150  # ancho de la columna de texto a la izquierda


def miniatura(img_path: Path, cajas, id_objetivo: int) -> Image.Image:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    grosor = max(2, round(max(w, h) / 200))
    for cid, x, y, bw, bh in cajas:
        caja = [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
        if cid == id_objetivo:
            d.rectangle(caja, outline=(255, 0, 0), width=grosor + 1)
        else:
            d.rectangle(caja, outline=(150, 150, 150), width=1)
    im.thumbnail((LADO, LADO))
    lienzo = Image.new("RGB", (LADO, LADO), (255, 255, 255))
    lienzo.paste(im, ((LADO - im.width) // 2, (LADO - im.height) // 2))
    return lienzo


def main() -> int:
    ap = argparse.ArgumentParser(description="Ver qué objeto es cada ID de clase")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--muestras", type=int, default=5, help="imágenes de ejemplo por ID")
    ap.add_argument("--dataset", help="solo este dataset (nombre de config.yaml)")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    cfg = cargar_config(Path(r"C:/Ever/Ciclo 4/detector_herramientas/scripts/config.yaml"))
    salida = Path(cfg.get("proyecto", {}).get("directorio_resultados", "runs")) / "informes" / "ids"
    salida.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.semilla)

    for d in cfg["datasets"]:
        if args.dataset and d["nombre"] != args.dataset:
            continue
        ruta = Path(d["ruta"])
        if not ruta.is_dir():
            print(f"[ERROR] '{d['nombre']}': la ruta no existe ({ruta})")
            continue
        imagenes, indice, _ = indexar(ruta)
        por_id: dict[int, list[tuple[Path, list]]] = defaultdict(list)
        for img in imagenes:
            txt, _ = buscar_etiqueta(img, indice)
            if txt is None:
                continue
            cajas, _, _ = leer_cajas(txt)
            for cid in {c[0] for c in cajas}:
                por_id[cid].append((img, cajas))
        if not por_id:
            print(f"[AVISO] '{d['nombre']}': sin cajas legibles.")
            continue

        ids = sorted(por_id)
        hoja = Image.new("RGB", (MARGEN_TEXTO + LADO * args.muestras, LADO * len(ids)), (255, 255, 255))
        dib = ImageDraw.Draw(hoja)
        for fila, cid in enumerate(ids):
            ejemplos = rng.sample(por_id[cid], min(args.muestras, len(por_id[cid])))
            dib.text((10, fila * LADO + LADO // 2 - 12), f"ID {cid}", fill=(200, 0, 0))
            dib.text((10, fila * LADO + LADO // 2 + 6), f"{len(por_id[cid])} imágenes", fill=(0, 0, 0))
            for col, (img, cajas) in enumerate(ejemplos):
                hoja.paste(miniatura(img, cajas, cid), (MARGEN_TEXTO + col * LADO, fila * LADO))
        destino = salida / f"{d['nombre']}.jpg"
        hoja.save(destino, quality=88)
        print(f"{d['nombre']:<20} IDs {ids}  ->  {destino}")

    print("\nAbre las imágenes de runs\\informes\\ids\\ y anota qué objeto es cada ID "
          "(rojo = el ID de esa fila).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
