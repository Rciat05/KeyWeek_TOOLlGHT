"""
dividir_datasets.py - Divide el dataset unificado en train / val / test SIN fuga de datos.

Las imágenes casi idénticas se agrupan y cada GRUPO va entero a un único subconjunto:
  * copias de aumento de Roboflow (nombre "<base>.rf.<hash>"): mismo grupo
  * imágenes visualmente casi idénticas (hash perceptual dHash): mismo grupo
Luego se asignan los grupos a train/val/test con una heurística voraz que intenta
acercarse a las proporciones pedidas tanto en imágenes como en cajas por clase.

Genera dataset_yolo/ (images/ y labels/ por subconjunto + data.yaml) usando
enlaces duros (sin duplicar espacio) o copia si no es posible.

Uso:
    python scripts/dividir_datasets.py
    python scripts/dividir_datasets.py --rehacer
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
    import yaml
    from PIL import Image
except ImportError as e:
    sys.exit(f"[ERROR] Falta una dependencia ({e.name}). Ejecuta: pip install -r requirements.txt")

SPLITS = ("train", "val", "test")


def cargar_config(ruta: Path) -> dict:
    if not ruta.exists():
        sys.exit(f"[ERROR] No existe {ruta}")
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dhash_bits(ruta: Path, lado: int = 16) -> np.ndarray:
    with Image.open(ruta) as im:
        g = im.convert("L").resize((lado + 1, lado), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def construir_grupos(filas: list[dict], carpeta_img: Path, umbral: int) -> list[list[int]]:
    n = len(filas)
    uf = UnionFind(n)

    # 1) Copias de aumento de Roboflow: "<base>.rf.<hash>"
    por_base: dict[tuple, int] = {}
    for i, f in enumerate(filas):
        m = re.match(r"^(.*)\.rf\.[0-9a-fA-F]+$", f["stem_original"])
        if m:
            clave = (f["dataset"], m.group(1))
            if clave in por_base:
                uf.union(por_base[clave], i)
            else:
                por_base[clave] = i

    # 2) Casi duplicados por hash perceptual
    bits = np.stack([dhash_bits(carpeta_img / f"{f['nombre_nuevo']}{f['extension']}") for f in filas])
    A = bits.astype(np.float32)
    B = 1.0 - A
    dist = A.shape[1] - (A @ A.T + B @ B.T)
    pares = np.argwhere(np.triu(dist <= umbral, k=1))
    for i, j in pares:
        uf.union(int(i), int(j))

    grupos: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        grupos[uf.find(i)].append(i)
    return list(grupos.values())


def leer_ids(txt: Path) -> list[int]:
    ids = []
    for linea in txt.read_text(encoding="utf-8").splitlines():
        p = linea.split()
        if p:
            ids.append(int(p[0]))
    return ids


def asignar(grupos, info_cajas, n_img, ratios, semilla):
    """Asignación voraz de grupos a subconjuntos."""
    rng = random.Random(semilla)
    total_cls: Counter = Counter()
    for g in grupos:
        for i in g:
            total_cls.update(info_cajas[i])
    objetivo_img = {s: n_img * ratios[s] for s in SPLITS}
    objetivo_cls = {s: {c: t * ratios[s] for c, t in total_cls.items()} for s in SPLITS}
    act_img = {s: 0 for s in SPLITS}
    act_cls = {s: Counter() for s in SPLITS}

    orden = list(range(len(grupos)))
    rng.shuffle(orden)
    orden.sort(key=lambda k: -len(grupos[k]))  # estable: empates en orden aleatorio

    asignacion = {}
    for k in orden:
        g = grupos[k]
        cnt: Counter = Counter()
        for i in g:
            cnt.update(info_cajas[i])
        mejor, mejor_score = [], None
        for s in SPLITS:
            if ratios[s] <= 0:
                continue
            score = len(g) * (objetivo_img[s] - act_img[s]) / max(n_img, 1)
            for c, n in cnt.items():
                score += n * (objetivo_cls[s][c] - act_cls[s][c]) / total_cls[c]
            if mejor_score is None or score > mejor_score + 1e-12:
                mejor, mejor_score = [s], score
            elif abs(score - mejor_score) <= 1e-12:
                mejor.append(s)
        s = rng.choice(mejor)
        asignacion[k] = s
        act_img[s] += len(g)
        for c, n in cnt.items():
            act_cls[s][c] += n
    return asignacion


def enlazar_o_copiar(origen: Path, destino: Path):
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)


def main() -> int:
    ap = argparse.ArgumentParser(description="División train/val/test por grupos")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rehacer", action="store_true")
    args = ap.parse_args()

    cfg = cargar_config(Path(r"C:/Ever/Ciclo 4/detector_herramientas/scripts/config.yaml"))
    clases = cfg["clases"]
    div = cfg.get("division", {})
    ratios = {"train": div.get("train", 0.7), "val": div.get("val", 0.2), "test": div.get("test", 0.1)}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        sys.exit(f"[ERROR] Las proporciones deben sumar 1.0 (suman {sum(ratios.values())}).")
    semilla = int(div.get("semilla", cfg.get("proyecto", {}).get("semilla", 42)))
    umbral = int(div.get("umbral_hamming", 10))
    unif = Path(cfg.get("preparacion", {}).get("directorio_salida", "datos_unificados"))
    salida = Path(div.get("directorio_salida", "dataset_yolo"))

    if not (unif / "manifiesto.csv").exists():
        sys.exit(f"[ERROR] No existe {unif / 'manifiesto.csv'}. Ejecuta antes preparar_datasets.py")
    if salida.exists() and any(salida.iterdir()):
        if args.rehacer and (salida / "data.yaml").exists():
            shutil.rmtree(salida)
        else:
            sys.exit(f"[ERROR] '{salida}' ya existe. Usa --rehacer para regenerarla.")

    with open(unif / "manifiesto.csv", newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        sys.exit("[ERROR] El manifiesto está vacío.")
    print(f"Imágenes unificadas: {len(filas)}. Calculando grupos (casi duplicados)...")

    carpeta_img, carpeta_lab = unif / "images", unif / "labels"
    info_cajas = [Counter(leer_ids(carpeta_lab / f"{r['nombre_nuevo']}.txt")) for r in filas]
    for c in set().union(*[set(ic) for ic in info_cajas]):
        if c >= len(clases):
            sys.exit(f"[ERROR] ID de clase {c} fuera de rango para {len(clases)} clases en config.yaml.")

    grupos = construir_grupos(filas, carpeta_img, umbral)
    tamanos = sorted((len(g) for g in grupos), reverse=True)
    print(f"Grupos: {len(grupos)} | mayor grupo: {tamanos[0]} imágenes | "
          f"grupos de 1 imagen: {sum(1 for t in tamanos if t == 1)}")
    if tamanos[0] > 0.05 * len(filas):
        print(f"[AVISO] El mayor grupo ({tamanos[0]}) supera el 5% de las imágenes: puede desequilibrar "
              f"el reparto. Si son imágenes distintas, baja 'umbral_hamming' en config.yaml.")

    asignacion = asignar(grupos, info_cajas, len(filas), ratios, semilla)
    split_de: dict[int, str] = {}
    for k, g in enumerate(grupos):
        for i in g:
            split_de[i] = asignacion[k]
    grupo_de = {i: k for k, g in enumerate(grupos) for i in g}

    # --- Escribir dataset_yolo ---
    for s in SPLITS:
        (salida / "images" / s).mkdir(parents=True, exist_ok=True)
        (salida / "labels" / s).mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(filas):
        s = split_de[i]
        enlazar_o_copiar(carpeta_img / f"{r['nombre_nuevo']}{r['extension']}",
                         salida / "images" / s / f"{r['nombre_nuevo']}{r['extension']}")
        enlazar_o_copiar(carpeta_lab / f"{r['nombre_nuevo']}.txt",
                         salida / "labels" / s / f"{r['nombre_nuevo']}.txt")

    data = {"path": salida.resolve().as_posix(), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {i: c for i, c in enumerate(clases)}}
    with open(salida / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    # --- Informe ---
    img_cls = {s: Counter() for s in SPLITS}
    caj_cls = {s: Counter() for s in SPLITS}
    n_img = Counter()
    fondo = Counter()
    for i, ic in enumerate(info_cajas):
        s = split_de[i]
        n_img[s] += 1
        if not ic:
            fondo[s] += 1
        for c, n in ic.items():
            img_cls[s][c] += 1
            caj_cls[s][c] += n

    print("\n" + "=" * 76)
    print("DISTRIBUCIÓN (imágenes que contienen la clase / cajas)")
    print("=" * 76)
    print(f"{'clase':<18}" + "".join(f"{s:>16}" for s in SPLITS))
    advertencias = []
    for c, nombre in enumerate(clases):
        fila = f"{nombre:<18}"
        for s in SPLITS:
            fila += f"{str(img_cls[s][c]) + ' / ' + str(caj_cls[s][c]):>16}"
            if img_cls[s][c] == 0 and s != "train":
                advertencias.append(f"'{nombre}' no aparece en {s}: no se podrá evaluar.")
            elif s != "train" and img_cls[s][c] < 5:
                advertencias.append(f"'{nombre}' tiene solo {img_cls[s][c]} imágenes en {s}: "
                                    f"su evaluación será poco fiable.")
        print(fila)
    print(f"{'TOTAL imágenes':<18}" + "".join(f"{n_img[s]:>16}" for s in SPLITS))
    print(f"{'(sin objetos)':<18}" + "".join(f"{fondo[s]:>16}" for s in SPLITS))
    reales = {s: n_img[s] / len(filas) for s in SPLITS}
    print("Proporción real     : " + ", ".join(f"{s} {reales[s]:.1%}" for s in SPLITS)
          + "  (pedida: " + ", ".join(f"{s} {ratios[s]:.0%}" for s in SPLITS) + ")")
    if advertencias:
        print("\n[AVISOS]")
        for a in dict.fromkeys(advertencias):
            print("  -", a)

    destino = Path(cfg.get("proyecto", {}).get("directorio_resultados", "runs")) / "informes"
    destino.mkdir(parents=True, exist_ok=True)
    with open(destino / "division_asignacion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nombre_nuevo", "dataset", "subconjunto", "grupo"])
        for i, r in enumerate(filas):
            w.writerow([r["nombre_nuevo"], r["dataset"], split_de[i], grupo_de[i]])
    with open(destino / "division.json", "w", encoding="utf-8") as f:
        json.dump({"semilla": semilla, "proporciones": ratios, "umbral_hamming": umbral,
                   "grupos": len(grupos), "imagenes": dict(n_img),
                   "imagenes_por_clase": {s: {clases[c]: n for c, n in img_cls[s].items()} for s in SPLITS},
                   "cajas_por_clase": {s: {clases[c]: n for c, n in caj_cls[s].items()} for s in SPLITS},
                   "advertencias": advertencias}, f, ensure_ascii=False, indent=2)
    print(f"\nDataset listo en : {salida}   (data.yaml incluido)")
    print(f"Informes en      : {destino}")
    print("Recuerda: el subconjunto 'test' NO debe usarse para ajustar hiperparámetros.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
