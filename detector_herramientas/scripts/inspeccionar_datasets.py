"""
inspeccionar_datasets.py - Inspección de SOLO LECTURA de los datasets originales.

Para cada dataset definido en config.yaml informa de:
  - imágenes encontradas y archivos dañados
  - formato(s) de anotación detectados (YOLO .txt, Pascal VOC .xml, COCO .json)
  - clases reales encontradas (IDs o nombres) y cuántas anotaciones tiene cada una
  - imágenes sin etiqueta y etiquetas inválidas
  - duplicados exactos (dentro de cada dataset y entre datasets)

No modifica nada dentro de los datasets. Guarda un informe en
runs/informes/inspeccion.json

Uso:
    python scripts/inspeccionar_datasets.py
    python scripts/inspeccionar_datasets.py --config config.yaml
    python scripts/inspeccionar_datasets.py --sin-verificar-imagenes   (más rápido)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
    from PIL import Image
except ImportError as e:
    sys.exit(f"[ERROR] Falta una dependencia ({e.name}). Ejecuta: pip install -r requirements.txt")

EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TXT_IGNORADOS = {"classes.txt", "labels.txt", "obj.names", "requirements.txt", "license.txt"}
MAX_EJEMPLOS = 5  # ejemplos de errores a mostrar por categoría


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def cargar_config(ruta: Path) -> dict:
    if not ruta.exists():
        sys.exit(f"[ERROR] No existe el archivo de configuración: {ruta}")
    with open(ruta, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg or "datasets" not in cfg:
        sys.exit("[ERROR] config.yaml no contiene la sección 'datasets'.")
    return cfg


def hash_archivo(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def imagen_legible(ruta: Path) -> bool:
    try:
        with Image.open(ruta) as im:
            im.verify()
        with Image.open(ruta) as im:
            im.load()
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Lectores de anotaciones
# ----------------------------------------------------------------------------
def leer_yolo(txt: Path) -> tuple[list[int], list[str]]:
    """Devuelve (ids_de_clase, errores). Un .txt vacío es válido (imagen de fondo)."""
    ids, errores = [], []
    try:
        lineas = txt.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return ids, [f"{txt.name}: ilegible ({e})"]
    for n, linea in enumerate(lineas, 1):
        partes = linea.split()
        if not partes:
            continue
        try:
            cid = int(partes[0])
            vals = [float(v) for v in partes[1:]]
        except ValueError:
            errores.append(f"{txt.name}:{n} no tiene formato YOLO numérico")
            continue
        if len(vals) == 4:
            if not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                errores.append(f"{txt.name}:{n} coordenadas fuera de [0,1] o tamaño nulo")
                continue
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            errores.append(f"{txt.name}:{n} es un polígono (segmentación), no un bounding box")
            continue
        else:
            errores.append(f"{txt.name}:{n} número de valores inesperado ({len(vals)})")
            continue
        if cid < 0:
            errores.append(f"{txt.name}:{n} ID de clase negativo")
            continue
        ids.append(cid)
    return ids, errores


def leer_voc(xml: Path) -> tuple[list[str], list[str], str | None]:
    """Devuelve (nombres_de_clase, errores, nombre_de_archivo_de_imagen)."""
    nombres, errores = [], []
    try:
        raiz = ET.parse(xml).getroot()
    except Exception as e:
        return nombres, [f"{xml.name}: XML ilegible ({e})"], None
    archivo = raiz.findtext("filename")
    for obj in raiz.iter("object"):
        nombre = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if not nombre or bb is None:
            errores.append(f"{xml.name}: objeto sin nombre o sin bndbox")
            continue
        try:
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            errores.append(f"{xml.name}: bndbox no numérico para '{nombre}'")
            continue
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
            errores.append(f"{xml.name}: bndbox inválido para '{nombre}'")
            continue
        nombres.append(nombre)
    return nombres, errores, archivo


def leer_coco(js: Path) -> dict | None:
    """Devuelve un resumen del COCO JSON o None si no tiene estructura COCO."""
    try:
        datos = json.loads(js.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(datos, dict) or not {"images", "annotations", "categories"} <= datos.keys():
        return None
    categorias = {c["id"]: c["name"] for c in datos["categories"]}
    id_a_archivo = {im["id"]: Path(im["file_name"]).name for im in datos["images"]}
    por_clase: Counter = Counter()
    imagenes_con_anot: set[str] = set()
    errores: list[str] = []
    for a in datos["annotations"]:
        nombre = categorias.get(a.get("category_id"))
        if nombre is None:
            errores.append(f"{js.name}: anotación {a.get('id')} con category_id inexistente")
            continue
        bbox = a.get("bbox")
        if not bbox or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            errores.append(f"{js.name}: anotación {a.get('id')} con bbox inválido")
            continue
        por_clase[nombre] += 1
        if a.get("image_id") in id_a_archivo:
            imagenes_con_anot.add(id_a_archivo[a["image_id"]])
    return {
        "archivo": js.name,
        "categorias": categorias,
        "imagenes_listadas": set(id_a_archivo.values()),
        "imagenes_con_anotacion": imagenes_con_anot,
        "anotaciones_por_clase": por_clase,
        "errores": errores,
    }


def nombres_en_yaml(archivos_yaml: list[Path]) -> list[str] | None:
    """Busca la lista 'names' de un data.yaml (formato YOLO/Roboflow)."""
    for y in archivos_yaml:
        try:
            d = yaml.safe_load(y.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "names" in d:
            n = d["names"]
            return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    return None


# ----------------------------------------------------------------------------
# Inspección de un dataset
# ----------------------------------------------------------------------------
def inspeccionar_dataset(nombre: str, ruta: Path, verificar: bool) -> dict:
    r: dict = {"nombre": nombre, "ruta": str(ruta), "errores_fatales": []}
    if not ruta.exists() or not ruta.is_dir():
        r["errores_fatales"].append(f"La ruta no existe o no es una carpeta: {ruta}")
        return r

    archivos = [p for p in ruta.rglob("*") if p.is_file()]
    imagenes = [p for p in archivos if p.suffix.lower() in EXT_IMAGEN]
    txts = [p for p in archivos if p.suffix.lower() == ".txt" and p.name.lower() not in TXT_IGNORADOS]
    xmls = [p for p in archivos if p.suffix.lower() == ".xml"]
    jsons = [p for p in archivos if p.suffix.lower() == ".json"]
    yamls = [p for p in archivos if p.suffix.lower() in (".yaml", ".yml")]

    r["n_imagenes"] = len(imagenes)
    if not imagenes:
        r["errores_fatales"].append("El dataset no contiene imágenes.")
        return r

    # Imágenes dañadas
    r["imagenes_danadas"] = []
    if verificar:
        r["imagenes_danadas"] = [str(p) for p in imagenes if not imagen_legible(p)]

    # Hashes (duplicados exactos)
    r["hashes"] = {str(p): hash_archivo(p) for p in imagenes}

    stems_imagen = defaultdict(list)
    for p in imagenes:
        stems_imagen[p.stem].append(p)

    formatos: list[str] = []
    errores_etiquetas: list[str] = []
    imagenes_con_etiqueta: set[str] = set()

    # --- YOLO ---
    yolo_ids: Counter = Counter()
    yolo_validos = 0
    for t in txts:
        ids, errs = leer_yolo(t)
        # Un .txt solo cuenta como etiqueta si emparenta con una imagen
        if t.stem in stems_imagen:
            yolo_validos += 1
            yolo_ids.update(ids)
            errores_etiquetas += errs
            for im in stems_imagen[t.stem]:
                imagenes_con_etiqueta.add(str(im))
    if yolo_validos:
        formatos.append("YOLO")
        r["yolo_ids"] = dict(sorted(yolo_ids.items()))
        r["yolo_nombres_en_yaml"] = nombres_en_yaml(yamls)
    r["txt_sin_imagen"] = sum(1 for t in txts if t.stem not in stems_imagen)

    # --- Pascal VOC ---
    voc_nombres: Counter = Counter()
    voc_validos = 0
    for x in xmls:
        nombres, errs, archivo_img = leer_voc(x)
        stem = x.stem if x.stem in stems_imagen else (Path(archivo_img).stem if archivo_img else None)
        if stem in stems_imagen:
            voc_validos += 1
            voc_nombres.update(nombres)
            errores_etiquetas += errs
            for im in stems_imagen[stem]:
                imagenes_con_etiqueta.add(str(im))
    if voc_validos:
        formatos.append("VOC")
        r["voc_nombres"] = dict(sorted(voc_nombres.items()))

    # --- COCO ---
    coco_nombres: Counter = Counter()
    resumenes_coco = []
    for j in jsons:
        c = leer_coco(j)
        if c is None:
            continue
        resumenes_coco.append(c)
        coco_nombres.update(c["anotaciones_por_clase"])
        errores_etiquetas += c["errores"]
        for archivo in c["imagenes_con_anotacion"]:
            for p in imagenes:
                if p.name == archivo:
                    imagenes_con_etiqueta.add(str(p))
    if resumenes_coco:
        formatos.append("COCO")
        r["coco_nombres"] = dict(sorted(coco_nombres.items()))
        r["coco_categorias"] = sorted({n for c in resumenes_coco for n in c["categorias"].values()})
        r["coco_archivos"] = [c["archivo"] for c in resumenes_coco]

    r["formatos"] = formatos
    r["tiene_anotaciones"] = bool(formatos)
    r["imagenes_sin_etiqueta"] = [str(p) for p in imagenes if str(p) not in imagenes_con_etiqueta]
    r["n_etiquetas_invalidas"] = len(errores_etiquetas)
    r["ejemplos_etiquetas_invalidas"] = errores_etiquetas[:20]
    if not formatos:
        r["errores_fatales"].append(
            "No se encontraron anotaciones compatibles (YOLO .txt, VOC .xml, COCO .json) "
            "emparejadas con las imágenes. Hay que anotar este dataset antes de entrenar.")
    return r


# ----------------------------------------------------------------------------
# Presentación
# ----------------------------------------------------------------------------
def mostrar(r: dict, minimo: int) -> None:
    print("\n" + "=" * 68)
    print(f"DATASET: {r['nombre']}")
    print(f"Ruta   : {r['ruta']}")
    print("=" * 68)
    for e in r["errores_fatales"]:
        print(f"[ERROR] {e}")
    if "n_imagenes" not in r or r["n_imagenes"] == 0:
        return
    print(f"Imágenes encontradas       : {r['n_imagenes']}")
    print(f"Imágenes dañadas           : {len(r['imagenes_danadas'])}")
    for ej in r["imagenes_danadas"][:MAX_EJEMPLOS]:
        print(f"    - {ej}")
    if not r.get("tiene_anotaciones"):
        return
    print(f"Formato(s) de anotación    : {', '.join(r['formatos'])}")
    if "yolo_ids" in r:
        print(f"  YOLO - IDs de clase y nº de cajas: {r['yolo_ids']}")
        if r.get("yolo_nombres_en_yaml"):
            print(f"  YOLO - 'names' en el yaml del dataset: {r['yolo_nombres_en_yaml']}")
        else:
            print("  YOLO - no se encontró 'names' en un yaml: confirma qué clase es cada ID.")
    if "voc_nombres" in r:
        print(f"  VOC  - clases y nº de cajas: {r['voc_nombres']}")
    if "coco_nombres" in r:
        print(f"  COCO - clases y nº de cajas: {r['coco_nombres']}")
        print(f"  COCO - categorías declaradas: {r['coco_categorias']}")
    print(f"Imágenes sin etiqueta      : {len(r['imagenes_sin_etiqueta'])}")
    for ej in r["imagenes_sin_etiqueta"][:MAX_EJEMPLOS]:
        print(f"    - {ej}")
    print(f"Etiquetas inválidas        : {r['n_etiquetas_invalidas']}")
    for ej in r["ejemplos_etiquetas_invalidas"][:MAX_EJEMPLOS]:
        print(f"    - {ej}")
    if r.get("txt_sin_imagen"):
        print(f"Archivos .txt sin imagen asociada: {r['txt_sin_imagen']}")

    conteos = {}
    for k in ("yolo_ids", "voc_nombres", "coco_nombres"):
        conteos.update({str(c): n for c, n in r.get(k, {}).items()})
    for clase, n in conteos.items():
        if n < minimo:
            print(f"[AVISO] La clase '{clase}' tiene solo {n} anotaciones (< {minimo}): "
                  f"su evaluación podría ser poco fiable.")


def buscar_duplicados(resultados: list[dict]) -> dict:
    por_hash: dict[str, list[str]] = defaultdict(list)
    for r in resultados:
        for ruta, h in r.get("hashes", {}).items():
            por_hash[h].append(f"[{r['nombre']}] {ruta}")
    return {h: v for h, v in por_hash.items() if len(v) > 1}


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspección de solo lectura de los datasets")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sin-verificar-imagenes", action="store_true",
                    help="No abre cada imagen (más rápido, no detecta archivos dañados)")
    args = ap.parse_args()

    cfg = cargar_config(Path(r"C:/Ever/Ciclo 4/detector_herramientas/scripts/config.yaml"))
    ins = cfg.get("inspeccion", {})
    verificar = ins.get("verificar_imagenes", True) and not args.sin_verificar_imagenes
    minimo = int(ins.get("minimo_anotaciones_por_clase", 30))

    resultados = []
    for d in cfg["datasets"]:
        print(f"\nInspeccionando '{d['nombre']}' ...")
        resultados.append(inspeccionar_dataset(d["nombre"], Path(d["ruta"]), verificar))

    for r in resultados:
        mostrar(r, minimo)

    duplicados = buscar_duplicados(resultados)
    print("\n" + "=" * 68)
    print(f"DUPLICADOS EXACTOS (mismo contenido): {len(duplicados)} grupos")
    for v in list(duplicados.values())[:MAX_EJEMPLOS]:
        print("  - " + "\n    ".join(v))

    print("\n" + "=" * 68)
    print("RESUMEN")
    print("=" * 68)
    print(f"{'dataset':<20}{'imágenes':>10}{'formato':>14}{'sin etiqueta':>14}{'problemas':>11}")
    hay_fatal = False
    for r in resultados:
        fatal = bool(r["errores_fatales"])
        hay_fatal |= fatal
        print(f"{r['nombre']:<20}{r.get('n_imagenes', 0):>10}"
              f"{'/'.join(r.get('formatos', [])) or '-':>14}"
              f"{len(r.get('imagenes_sin_etiqueta', [])):>14}"
              f"{'SÍ' if fatal else 'no':>11}")

    # Informe JSON (sin la lista completa de hashes)
    salida = Path(cfg.get("proyecto", {}).get("directorio_resultados", "runs")) / "informes"
    salida.mkdir(parents=True, exist_ok=True)
    ligero = [{k: v for k, v in r.items() if k != "hashes"} for r in resultados]
    with open(salida / "inspeccion.json", "w", encoding="utf-8") as f:
        json.dump({"datasets": ligero, "grupos_duplicados": len(duplicados)}, f,
                  ensure_ascii=False, indent=2, default=list)
    print(f"\nInforme guardado en: {salida / 'inspeccion.json'}")

    if hay_fatal:
        print("\n[ATENCIÓN] Hay datasets con problemas que impiden entrenar. Revisa los [ERROR].")
        return 1
    print("\nInspección terminada. Envíame este resultado para diseñar la unificación de clases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
