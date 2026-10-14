"""
preparar_datasets.py - Unifica los datasets originales (YOLO) en uno solo.

  * NO modifica los datasets originales: copia imágenes y etiquetas a
    la carpeta de salida (por defecto "datos_unificados").
  * Reasigna los IDs de clase de cada dataset a los IDs finales de config.yaml
    mediante 'mapa_clases' (ID del dataset -> nombre de clase final).
  * Evita colisiones de nombres: <dataset>__<nombre>__<hash8>.ext
  * Excluye (y lo informa) imágenes ilegibles, sin etiqueta, con etiqueta
    inválida o ambigua, duplicadas exactas, o con clases sin mapear.
  * IDs sin mapear: 'ids_no_mapeados' = excluir_imagen (por defecto) | ignorar_cajas |
    error. Puede fijarse global (sección preparacion) o por dataset.
  * Opcional por dataset: 'maximo_imagenes: N' usa una muestra aleatoria reproducible
    (útil para equilibrar clases con muchísimas imágenes frente a otras con pocas).
  * Los polígonos de segmentación se convierten a su caja envolvente (y se cuentan).

Paso 1 (obligatorio, no escribe nada):
    python scripts/preparar_datasets.py --revisar
  Muestra los IDs de clase reales de cada dataset y una SUGERENCIA de
  'mapa_clases' para pegar en config.yaml (verifícala siempre).

Paso 2:
    python scripts/preparar_datasets.py
    python scripts/preparar_datasets.py --rehacer    (borra y regenera la salida)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspeccionar_datasets import (  # noqa: E402
    EXT_IMAGEN, TXT_IGNORADOS, cargar_config, hash_archivo, imagen_legible, nombres_en_yaml,
)

# Nombres alternativos (solo para la SUGERENCIA de --revisar; nunca se aplican solos)
ALIAS = {
    "martillo": {"martillo", "hammer", "hammers"},
    "pinzas": {"pinzas", "pinza", "pliers", "plier", "alicate", "alicates"},
    "cutter": {"cutter", "cutters", "utility_knife", "box_cutter", "cuchilla"},
    "cinta_metrica": {"cinta_metrica", "tape_measure", "measuring_tape", "tape", "flexometro"},
    "guantes": {"guantes", "guante", "gloves", "glove"},
    "pistola_silicona": {"pistola_silicona", "silicone_gun", "glue_gun", "hot_glue_gun",
                         "caulking_gun", "caulk_gun", "pistola_de_silicona"},
    "pie_de_rey": {"pie_de_rey", "caliper", "calipers", "vernier_caliper", "vernier",
                   "calibrador", "calibre"},
    "casco": {"casco", "cascos", "helmet", "hardhat", "hard_hat", "safety_helmet"},
}


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def normalizar(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def indexar(ruta: Path):
    """Devuelve (imagenes, indice_txt por stem, yamls)."""
    archivos = [p for p in ruta.rglob("*") if p.is_file()]
    imagenes = sorted(p for p in archivos if p.suffix.lower() in EXT_IMAGEN)
    indice: dict[str, list[Path]] = defaultdict(list)
    for p in archivos:
        if p.suffix.lower() == ".txt" and p.name.lower() not in TXT_IGNORADOS:
            indice[p.stem].append(p)
    yamls = [p for p in archivos if p.suffix.lower() in (".yaml", ".yml")]
    return imagenes, indice, yamls


def buscar_etiqueta(img: Path, indice: dict) -> tuple[Path | None, bool]:
    """Devuelve (txt, ambiguo). Prefiere la estructura images/ -> labels/."""
    cands = indice.get(img.stem, [])
    if not cands:
        return None, False
    if len(cands) == 1:
        return cands[0], False
    partes = list(img.parts)
    for i in range(len(partes) - 1, -1, -1):
        if partes[i].lower() == "images":
            alt = Path(*partes[:i], "labels", *partes[i + 1:]).with_suffix(".txt")
            if alt in cands:
                return alt, False
            break
    mismo = [c for c in cands if c.parent == img.parent]
    if len(mismo) == 1:
        return mismo[0], False
    return None, True


def leer_cajas(txt: Path):
    """Devuelve (cajas[(id,x,y,w,h)], errores, poligonos_convertidos)."""
    cajas, errores, poligonos = [], [], 0
    try:
        lineas = txt.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return [], [f"{txt.name}: ilegible ({e})"], 0
    for n, linea in enumerate(lineas, 1):
        p = linea.split()
        if not p:
            continue
        try:
            cid = int(p[0])
            v = [float(x) for x in p[1:]]
        except ValueError:
            errores.append(f"{txt.name}:{n} no numérico")
            continue
        if cid < 0 or not all(0 <= x <= 1 for x in v):
            errores.append(f"{txt.name}:{n} ID negativo o coordenadas fuera de [0,1]")
            continue
        if len(v) == 4:
            x, y, w, h = v
        elif len(v) >= 6 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            poligonos += 1
        else:
            errores.append(f"{txt.name}:{n} número de valores inesperado ({len(v)})")
            continue
        if w <= 0 or h <= 0:
            errores.append(f"{txt.name}:{n} caja con tamaño nulo")
            continue
        cajas.append((cid, x, y, w, h))
    return cajas, errores, poligonos


# ----------------------------------------------------------------------------
# Paso 1: revisar
# ----------------------------------------------------------------------------
def revisar(cfg: dict) -> int:
    inverso = {normalizar(a): c for c, alias in ALIAS.items() for a in alias}
    bloques = []
    for d in cfg["datasets"]:
        ruta = Path(d["ruta"])
        print("\n" + "=" * 68)
        print(f"DATASET: {d['nombre']}   ({ruta})")
        print("=" * 68)
        if not ruta.is_dir():
            print("[ERROR] La ruta no existe.")
            continue
        imagenes, indice, yamls = indexar(ruta)
        nombres = nombres_en_yaml(yamls)
        ids: Counter = Counter()
        sin_etiqueta = 0
        for img in imagenes:
            txt, _ = buscar_etiqueta(img, indice)
            if txt is None:
                sin_etiqueta += 1
                continue
            cajas, _, _ = leer_cajas(txt)
            ids.update(c[0] for c in cajas)
        print(f"Imágenes: {len(imagenes)}  |  sin etiqueta: {sin_etiqueta}")
        print(f"'names' del yaml del dataset: {nombres if nombres else 'NO ENCONTRADO'}")
        sugerido = {}
        for cid, n in sorted(ids.items()):
            nom = nombres[cid] if nombres and cid < len(nombres) else None
            clase = inverso.get(normalizar(nom)) if nom is not None else None
            print(f"  ID {cid}: {n} cajas | nombre en yaml: {nom!r} | sugerencia: {clase or '???'}")
            sugerido[cid] = clase or "COMPLETAR"
        bloques.append((d["nombre"], sugerido))

    print("\n" + "=" * 68)
    print("SUGERENCIA para config.yaml (VERIFÍCALA; sustituye COMPLETAR):")
    print("=" * 68)
    for nombre, sug in bloques:
        print(f"  # {nombre}")
        print("    mapa_clases:")
        for cid, clase in sug.items():
            print(f"      {cid}: {clase}")
    print("\nClases finales válidas:", cfg["clases"])
    return 0


# ----------------------------------------------------------------------------
# Paso 2: preparar
# ----------------------------------------------------------------------------
def validar_mapas(cfg: dict) -> dict[str, dict[int, int]]:
    clases = cfg["clases"]
    if len(set(clases)) != len(clases):
        sys.exit("[ERROR] Hay clases repetidas en 'clases' de config.yaml.")
    destino = {c: i for i, c in enumerate(clases)}
    mapas, problemas = {}, []
    for d in cfg["datasets"]:
        mc = d.get("mapa_clases") or {}
        if not mc and not d.get("fondo"):
            problemas.append(f"'{d['nombre']}': falta 'mapa_clases' (ejecuta --revisar). "
                             f"Si son imágenes de fondo sin herramientas, añade 'fondo: true'.")
            continue
        m = {}
        for k, v in mc.items():
            if v not in destino:
                problemas.append(f"'{d['nombre']}': la clase '{v}' no está en 'clases' de config.yaml.")
            else:
                m[int(k)] = destino[v]
        mapas[d["nombre"]] = m
    if problemas:
        print("[ERROR] Problemas en config.yaml:")
        for p in problemas:
            print("  -", p)
        sys.exit(1)
    return mapas


def preparar(cfg: dict, rehacer: bool) -> int:
    clases = cfg["clases"]
    mapas = validar_mapas(cfg)
    prep = cfg.get("preparacion", {})
    politica = prep.get("ids_no_mapeados", "excluir_imagen")
    salida = Path(prep.get("directorio_salida", "datos_unificados"))

    if salida.exists() and any(salida.iterdir()):
        if rehacer and (salida / "manifiesto.csv").exists():
            shutil.rmtree(salida)
        else:
            sys.exit(f"[ERROR] '{salida}' ya existe y no está vacía. "
                     f"Usa --rehacer para regenerarla (solo borra esa carpeta generada).")
    (salida / "images").mkdir(parents=True, exist_ok=True)
    (salida / "labels").mkdir(parents=True, exist_ok=True)

    vistos: dict[str, str] = {}
    manifiesto, informe = [], {}

    for d in cfg["datasets"]:
        nombre, ruta = d["nombre"], Path(d["ruta"])
        if not ruta.is_dir():
            sys.exit(f"[ERROR] La ruta de '{nombre}' no existe: {ruta}")
        imagenes, indice, _ = indexar(ruta)
        if not imagenes:
            sys.exit(f"[ERROR] El dataset '{nombre}' no contiene imágenes.")
        tope = d.get("maximo_imagenes")
        if tope and len(imagenes) > int(tope):
            import random
            semilla = cfg.get("proyecto", {}).get("semilla", 42)
            print(f"[AVISO] '{nombre}': se usará una muestra aleatoria de {int(tope)} de "
                  f"{len(imagenes)} imágenes (maximo_imagenes, semilla {semilla}).")
            imagenes = sorted(random.Random(semilla).sample(imagenes, int(tope)))
        mapa = mapas[nombre]
        es_fondo = bool(d.get("fondo", False))   # imágenes sin herramientas (etiqueta vacía o ausente)
        politica_d = d.get("ids_no_mapeados", politica)
        if politica_d not in ("excluir_imagen", "ignorar_cajas", "error"):
            sys.exit(f"[ERROR] '{nombre}': ids_no_mapeados='{politica_d}' no válido "
                     f"(usa excluir_imagen, ignorar_cajas o error).")
        ignoradas: Counter = Counter()
        ex: Counter = Counter()
        cajas_clase: Counter = Counter()
        poligonos = usadas = 0
        ejemplos: dict[str, list[str]] = defaultdict(list)

        def excluir(causa: str, detalle: str):
            ex[causa] += 1
            if len(ejemplos[causa]) < 5:
                ejemplos[causa].append(detalle)

        for img in imagenes:
            if not imagen_legible(img):
                excluir("imagen_ilegible", str(img))
                continue
            txt, ambiguo = buscar_etiqueta(img, indice)
            if ambiguo:
                excluir("etiqueta_ambigua", str(img))
                continue
            if txt is None and not es_fondo:
                excluir("sin_etiqueta", str(img))
                continue
            cajas, errores, pol = leer_cajas(txt) if txt is not None else ([], [], 0)
            if errores:
                excluir("etiqueta_invalida", errores[0])
                continue
            if es_fondo and cajas:
                excluir("fondo_con_cajas", f"{img.name}: un dataset 'fondo' no debe tener cajas")
                continue
            ids_ajenos = {c[0] for c in cajas if c[0] not in mapa}
            if ids_ajenos:
                if politica_d == "error":
                    sys.exit(f"[ERROR] '{nombre}': IDs sin mapear {sorted(ids_ajenos)} en {txt}")
                if politica_d == "excluir_imagen":
                    excluir("id_sin_mapear", f"{img.name} (IDs {sorted(ids_ajenos)})")
                    continue
                # ignorar_cajas: se descartan solo esas cajas; si no queda ninguna, se excluye la imagen
                propias = [c for c in cajas if c[0] in mapa]
                if not propias:
                    excluir("solo_ids_sin_mapear", f"{img.name} (IDs {sorted(ids_ajenos)})")
                    continue
                for c in cajas:
                    if c[0] not in mapa:
                        ignoradas[c[0]] += 1
                cajas = propias
            h = hash_archivo(img)
            if h in vistos:
                excluir("duplicada_exacta", f"{img.name} == {vistos[h]}")
                continue
            vistos[h] = f"[{nombre}] {img.name}"

            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", img.stem)
            nuevo = f"{nombre}__{stem}__{h[:8]}"
            shutil.copy2(img, salida / "images" / f"{nuevo}{img.suffix.lower()}")
            lineas = [f"{mapa[c]} {x:.6f} {y:.6f} {w:.6f} {hh:.6f}" for c, x, y, w, hh in cajas]
            (salida / "labels" / f"{nuevo}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""),
                                                          encoding="utf-8")
            for c in cajas:
                cajas_clase[clases[mapa[c[0]]]] += 1
            poligonos += pol
            usadas += 1
            manifiesto.append({"nombre_nuevo": nuevo, "extension": img.suffix.lower(),
                               "dataset": nombre, "stem_original": img.stem,
                               "ruta_original": str(img), "n_cajas": len(cajas)})

        informe[nombre] = {"imagenes_total": len(imagenes), "imagenes_usadas": usadas,
                           "excluidas": dict(ex), "ejemplos_exclusion": dict(ejemplos),
                           "cajas_por_clase": dict(cajas_clase), "poligonos_convertidos": poligonos,
                           "cajas_ignoradas_por_id": {str(k): v for k, v in ignoradas.items()}}

    with open(salida / "manifiesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["nombre_nuevo", "extension", "dataset", "stem_original",
                                          "ruta_original", "n_cajas"])
        w.writeheader()
        w.writerows(manifiesto)

    # --- Resumen ---
    print("\n" + "=" * 68)
    print("RESUMEN DE LA UNIFICACIÓN")
    print("=" * 68)
    total_cajas: Counter = Counter()
    for nombre, r in informe.items():
        print(f"\n{nombre}: {r['imagenes_usadas']}/{r['imagenes_total']} imágenes usadas")
        for causa, n in r["excluidas"].items():
            print(f"   excluidas por {causa}: {n}")
            for e in r["ejemplos_exclusion"][causa]:
                print(f"       - {e}")
        if r["cajas_ignoradas_por_id"]:
            print(f"   [AVISO] cajas ignoradas (IDs sin mapear, imagen conservada): {r['cajas_ignoradas_por_id']}")
        if r["poligonos_convertidos"]:
            print(f"   [AVISO] {r['poligonos_convertidos']} polígonos convertidos a caja envolvente")
        print(f"   cajas por clase: {r['cajas_por_clase']}")
        total_cajas.update(r["cajas_por_clase"])

    print("\nTOTAL de cajas por clase final:")
    for c in clases:
        n = total_cajas.get(c, 0)
        marca = "   <-- [ERROR] sin ejemplos" if n == 0 else ("   <-- [AVISO] pocas" if n < 50 else "")
        print(f"  {c:<18}{n:>7}{marca}")

    destino = Path(cfg.get("proyecto", {}).get("directorio_resultados", "runs")) / "informes"
    destino.mkdir(parents=True, exist_ok=True)
    with open(destino / "preparacion.json", "w", encoding="utf-8") as f:
        json.dump(informe, f, ensure_ascii=False, indent=2)
    print(f"\nImágenes unificadas : {len(manifiesto)}  ->  {salida}")
    print(f"Informe             : {destino / 'preparacion.json'}")
    if any(total_cajas.get(c, 0) == 0 for c in clases):
        print("\n[ATENCIÓN] Hay clases configuradas sin ninguna anotación: revisa 'mapa_clases'.")
        return 1
    print("\nSiguiente paso: python scripts/dividir_datasets.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Unificación de datasets YOLO")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--revisar", action="store_true", help="Solo muestra IDs y sugiere mapa_clases")
    ap.add_argument("--rehacer", action="store_true", help="Regenera la carpeta de salida")
    args = ap.parse_args()
    cfg = cargar_config(Path(r"C:/Ever/Ciclo 4/detector_herramientas/scripts/config.yaml"))
    if "clases" not in cfg:
        sys.exit("[ERROR] config.yaml no tiene la sección 'clases'.")
    return revisar(cfg) if args.revisar else preparar(cfg, args.rehacer)


if __name__ == "__main__":
    sys.exit(main())
