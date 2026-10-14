"""
capturar_imagenes.py - Captura imágenes propias con la webcam para mejorar el modelo.

Guarda las fotos en  datasets_propios/<sesion>/<clase>/  y crea
datasets_propios/classes.txt con las clases de config.yaml EN ORDEN (los IDs coinciden
con los del modelo final), listo para cargarlo en una herramienta de anotación.

Teclas en la ventana:
    ESPACIO : guardar una imagen
    a       : activar/desactivar captura automática (una imagen cada --intervalo segundos)
    q / ESC : salir

IMÁGENES DE FONDO (sin ninguna herramienta): enseñan al modelo qué NO es una herramienta
y reducen las falsas detecciones. Con --fondo se guardan junto a un .txt vacío (YOLO lo
interpreta como "aquí no hay nada"), así que NO hay que anotar nada:
    python scripts/capturar_imagenes.py --fondo --sesion entrenamiento1 --intervalo 2

Ejemplos:
    python scripts/capturar_imagenes.py --clase pistola_silicona --sesion entrenamiento1
    python scripts/capturar_imagenes.py --clase guantes --sesion entrenamiento1
    python scripts/capturar_imagenes.py --clase martillo --sesion prueba_final   (TEST aparte)

Consejos para fotos útiles:
  * Cambia entre fotos: ángulo, distancia, fondo, luz, mano sosteniéndola / sobre la mesa.
  * Que la herramienta ocupe entre un 20 % y un 60 % de la imagen, y también alguna más lejos.
  * Si en la imagen hay OTRAS herramientas de tus 8 clases, hay que anotarlas también.
  * La sesión 'prueba_final' (otro día, otro fondo) NO se usa para entrenar: es tu test honesto.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("[ERROR] Falta pyyaml. Ejecuta: pip install -r requirements.txt")


def clases_desde_config(ruta: Path) -> list[str]:
    if not ruta.exists():
        sys.exit(f"[ERROR] No existe {ruta}")
    cfg = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    clases = cfg.get("clases")
    if not clases:
        sys.exit("[ERROR] config.yaml no tiene 'clases'.")
    return list(clases)


def nombre_imagen(clase: str) -> str:
    return f"{clase}_{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3] + ".jpg"


def contar_existentes(carpeta: Path) -> int:
    return len(list(carpeta.glob("*.jpg"))) if carpeta.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Captura de imágenes con la webcam")
    ap.add_argument(
    "--config",
    default=r"C:/Ever/Ciclo 4/detector_herramientas/scripts/config.yaml"
)
    ap.add_argument("--clase", help="clase que vas a fotografiar (de config.yaml)")
    ap.add_argument("--fondo", action="store_true",
                    help="captura imágenes de FONDO (sin herramientas) con etiqueta vacía")
    ap.add_argument("--sesion", default="sesion1", help="nombre de la sesión (p. ej. entrenamiento1, prueba_final)")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    ap.add_argument("--intervalo", type=float, default=1.0, help="segundos entre fotos en modo automático")
    ap.add_argument("--salida", default="datasets_propios")
    args = ap.parse_args()

    clases = clases_desde_config(Path(args.config))
    if args.fondo:
        args.clase = "fondo"
    elif not args.clase:
        sys.exit("[ERROR] Indica --clase <nombre> o --fondo.")
    elif args.clase not in clases:
        sys.exit(f"[ERROR] La clase '{args.clase}' no está en config.yaml. Opciones: {clases}")

    try:
        import cv2
    except ImportError:
        sys.exit("[ERROR] Falta opencv-python. Ejecuta: pip install -r requirements.txt")

    raiz = Path(args.salida)
    carpeta = raiz / args.sesion / args.clase
    carpeta.mkdir(parents=True, exist_ok=True)
    fichero_clases = raiz / "classes.txt"
    if not fichero_clases.exists():
        fichero_clases.write_text("\n".join(clases) + "\n", encoding="utf-8")

    cap = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        sys.exit(f"[ERROR] No se pudo abrir la cámara {args.camara}. Cierra otras apps que la usen "
                 f"o prueba --camara 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)

    guardadas = 0
    previas = contar_existentes(carpeta)
    auto = False
    ultimo = 0.0
    titulo = "Captura (ESPACIO: guardar | a: auto | q: salir)"
    print(f"Guardando en: {carpeta}  (ya había {previas} imágenes)")
    print(f"IDs de clase para anotar, en orden: {dict(enumerate(clases))}")

    def guardar(frame) -> None:
        nonlocal guardadas
        destino = carpeta / nombre_imagen(args.clase)
        cv2.imwrite(str(destino), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if args.fondo:
            destino.with_suffix(".txt").write_text("", encoding="utf-8")   # etiqueta vacía = fondo
        guardadas += 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("La cámara dejó de responder.")
                break

            ahora = time.perf_counter()
            if auto and ahora - ultimo >= args.intervalo:
                guardar(frame)
                ultimo = ahora

            vista = frame.copy()
            textos = [("FONDO: sin ninguna herramienta en la imagen" if args.fondo else f"Clase: {args.clase}")
                      + f"   Sesion: {args.sesion}",
                      f"Guardadas ahora: {guardadas}   Total en carpeta: {previas + guardadas}",
                      "AUTO: ACTIVADO" if auto else "AUTO: apagado"]
            for k, t in enumerate(textos):
                y = 28 + 26 * k
                cv2.putText(vista, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(vista, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255) if auto else (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(titulo, vista)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == 32:
                guardar(frame)
            elif tecla == ord("a"):
                auto = not auto
                ultimo = ahora
            if cv2.getWindowProperty(titulo, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nImágenes guardadas en esta ejecución: {guardadas}  ->  {carpeta}")
    if args.fondo:
        print("Imágenes de fondo listas (con .txt vacíos): no hay que anotar nada. Añade la carpeta a "
              "config.yaml como dataset con 'fondo: true'.")
    else:
        print("Siguiente paso: anotar las cajas (X-AnyLabeling, CVAT o LabelImg, exportando en formato YOLO).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
