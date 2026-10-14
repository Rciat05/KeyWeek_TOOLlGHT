"""
detectar_camara.py - Detección de herramientas en tiempo real con la cámara web.

Usa los mejores pesos entrenados (los busca solos) y la GPU si está disponible.

Teclas dentro de la ventana:
    q o ESC : salir
    s       : guardar una captura en runs/capturas/
    + / -   : subir / bajar el umbral de confianza en pasos de 0.05

Ejemplos:
    python scripts/detectar_camara.py
    python scripts/detectar_camara.py --conf 0.5
    python scripts/detectar_camara.py --conf 0.3 --conf-clase pinzas=0.7,martillo=0.5
    python scripts/detectar_camara.py --camara 1                (otra cámara)
    python scripts/detectar_camara.py --pesos weights/entrenamiento_v1_best.pt
    python scripts/detectar_camara.py --fuente video.mp4        (probar con un video)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def buscar_pesos(carpeta_pesos: Path, carpeta_runs: Path) -> Path | None:
    """Pesos más recientes: primero weights/*_best.pt, luego runs/entrenamientos/*/weights/best.pt."""
    candidatos = list(carpeta_pesos.glob("*_best.pt"))
    if not candidatos:
        candidatos = list(carpeta_runs.glob("entrenamientos/*/weights/best.pt"))
    return max(candidatos, key=lambda p: p.stat().st_mtime) if candidatos else None


def parsear_umbrales(texto: str, nombres: dict) -> dict[str, float]:
    """'pinzas=0.7,martillo=0.5' -> {'pinzas': 0.7, 'martillo': 0.5} (valida nombres y rango)."""
    umbrales: dict[str, float] = {}
    for par in filter(None, (x.strip() for x in (texto or "").split(","))):
        try:
            clase, valor = par.split("=")
            umbrales[clase.strip()] = float(valor)
        except ValueError:
            sys.exit(f"[ERROR] Formato inválido en --conf-clase: '{par}' (usa clase=umbral, p. ej. pinzas=0.7)")
    validas = set(nombres.values())
    for c, v in umbrales.items():
        if c not in validas:
            sys.exit(f"[ERROR] Clase desconocida en --conf-clase: '{c}'. Válidas: {sorted(validas)}")
        if not 0 < v < 1:
            sys.exit(f"[ERROR] El umbral de '{c}' debe estar entre 0 y 1 (recibido {v}).")
    return umbrales


PALETA = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
          (10, 249, 72), (23, 204, 146), (134, 219, 61)]


def dibujar_deteccion(cv2, img, xyxy, texto: str, color) -> None:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_texto = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 10
    cv2.rectangle(img, (x1, y_texto - th - 6), (x1 + tw + 6, y_texto + base), color, -1)
    cv2.putText(img, texto, (x1 + 3, y_texto - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def abrir_fuente(fuente: str, ancho: int, alto: int):
    import cv2
    if fuente.isdigit():
        indice = int(fuente)
        # CAP_DSHOW abre más rápido y es más estable en Windows
        cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(indice)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    else:
        cap = cv2.VideoCapture(fuente)
    return cap


def main() -> int:
    ap = argparse.ArgumentParser(description="Detección en tiempo real con cámara web")
    ap.add_argument("--pesos", help="ruta a best.pt (por defecto, los más recientes)")
    ap.add_argument("--camara", type=int, default=0, help="índice de la cámara (0, 1, 2...)")
    ap.add_argument("--fuente", help="ruta a un video en lugar de la cámara")
    ap.add_argument("--conf", type=float, default=0.40, help="umbral de confianza (0-1)")
    ap.add_argument("--conf-clase", default="",
                    help="umbral propio por clase, p. ej. pinzas=0.7,martillo=0.5 (el resto usa --conf)")
    ap.add_argument("--iou", type=float, default=0.50, help="umbral IoU de NMS")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dispositivo", default="auto", help="auto, cpu, 0, 1, ...")
    ap.add_argument("--ancho", type=int, default=1280)
    ap.add_argument("--alto", type=int, default=720)
    args = ap.parse_args()

    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"[ERROR] Falta {e.name}. Ejecuta: pip install -r requirements.txt")

    # Pesos
    pesos = Path(args.pesos) if args.pesos else buscar_pesos(Path("weights"), Path("runs"))
    if pesos is None or not pesos.exists():
        sys.exit("[ERROR] No encontré pesos entrenados. Indica la ruta con --pesos, por ejemplo:\n"
                 "  python scripts/detectar_camara.py --pesos runs/entrenamientos/<nombre>/weights/best.pt")
    print(f"Pesos: {pesos}")

    # Dispositivo
    from verificar_gpu import seleccionar_dispositivo
    try:
        dev_str, _ = seleccionar_dispositivo(args.dispositivo)
    except (RuntimeError, ValueError) as e:
        sys.exit(f"[ERROR] {e}")
    usa_gpu = dev_str.startswith("cuda")
    dispositivo = int(dev_str.split(":")[1]) if usa_gpu else "cpu"
    if usa_gpu:
        print(f"Dispositivo: GPU -> {torch.cuda.get_device_name(dispositivo)} (media precisión activada)")
    else:
        print("Dispositivo: CPU (irá más lento; ejecuta verificar_gpu.py si esperabas usar la GPU)")

    modelo = YOLO(str(pesos))
    nombres = modelo.names
    print(f"Clases del modelo: {list(nombres.values())}")
    umbrales = parsear_umbrales(args.conf_clase, nombres)
    if umbrales:
        print(f"Umbrales propios por clase: {umbrales}")

    # Cámara / video
    fuente = args.fuente if args.fuente else str(args.camara)
    cap = abrir_fuente(fuente, args.ancho, args.alto)
    if not cap.isOpened():
        sys.exit(f"[ERROR] No se pudo abrir la fuente '{fuente}'.\n"
                 "  - Cierra otras apps que usen la cámara (Zoom, Teams, navegador).\n"
                 "  - Prueba otro índice: --camara 1 o --camara 2.\n"
                 "  - Revisa Configuración > Privacidad > Cámara y permite el acceso a apps de escritorio.")

    # Calentamiento (la primera inferencia es lenta)
    ok, frame = cap.read()
    if not ok:
        sys.exit("[ERROR] La cámara se abrió pero no entrega imágenes.")
    modelo.predict(frame, imgsz=args.imgsz, device=dispositivo, half=usa_gpu, verbose=False)

    conf = args.conf
    ventana = "Detector de herramientas  (q: salir | s: captura | +/-: confianza)"
    carpeta_capturas = Path("runs") / "capturas"
    fps = 0.0
    print("\nCámara lista. Pulsa 'q' o ESC en la ventana para salir.")

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("Fin de la fuente de video o la cámara dejó de responder.")
                break

            # Se infiere con el umbral más bajo y se filtra por clase al dibujar
            umbral_pred = min([conf] + list(umbrales.values()))
            resultado = modelo.predict(frame, conf=umbral_pred, iou=args.iou, imgsz=args.imgsz,
                                       device=dispositivo, half=usa_gpu, verbose=False)[0]
            anotado = frame.copy()
            detecciones = []
            if len(resultado.boxes):
                for xyxy, c, k in zip(resultado.boxes.xyxy.cpu().numpy(), resultado.boxes.conf.cpu().numpy(),
                                      resultado.boxes.cls.cpu().numpy()):
                    nombre = nombres[int(k)]
                    if float(c) < umbrales.get(nombre, conf):
                        continue
                    texto = f"{nombre.replace('_', ' ')} {float(c) * 100:.0f}%"
                    dibujar_deteccion(cv2, anotado, xyxy, texto, PALETA[int(k) % len(PALETA)])
                    detecciones.append(texto)

            # Lista de detecciones en pantalla
            extra = "  (propios: " + ", ".join(f"{c}>={v:.2f}" for c, v in umbrales.items()) + ")" if umbrales else ""
            lineas = [f"FPS: {fps:4.1f}   confianza min: {conf:.2f}   {'GPU' if usa_gpu else 'CPU'}{extra}"]
            lineas += detecciones
            for k, texto in enumerate(lineas):
                y = 28 + 26 * k
                cv2.putText(anotado, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(anotado, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(ventana, anotado)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("s"):
                carpeta_capturas.mkdir(parents=True, exist_ok=True)
                destino = carpeta_capturas / f"captura_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(destino), anotado)
                print(f"Captura guardada: {destino}")
            elif tecla in (ord("+"), ord("=")):
                conf = min(0.95, round(conf + 0.05, 2))
            elif tecla in (ord("-"), ord("_")):
                conf = max(0.05, round(conf - 0.05, 2))
            if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                break  # cerraste la ventana con la X

            dt = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] Sin memoria de GPU. Cierra otras apps o usa --imgsz 480.")
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Detección finalizada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
