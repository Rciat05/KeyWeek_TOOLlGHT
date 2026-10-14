"""
dashboard.py - Dashboard del detector de herramientas.

  * Cámara + modelo YOLO: cada herramienta detectada enciende SU foco (numeradas 1 a 8)
    mediante un número enviado por Bluetooth a un Arduino.
  * Voz: "¿dónde está el martillo?" -> reconoce "martillo", lo relaciona con su clase y
    envía el mismo número al Arduino (el foco se enciende unos segundos).
  * Panel de prueba de focos, estado de la conexión, registro de eventos y modo simulado
    (sin Arduino) para probarlo todo sin hardware.

Uso:
    python scripts/dashboard.py
    python scripts/dashboard.py --pesos weights/entrenamiento_v3_best.pt
    python scripts/dashboard.py --sin-camara          (solo Bluetooth y voz; no usa la GPU)
    python scripts/dashboard.py --fuente video.mp4    (probar con un video en vez de la cámara)
    python scripts/dashboard.py --dispositivo cpu     (sin GPU)

NOTA: si hay un entrenamiento en curso, NO lances la detección en la GPU (4 GB de VRAM):
podrías hacer fallar el entrenamiento. Mientras tanto usa --sin-camara.

Los ajustes (numeración de herramientas, puerto, idioma de voz, tiempos...) se pueden fijar en
la sección 'dashboard' de config.yaml; todo tiene valores por defecto razonables.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402

try:
    import yaml
    from PIL import Image, ImageTk
except ImportError as e:
    sys.exit(f"[ERROR] Falta {e.name}. Ejecuta: pip install -r requirements.txt")

from arduino_bt import MAX_FOCOS, SIMULADO, EnlaceArduino, listar_puertos  # noqa: E402
from comandos_voz import EscuchaVoz, alias_para, interpretar  # noqa: E402
from logica_focos import LogicaFocos  # noqa: E402

CLASES_POR_DEFECTO = ["martillo", "pinzas", "cutter", "cinta_metrica", "guantes",
                      "pistola_silicona", "pie_de_rey", "casco"]

# Colores
FONDO, PANEL, TEXTO, TENUE = "#16181d", "#22252e", "#eceef4", "#8b90a0"
VERDE, AMBAR, AZUL, ROJO, GRIS = "#34d399", "#fbbf24", "#60a5fa", "#f87171", "#3b3f4c"


# ----------------------------------------------------------------------------
# Ajustes
# ----------------------------------------------------------------------------
def cargar_ajustes(ruta: Path) -> dict:
    cfg = {}
    if ruta.exists():
        cfg = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    clases = cfg.get("clases") or CLASES_POR_DEFECTO
    d = cfg.get("dashboard") or {}
    numeracion = d.get("numeracion") or {c: i for i, c in enumerate(clases, 1)}

    faltan = [c for c in clases if c not in numeracion]
    if faltan:
        sys.exit(f"[ERROR] dashboard.numeracion no incluye las clases: {faltan}")
    numeros = [numeracion[c] for c in clases]
    if len(set(numeros)) != len(numeros) or any(not 1 <= int(n) <= MAX_FOCOS for n in numeros):
        sys.exit(f"[ERROR] La numeración debe usar números distintos entre 1 y {MAX_FOCOS}: {numeracion}")

    bt, voz, det = d.get("bluetooth") or {}, d.get("voz") or {}, d.get("deteccion") or {}
    return {
        "clases": clases,
        "numeracion": {c: int(numeracion[c]) for c in clases},
        "puerto": str(bt.get("puerto") or ""),
        "baudios": int(bt.get("baudios", 9600)),
        "idioma": voz.get("idioma", "es-MX"),
        "duracion_voz_s": float(voz.get("duracion_foco_s", 8)),
        "microfono": voz.get("microfono"),
        "alias": alias_para(clases, voz.get("alias")),
        "confianza": float(det.get("confianza", 0.40)),
        "logica": {"confirmacion_s": float(det.get("confirmacion_s", 0.3)),
                   "liberacion_s": float(det.get("liberacion_s", 1.0)),
                   "keepalive_s": 0.5},
    }


# ----------------------------------------------------------------------------
# Aplicación
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root: tk.Tk, aj: dict, args: argparse.Namespace) -> None:
        self.root, self.aj, self.args = root, aj, args
        self.numeracion: dict[str, int] = aj["numeracion"]
        self.nombre_de = {n: c for c, n in self.numeracion.items()}
        self.enlace = EnlaceArduino()
        self.logica = LogicaFocos(**aj["logica"])
        self.cola: queue.Queue = queue.Queue()
        self.parar = threading.Event()
        self.lock = threading.Lock()
        self.conf = aj["confianza"]
        self.frame = None
        self.dets: tuple[float, list[tuple[str, float]]] = (0.0, [])
        self.fps = 0.0
        self._foto = None
        self.voz_ultima_orden: dict[int, str] = {}
        self.voz = EscuchaVoz(lambda t: self.cola.put(("voz_texto", t)),
                              lambda s: self.cola.put(("voz_estado", s)),
                              idioma=aj["idioma"], microfono=aj["microfono"])

        self._estilo()
        self._construir()
        self._refrescar_puertos()
        self.log("Dashboard iniciado. Conecta el Bluetooth (o elige SIMULADO) para enviar señales.")
        self.log("Numeración: " + ", ".join(f"{n}={c}" for c, n in sorted(self.numeracion.items(), key=lambda x: x[1])))

        if args.sin_camara:
            self.estado_camara.set("Cámara: desactivada (--sin-camara)")
        else:
            threading.Thread(target=self._bucle_camara, daemon=True, name="camara").start()

        self.root.after(33, self._pintar_video)
        self.root.after(100, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._cerrar)

    # ---------------- Interfaz ----------------
    def _estilo(self) -> None:
        self.root.title("Detector de herramientas - Dashboard")
        self.root.configure(bg=FONDO)
        alto = min(820, max(640, self.root.winfo_screenheight() - 90))
        self.root.geometry(f"1340x{alto}")
        self.root.minsize(1100, 620)
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=FONDO, foreground=TEXTO, fieldbackground=PANEL, bordercolor=GRIS)
        s.configure("TFrame", background=FONDO)
        s.configure("Panel.TLabelframe", background=PANEL, bordercolor=GRIS)
        s.configure("Panel.TLabelframe.Label", background=PANEL, foreground=AZUL, font=("Segoe UI", 10, "bold"))
        s.configure("TLabel", background=PANEL, foreground=TEXTO)
        s.configure("Tenue.TLabel", background=PANEL, foreground=TENUE)
        s.configure("TButton", background=GRIS, foreground=TEXTO, padding=5, borderwidth=0)
        s.map("TButton", background=[("active", "#4b5063")])
        s.configure("Acento.TButton", background="#2563eb", foreground="white")
        s.map("Acento.TButton", background=[("active", "#3b7bff")])
        s.configure("TCheckbutton", background=PANEL, foreground=TEXTO)
        s.map("TCheckbutton", background=[("active", PANEL)])
        s.configure("TCombobox", fieldbackground=PANEL, background=GRIS, foreground=TEXTO)
        s.configure("TEntry", fieldbackground="#2d313d", foreground=TEXTO)
        s.configure("Horizontal.TScale", background=PANEL)

    def _construir(self) -> None:
        r = self.root
        r.columnconfigure(0, weight=1)
        r.columnconfigure(1, weight=0)
        r.rowconfigure(1, weight=1)   # el vídeo ocupa el espacio sobrante

        # Barra superior de estado
        barra = tk.Frame(r, bg=FONDO)
        barra.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 2))
        tk.Label(barra, text="Detector de herramientas", bg=FONDO, fg=TEXTO,
                 font=("Segoe UI", 16, "bold")).pack(side="left")
        self.estado_camara = tk.StringVar(value="Cámara: iniciando...")
        self.estado_bt = tk.StringVar(value="Bluetooth: desconectado")
        self.estado_voz = tk.StringVar(value="Voz: desactivada")
        for var in (self.estado_voz, self.estado_bt, self.estado_camara):
            tk.Label(barra, textvariable=var, bg=PANEL, fg=TEXTO, padx=10, pady=3,
                     font=("Segoe UI", 9)).pack(side="right", padx=4)

        # Vídeo
        self.canvas = tk.Canvas(r, bg="black", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)

        # Panel derecho
        der = tk.Frame(r, bg=FONDO)
        der.grid(row=1, column=1, rowspan=3, sticky="n", padx=(5, 10), pady=5)
        self._panel_herramientas(der)
        self._panel_bluetooth(der)
        self._panel_voz(der)
        self._panel_deteccion(r)     # barra delgada bajo el vídeo

        # Registro
        reg = ttk.Labelframe(r, text="Registro", style="Panel.TLabelframe")
        reg.grid(row=3, column=0, sticky="ew", padx=(10, 5), pady=(0, 10))
        self.txt = tk.Text(reg, height=6, bg="#101216", fg="#c9ced9", relief="flat", font=("Consolas", 9),
                           state="disabled", wrap="word")
        sb = ttk.Scrollbar(reg, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        sb.pack(side="right", fill="y", pady=4)

    def _panel_herramientas(self, padre) -> None:
        p = ttk.Labelframe(padre, text="Herramientas y focos", style="Panel.TLabelframe")
        p.pack(fill="x", pady=(0, 8))
        self.filas: dict[int, dict] = {}
        for clase, n in sorted(self.numeracion.items(), key=lambda x: x[1]):
            f = tk.Frame(p, bg=PANEL)
            f.pack(fill="x", padx=8, pady=0)
            lampara = tk.Canvas(f, width=18, height=18, bg=PANEL, highlightthickness=0)
            ovalo = lampara.create_oval(2, 2, 16, 16, fill=GRIS, outline="")
            lampara.pack(side="left")
            tk.Label(f, text=str(n), bg=PANEL, fg=AMBAR, width=2, font=("Segoe UI", 13, "bold")).pack(side="left")
            tk.Label(f, text=clase.replace("_", " "), bg=PANEL, fg=TEXTO, width=17, anchor="w",
                     font=("Segoe UI", 10)).pack(side="left")
            estado = tk.Label(f, text="—", bg=PANEL, fg=TENUE, anchor="w", font=("Segoe UI", 9))
            estado.pack(side="left", fill="x", expand=True)
            self.filas[n] = {"lampara": lampara, "ovalo": ovalo, "estado": estado}
        self.lbl_ultimo = tk.Label(p, text="Último mensaje enviado: —", bg=PANEL, fg=TENUE, anchor="w",
                                   font=("Consolas", 9))
        self.lbl_ultimo.pack(fill="x", padx=8, pady=(4, 6))

    def _panel_bluetooth(self, padre) -> None:
        p = ttk.Labelframe(padre, text="Bluetooth / Arduino", style="Panel.TLabelframe")
        p.pack(fill="x", pady=(0, 8))
        fila = tk.Frame(p, bg=PANEL)
        fila.pack(fill="x", padx=8, pady=(6, 2))
        self.cmb_puerto = ttk.Combobox(fila, width=34)
        self.cmb_puerto.pack(side="left", fill="x", expand=True)
        ttk.Button(fila, text="↻", width=3, command=self._refrescar_puertos).pack(side="left", padx=(4, 0))
        fila2 = tk.Frame(p, bg=PANEL)
        fila2.pack(fill="x", padx=8, pady=2)
        ttk.Label(fila2, text="Baudios:").pack(side="left")
        self.ent_baudios = ttk.Entry(fila2, width=8)
        self.ent_baudios.insert(0, str(self.aj["baudios"]))
        self.ent_baudios.pack(side="left", padx=4)
        self.btn_conectar = ttk.Button(fila2, text="Conectar", style="Acento.TButton", command=self._alternar_conexion)
        self.btn_conectar.pack(side="right")
        fila3 = tk.Frame(p, bg=PANEL)
        fila3.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(fila3, text="Probar:", style="Tenue.TLabel").pack(side="left", padx=(0, 4))
        for n in sorted(self.nombre_de):
            ttk.Button(fila3, text=str(n), width=2, command=lambda n=n: self._probar_foco(n)).pack(side="left", padx=1)
        ttk.Button(fila3, text="Apagar", command=self._apagar_todo).pack(side="right")

    def _panel_voz(self, padre) -> None:
        p = ttk.Labelframe(padre, text="Voz", style="Panel.TLabelframe")
        p.pack(fill="x", pady=(0, 8))
        self.var_voz = tk.BooleanVar(value=False)
        ttk.Checkbutton(p, text="Activar escucha por voz", variable=self.var_voz,
                        command=self._alternar_voz).pack(anchor="w", padx=8, pady=(6, 2))
        self.lbl_voz_estado = tk.Label(p, text="Desactivada", bg=PANEL, fg=TENUE, anchor="w")
        self.lbl_voz_estado.pack(fill="x", padx=8)
        self.lbl_frase = tk.Label(p, text="Frase: —", bg=PANEL, fg=TEXTO, anchor="w", wraplength=400, justify="left")
        self.lbl_frase.pack(fill="x", padx=8, pady=(4, 0))
        self.lbl_comando = tk.Label(p, text="Comando: —", bg=PANEL, fg=TENUE, anchor="w")
        self.lbl_comando.pack(fill="x", padx=8)
        fila = tk.Frame(p, bg=PANEL)
        fila.pack(fill="x", padx=8, pady=(6, 8))
        self.ent_frase = ttk.Entry(fila)
        self.ent_frase.insert(0, "dónde está el martillo")
        self.ent_frase.pack(side="left", fill="x", expand=True)
        self.ent_frase.bind("<Return>", lambda e: self._simular_voz())
        ttk.Button(fila, text="Simular voz", command=self._simular_voz).pack(side="left", padx=(4, 0))

    def _panel_deteccion(self, padre) -> None:
        p = ttk.Labelframe(padre, text="Detección", style="Panel.TLabelframe")
        p.grid(row=2, column=0, sticky="ew", padx=(10, 5), pady=(0, 5))
        fila = tk.Frame(p, bg=PANEL)
        fila.pack(fill="x", padx=8, pady=6)
        ttk.Label(fila, text="Confianza mínima:").pack(side="left")
        self.esc_conf = ttk.Scale(fila, from_=0.05, to=0.95, value=self.conf, command=self._cambiar_conf)
        self.esc_conf.pack(side="left", fill="x", expand=True, padx=8)
        self.lbl_conf = ttk.Label(fila, text=f"{self.conf:.2f}", width=5)
        self.lbl_conf.pack(side="left")
        self.var_enviar_cam = tk.BooleanVar(value=True)
        ttk.Checkbutton(fila, text="Encender focos con la cámara",
                        variable=self.var_enviar_cam).pack(side="left", padx=(12, 0))

    # ---------------- Utilidades ----------------
    def log(self, texto: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", f"{datetime.now():%H:%M:%S}  {texto}\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _cambiar_conf(self, valor) -> None:
        self.conf = round(float(valor), 2)
        self.lbl_conf.configure(text=f"{self.conf:.2f}")

    # ---------------- Bluetooth ----------------
    def _refrescar_puertos(self) -> None:
        valores = [SIMULADO] + [f"{dev} — {desc}" for dev, desc in listar_puertos()]
        self.cmb_puerto["values"] = valores
        actual = self.cmb_puerto.get()
        preferido = self.aj["puerto"] or self.args.puerto
        elegido = next((v for v in valores if preferido and v.startswith(preferido)), None)
        if elegido:
            self.cmb_puerto.set(elegido)
        elif actual not in valores:
            self.cmb_puerto.set(valores[1] if len(valores) > 1 else SIMULADO)

    def _puerto_elegido(self) -> str:
        sel = self.cmb_puerto.get().strip()
        return sel if sel == SIMULADO else sel.split(" — ")[0].strip()

    def _alternar_conexion(self) -> None:
        if self.enlace.conectado:
            self.enlace.enviar("0\n")
            self.enlace.cerrar()
            self._actualizar_estado_bt()
            self.log("Bluetooth desconectado.")
            return
        puerto = self._puerto_elegido()
        try:
            baudios = int(self.ent_baudios.get())
        except ValueError:
            self.log("ERROR: los baudios deben ser un número (p. ej. 9600).")
            return
        self.btn_conectar.configure(state="disabled")
        self.estado_bt.set(f"Bluetooth: conectando a {puerto}...")
        self.log(f"Conectando a {puerto} ({baudios} baudios)... puede tardar unos segundos.")

        def trabajo() -> None:
            try:
                self.enlace.conectar(puerto, baudios)
                self.cola.put(("bt", True, f"Conectado a {puerto}."))
            except RuntimeError as e:
                self.cola.put(("bt", False, str(e)))

        threading.Thread(target=trabajo, daemon=True).start()

    def _actualizar_estado_bt(self) -> None:
        if self.enlace.conectado:
            self.estado_bt.set(f"Bluetooth: {'SIMULADO' if self.enlace.simulado else self.enlace.puerto}")
            self.btn_conectar.configure(text="Desconectar")
        else:
            self.estado_bt.set("Bluetooth: desconectado")
            self.btn_conectar.configure(text="Conectar")

    def _probar_foco(self, n: int) -> None:
        if not self.enlace.conectado:
            self.log("Conecta primero el Bluetooth (o elige SIMULADO).")
        self.voz_ultima_orden[n] = "prueba"
        self.logica.activar_voz({n}, time.monotonic(), 2.0)
        self.log(f"Prueba: foco {n} ({self.nombre_de[n]}) durante 2 s.")

    def _apagar_todo(self) -> None:
        self.logica.cancelar_voz()
        self.voz_ultima_orden.clear()
        self.log("Focos de voz/prueba apagados.")

    # ---------------- Voz ----------------
    def _alternar_voz(self) -> None:
        if self.var_voz.get():
            self.voz.iniciar()
            self.estado_voz.set("Voz: activada")
            self.log("Escucha por voz activada (requiere internet y micrófono).")
        else:
            self.voz.detener()
            self.estado_voz.set("Voz: desactivada")
            self.lbl_voz_estado.configure(text="Desactivada", fg=TENUE)

    def _simular_voz(self) -> None:
        texto = self.ent_frase.get().strip()
        if texto:
            self.cola.put(("voz_texto", texto))

    def _procesar_voz(self, texto: str) -> None:
        clases, apagar = interpretar(texto, self.aj["alias"])
        self.lbl_frase.configure(text=f"Frase: “{texto}”")
        if apagar:
            self._apagar_todo()
            self.lbl_comando.configure(text="Comando: apagar", fg=AMBAR)
            self.log(f"Voz: “{texto}” -> apagar")
            return
        numeros = {self.numeracion[c] for c in clases if c in self.numeracion}
        if not numeros:
            self.lbl_comando.configure(text="Comando: ninguna herramienta reconocida", fg=TENUE)
            self.log(f"Voz: “{texto}” (sin herramienta)")
            return
        ahora = time.monotonic()
        self.logica.activar_voz(numeros, ahora, self.aj["duracion_voz_s"])
        for c in clases:
            self.voz_ultima_orden[self.numeracion[c]] = "voz"
        detalle = ", ".join(f"{c.replace('_', ' ')} (foco {self.numeracion[c]})" for c in clases)
        self.lbl_comando.configure(text=f"Comando: {detalle}", fg=VERDE)
        self.log(f"Voz: “{texto}” -> {detalle}")

    # ---------------- Cámara y modelo (hilo) ----------------
    def _bucle_camara(self) -> None:
        post = lambda *a: self.cola.put(a)  # noqa: E731
        try:
            import cv2
            import torch  # noqa: F401
            from ultralytics import YOLO
            from detectar_camara import PALETA, abrir_fuente, buscar_pesos, dibujar_deteccion
            from verificar_gpu import seleccionar_dispositivo
        except ImportError as e:
            post("cam_estado", f"Cámara: ERROR (falta {e.name})")
            post("log", f"ERROR: falta {e.name}. Ejecuta: pip install -r requirements.txt")
            return

        pesos = Path(self.args.pesos) if self.args.pesos else buscar_pesos(Path("weights"), Path("runs"))
        if pesos is None or not pesos.exists():
            post("cam_estado", "Cámara: sin pesos")
            post("log", "ERROR: no encontré pesos entrenados. Usa --pesos weights/<nombre>_best.pt")
            return
        try:
            dev_str, _ = seleccionar_dispositivo(self.args.dispositivo)
        except (RuntimeError, ValueError) as e:
            post("cam_estado", "Cámara: ERROR de GPU")
            post("log", f"ERROR: {e}")
            return
        usa_gpu = dev_str.startswith("cuda")
        dispositivo = int(dev_str.split(":")[1]) if usa_gpu else "cpu"

        post("cam_estado", "Cámara: cargando modelo...")
        modelo = YOLO(str(pesos))
        nombres = modelo.names
        desconocidas = [n for n in nombres.values() if n not in self.numeracion]
        if desconocidas:
            post("log", f"AVISO: el modelo tiene clases sin número asignado (se ignoran): {desconocidas}")
        post("log", f"Modelo: {pesos.name} en {'GPU' if usa_gpu else 'CPU'}.")

        fuente = self.args.fuente if self.args.fuente else str(self.args.camara)
        cap = abrir_fuente(fuente, 1280, 720)
        if not cap.isOpened():
            post("cam_estado", "Cámara: no se pudo abrir")
            post("log", f"ERROR: no se pudo abrir '{fuente}'. Cierra otras apps que usen la cámara o prueba --camara 1.")
            return
        post("cam_estado", f"Cámara: activa ({'GPU' if usa_gpu else 'CPU'})")

        fps = 0.0
        try:
            while not self.parar.is_set():
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    if self.args.fuente:                      # video: reinicia al terminar
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    post("cam_estado", "Cámara: sin imagen")
                    break
                res = modelo.predict(frame, conf=self.conf, iou=0.5, imgsz=self.args.imgsz,
                                     device=dispositivo, half=usa_gpu, verbose=False)[0]
                detecciones = []
                if len(res.boxes):
                    for xyxy, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                          res.boxes.cls.cpu().numpy()):
                        nombre = nombres[int(k)]
                        if nombre not in self.numeracion:
                            continue
                        detecciones.append((nombre, float(c)))
                        texto = f"{self.numeracion[nombre]}  {nombre.replace('_', ' ')} {float(c) * 100:.0f}%"
                        dibujar_deteccion(cv2, frame, xyxy, texto, PALETA[int(k) % len(PALETA)])
                dt = time.perf_counter() - t0
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
                with self.lock:
                    self.frame, self.dets, self.fps = frame, (time.monotonic(), detecciones), fps
        except Exception as e:
            texto = str(e)
            post("cam_estado", "Cámara: ERROR")
            if "out of memory" in texto.lower():
                post("log", "ERROR: sin memoria de GPU. Cierra otras apps (¿entrenamiento en curso?) o usa --dispositivo cpu.")
            else:
                post("log", f"ERROR en la cámara/modelo: {texto}")
        finally:
            cap.release()

    # ---------------- Bucles de la interfaz ----------------
    def _pintar_video(self) -> None:
        if self.parar.is_set():
            return
        with self.lock:
            frame, fps = self.frame, self.fps
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w > 10 and h > 10:
            if frame is None:
                self.canvas.delete("all")
                self.canvas.create_text(w // 2, h // 2, text=self.estado_camara.get(), fill=TENUE,
                                        font=("Segoe UI", 14))
            else:
                img = Image.fromarray(frame[:, :, ::-1])
                img.thumbnail((w, h), Image.BILINEAR)
                self._foto = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.create_image(w // 2, h // 2, image=self._foto)
                self.canvas.create_text(10, 10, anchor="nw", fill=VERDE, font=("Consolas", 11, "bold"),
                                        text=f"{fps:4.1f} FPS   conf >= {self.conf:.2f}")
        self.root.after(33, self._pintar_video)

    def _vaciar_cola(self) -> None:
        while True:
            try:
                ev = self.cola.get_nowait()
            except queue.Empty:
                return
            tipo = ev[0]
            if tipo == "log":
                self.log(ev[1])
            elif tipo == "cam_estado":
                self.estado_camara.set(ev[1])
            elif tipo == "voz_estado":
                texto = ev[1]
                self.lbl_voz_estado.configure(text=texto, fg=ROJO if texto.startswith(("ERROR", "Sin conexión")) else VERDE)
            elif tipo == "voz_texto":
                self._procesar_voz(ev[1])
            elif tipo == "bt":
                self.btn_conectar.configure(state="normal")
                self._actualizar_estado_bt()
                self.log(ev[2] if ev[1] else "ERROR de Bluetooth: " + ev[2])
                if ev[1]:
                    self.logica.reiniciar_envio()

    def _tick(self) -> None:
        if self.parar.is_set():
            return
        self._vaciar_cola()
        ahora = time.monotonic()

        with self.lock:
            t_det, lista = self.dets
        recientes = lista if ahora - t_det < 1.0 else []
        conf_por_clase: dict[str, float] = {}
        for nombre, c in recientes:
            conf_por_clase[nombre] = max(c, conf_por_clase.get(nombre, 0.0))
        if self.var_enviar_cam.get():
            self.logica.actualizar_camara({self.numeracion[n] for n in conf_por_clase}, ahora)
        else:
            self.logica.actualizar_camara(set(), ahora)

        mensaje = self.logica.mensaje_si_toca(ahora)
        if mensaje is not None and self.enlace.conectado:
            if self.enlace.enviar(mensaje):
                self.lbl_ultimo.configure(text=f"Último mensaje enviado: {mensaje.strip()!r}")
            else:
                self._actualizar_estado_bt()
                self.log("ERROR: se perdió la conexión Bluetooth. Pulsa Conectar para reconectar.")

        activos, por_voz = self.logica.activos(ahora), self.logica.activos_voz(ahora)
        for n, fila in self.filas.items():
            clase = self.nombre_de[n]
            fila["lampara"].itemconfig(fila["ovalo"], fill=AMBAR if n in activos else GRIS)
            if clase in conf_por_clase:
                fila["estado"].configure(text=f"Detectada {conf_por_clase[clase] * 100:.0f} %", fg=VERDE)
            elif n in por_voz:
                fila["estado"].configure(text="Pedida por voz" if self.voz_ultima_orden.get(n) != "prueba"
                                         else "Prueba", fg=AZUL)
            else:
                fila["estado"].configure(text="—", fg=TENUE)
        self.root.after(100, self._tick)

    def _cerrar(self) -> None:
        self.parar.set()
        try:
            self.voz.detener()
            if self.enlace.conectado:
                self.enlace.enviar("0\n")      # deja los focos apagados
            self.enlace.cerrar()
        finally:
            self.root.destroy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Dashboard del detector de herramientas")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pesos", help="ruta a best.pt (por defecto, los más recientes)")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--fuente", help="ruta a un video en lugar de la cámara")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dispositivo", default="auto", help="auto, cpu, 0, 1, ...")
    ap.add_argument("--puerto", default="", help="puerto COM a preseleccionar (p. ej. COM5)")
    ap.add_argument("--sin-camara", action="store_true", help="no abre la cámara ni carga el modelo")
    args = ap.parse_args()

    ajustes = cargar_ajustes(Path(args.config))
    root = tk.Tk()
    App(root, ajustes, args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
