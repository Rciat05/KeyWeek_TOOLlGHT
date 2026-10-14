"""
comandos_voz.py - Reconocimiento de voz para pedir herramientas ("¿dónde está el martillo?").

Piezas:
  * interpretar(texto, alias)  -> qué clases se mencionan y si se pidió apagar. (Pura, probada.)
  * SegmentadorVoz             -> detecta frases en el flujo del micrófono por energía.
  * EscuchaVoz                 -> hilo que graba (sounddevice), segmenta y reconoce con el
                                  servicio de Google (SpeechRecognition).

PRIVACIDAD Y REQUISITOS: el reconocimiento usa el servicio web de voz de Google. Necesita
INTERNET y el audio de cada frase se envía a ese servicio. Sin internet, el dashboard sigue
funcionando con la cámara; la voz mostrará "sin conexión".
"""
from __future__ import annotations

import difflib
import queue
import re
import threading
import unicodedata

import numpy as np

# Palabras con las que se puede pedir cada clase. Se pueden ampliar en config.yaml
# (dashboard.voz.alias). Incluye variantes que el reconocedor suele escribir mal
# (p. ej. "cutter" en inglés suele salir como "cuter", "cáter" o "cúter").
ALIAS_POR_DEFECTO = {
    "martillo": ["martillo", "martillos"],
    "pinzas": ["pinza", "pinzas", "alicate", "alicates"],
    "cutter": ["cutter", "cuter", "cúter", "cater", "kater", "cortador"],
    "cinta_metrica": ["cinta metrica", "cinta metro", "metro", "flexometro", "cinta"],
    "guantes": ["guante", "guantes"],
    "pistola_silicona": ["pistola de silicona", "pistola silicona", "pistola", "silicona"],
    "pie_de_rey": ["pie de rey", "pie rey", "calibrador", "calibre", "vernier"],
    "casco": ["casco", "cascos"],
}

PALABRAS_APAGAR = {"apaga", "apagar", "apagado", "cancela", "cancelar", "limpia", "limpiar"}


def normalizar(texto: str) -> str:
    """minúsculas, sin acentos ni signos: '¿Dónde está el Cúter?' -> 'donde esta el cuter'."""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def alias_para(clases: list[str], extra: dict | None = None) -> dict[str, list[str]]:
    """Alias de cada clase: los de por defecto, los de config.yaml y el propio nombre de la clase."""
    resultado: dict[str, list[str]] = {}
    for c in clases:
        lista = list(ALIAS_POR_DEFECTO.get(c, []))
        lista += list((extra or {}).get(c, []))
        lista.append(c.replace("_", " "))
        resultado[c] = list(dict.fromkeys(lista))
    return resultado


def _coincide(alias_norm: str, tokens: list[str], texto_norm: str, umbral: float) -> int | None:
    """Posición (índice de token) donde aparece el alias, o None."""
    palabras = alias_norm.split()
    n = len(palabras)
    if n == 0:
        return None
    for i in range(len(tokens) - n + 1):
        ventana = tokens[i:i + n]
        if ventana == palabras:
            return i
        if n == 1 and len(palabras[0]) >= 4 and len(ventana[0]) >= 4:
            if difflib.SequenceMatcher(None, ventana[0], palabras[0]).ratio() >= umbral:
                return i
        if n > 1 and difflib.SequenceMatcher(None, " ".join(ventana), alias_norm).ratio() >= umbral:
            return i
    return None


def interpretar(texto: str, alias: dict[str, list[str]], umbral: float = 0.82) -> tuple[list[str], bool]:
    """Devuelve (clases mencionadas en orden de aparición, ¿pidió apagar?)."""
    texto_norm = normalizar(texto)
    tokens = texto_norm.split()
    apagar = any(t in PALABRAS_APAGAR for t in tokens)
    posiciones: dict[str, int] = {}
    for clase, lista in alias.items():
        for a in lista:
            pos = _coincide(normalizar(a), tokens, texto_norm, umbral)
            if pos is not None:
                posiciones[clase] = min(pos, posiciones.get(clase, pos))
    return sorted(posiciones, key=posiciones.get), apagar


class SegmentadorVoz:
    """Agrupa bloques de audio (int16, mono) en frases usando la energía de la señal."""

    def __init__(self, fs: int = 16000, bloque_ms: int = 30, rms_minimo: float = 400.0,
                 factor_ruido: float = 3.0, silencio_s: float = 0.8, min_habla_s: float = 0.3,
                 max_frase_s: float = 8.0, preroll_s: float = 0.3, bloques_inicio: int = 2) -> None:
        self.fs = fs
        self.rms_minimo, self.factor_ruido = rms_minimo, factor_ruido
        self.bloques_silencio = max(1, int(silencio_s * 1000 / bloque_ms))
        self.bloques_min = max(1, int(min_habla_s * 1000 / bloque_ms))
        self.bloques_max = int(max_frase_s * 1000 / bloque_ms)
        self.bloques_preroll = max(0, int(preroll_s * 1000 / bloque_ms))
        self.bloques_inicio = bloques_inicio
        self._ruido = rms_minimo / 3
        self._preroll: list[np.ndarray] = []
        self._frase: list[np.ndarray] = []
        self._hablando = False
        self._seguidos = 0
        self._silencios = 0
        self._con_voz = 0

    @property
    def hablando(self) -> bool:
        return self._hablando

    def _umbral(self) -> float:
        return max(self.rms_minimo, self._ruido * self.factor_ruido)

    def procesar(self, bloque: np.ndarray) -> bytes | None:
        """Procesa un bloque. Devuelve el audio PCM int16 de una frase cuando esta termina."""
        rms = float(np.sqrt(np.mean(bloque.astype(np.float64) ** 2))) if len(bloque) else 0.0
        hay_voz = rms > self._umbral()

        if not self._hablando:
            if not hay_voz:
                self._ruido = 0.95 * self._ruido + 0.05 * rms      # aprende el ruido de fondo
                self._seguidos = 0
                self._preroll.append(bloque)
                if len(self._preroll) > self.bloques_preroll:
                    del self._preroll[:len(self._preroll) - self.bloques_preroll]
                return None
            self._seguidos += 1
            self._preroll.append(bloque)
            if self._seguidos >= self.bloques_inicio:
                self._hablando = True
                self._frase = list(self._preroll)
                self._preroll = []
                self._silencios, self._con_voz = 0, self._seguidos
            return None

        self._frase.append(bloque)
        if hay_voz:
            self._silencios = 0
            self._con_voz += 1
        else:
            self._silencios += 1
        if self._silencios >= self.bloques_silencio or len(self._frase) >= self.bloques_max:
            return self._terminar()
        return None

    def _terminar(self) -> bytes | None:
        audio = np.concatenate(self._frase).astype(np.int16).tobytes() if self._frase else None
        valido = self._con_voz >= self.bloques_min
        self._hablando, self._frase, self._preroll = False, [], []
        self._seguidos = self._silencios = self._con_voz = 0
        return audio if valido else None


def listar_microfonos() -> list[tuple[int, str]]:
    try:
        import sounddevice as sd
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    except Exception:
        return []


class EscuchaVoz:
    """Hilo de escucha: micrófono -> segmentación -> reconocimiento -> callback con el texto."""

    FS = 16000
    BLOQUE_MS = 30

    def __init__(self, al_texto, al_estado=None, idioma: str = "es-MX", microfono: int | None = None) -> None:
        self.al_texto = al_texto
        self.al_estado = al_estado or (lambda s: None)
        self.idioma = idioma
        self.microfono = microfono
        self._parar = threading.Event()
        self._hilo: threading.Thread | None = None

    @property
    def activa(self) -> bool:
        return self._hilo is not None and self._hilo.is_alive()

    def iniciar(self) -> None:
        if self.activa:
            return
        self._parar.clear()
        self._hilo = threading.Thread(target=self._bucle, daemon=True, name="escucha-voz")
        self._hilo.start()

    def detener(self) -> None:
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=3)
        self._hilo = None
        self.al_estado("Voz desactivada")

    def _bucle(self) -> None:
        try:
            import sounddevice as sd
            import speech_recognition as sr
        except ImportError as e:
            self.al_estado(f"ERROR: falta {e.name} (pip install -r requirements.txt)")
            return
        reconocedor = sr.Recognizer()
        cola: queue.Queue = queue.Queue(maxsize=400)
        segmentador = SegmentadorVoz(fs=self.FS, bloque_ms=self.BLOQUE_MS)

        def callback(indata, frames, tiempo, estado):
            try:
                cola.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        try:
            flujo = sd.InputStream(samplerate=self.FS, channels=1, dtype="int16", device=self.microfono,
                                   blocksize=int(self.FS * self.BLOQUE_MS / 1000), callback=callback)
            flujo.start()
        except Exception as e:
            self.al_estado(f"ERROR de micrófono: {e}")
            return

        self.al_estado("Escuchando...")
        try:
            while not self._parar.is_set():
                try:
                    bloque = cola.get(timeout=0.2)
                except queue.Empty:
                    continue
                audio = segmentador.procesar(bloque)
                if audio is None:
                    continue
                self.al_estado("Procesando...")
                try:
                    texto = reconocedor.recognize_google(sr.AudioData(audio, self.FS, 2), language=self.idioma)
                    self.al_texto(texto)
                except sr.UnknownValueError:
                    pass                                  # no se entendió: se ignora
                except sr.RequestError:
                    self.al_estado("Sin conexión con el servicio de voz (¿internet?)")
                    self._parar.wait(2)
                except Exception as e:
                    self.al_estado(f"ERROR de reconocimiento: {e}")
                if not self._parar.is_set():
                    self.al_estado("Escuchando...")
        finally:
            flujo.stop()
            flujo.close()
