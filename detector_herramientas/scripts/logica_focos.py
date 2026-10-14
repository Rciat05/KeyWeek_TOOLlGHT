"""
logica_focos.py - Decide qué focos deben estar encendidos y cuándo hay que enviar un mensaje.

Fuentes de encendido:
  * CÁMARA: una herramienta enciende su foco cuando se detecta de forma continua durante
    `confirmacion_s` (evita parpadeos por una detección suelta) y lo mantiene encendido
    `liberacion_s` después de dejar de verla (evita apagados por un fotograma perdido).
  * VOZ: al pedir una herramienta por voz, su foco se enciende `duracion_voz_s` segundos,
    aunque la cámara no la esté viendo.
Los focos encendidos son la unión de ambas fuentes.

Reenvío: el mensaje se envía al cambiar el estado y, mientras haya algún foco encendido,
cada `keepalive_s` (el Arduino apaga todo si deja de recibir mensajes).

Todas las funciones reciben el tiempo `ahora` (segundos) para poder probarse sin esperar.
"""
from __future__ import annotations

from arduino_bt import construir_mensaje, MAX_FOCOS


class LogicaFocos:
    def __init__(self, confirmacion_s: float = 0.3, liberacion_s: float = 1.0,
                 keepalive_s: float = 0.5) -> None:
        self.confirmacion_s = confirmacion_s
        self.liberacion_s = liberacion_s
        self.keepalive_s = keepalive_s
        self._desde: dict[int, float] = {}      # foco -> inicio de la detección continua
        self._ultima: dict[int, float] = {}     # foco -> última vez que se vio
        self._voz: dict[int, float] = {}        # foco -> instante en que expira el encendido por voz
        self._enviado: frozenset[int] = frozenset()
        self._ultimo_envio: float = float("-inf")

    # --- Entradas -------------------------------------------------------------
    def actualizar_camara(self, detectados: set[int], ahora: float) -> None:
        for n in detectados:
            self._desde.setdefault(n, ahora)
            self._ultima[n] = ahora
        for n in list(self._desde):
            if n not in detectados and ahora - self._ultima.get(n, float("-inf")) > self.liberacion_s:
                del self._desde[n]

    def activar_voz(self, numeros: set[int], ahora: float, duracion_s: float) -> None:
        for n in numeros:
            if 1 <= n <= MAX_FOCOS:
                self._voz[n] = ahora + duracion_s

    def cancelar_voz(self) -> None:
        self._voz.clear()

    # --- Estado ---------------------------------------------------------------
    def activos_camara(self, ahora: float) -> set[int]:
        return {n for n, t0 in self._desde.items()
                if ahora - t0 >= self.confirmacion_s and ahora - self._ultima[n] <= self.liberacion_s}

    def activos_voz(self, ahora: float) -> set[int]:
        return {n for n, exp in self._voz.items() if exp > ahora}

    def activos(self, ahora: float) -> set[int]:
        return self.activos_camara(ahora) | self.activos_voz(ahora)

    # --- Salida ---------------------------------------------------------------
    def mensaje_si_toca(self, ahora: float) -> str | None:
        """Devuelve el mensaje a enviar al Arduino, o None si no hace falta enviar nada."""
        actuales = frozenset(self.activos(ahora))
        cambio = actuales != self._enviado
        reenvio = bool(actuales) and ahora - self._ultimo_envio >= self.keepalive_s
        if not (cambio or reenvio):
            return None
        self._enviado = actuales
        self._ultimo_envio = ahora
        return construir_mensaje(actuales)

    def reiniciar_envio(self) -> None:
        """Fuerza que el próximo ciclo reenvíe el estado (p. ej. tras reconectar)."""
        self._enviado = frozenset({-1})
