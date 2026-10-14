"""
arduino_bt.py - Comunicación con el Arduino a través de un módulo Bluetooth (HC-05 / HC-06).

Un módulo Bluetooth clásico (HC-05, HC-06) se empareja con Windows y aparece como un
PUERTO COM; desde Python se usa igual que un cable serie (pyserial).

PROTOCOLO (texto, una línea por mensaje, terminada en '\\n'):
    "135\\n"  -> enciende los focos 1, 3 y 5 (y apaga los demás)
    "4\\n"    -> enciende solo el foco 4
    "0\\n"    -> apaga todos
El Arduino apaga todo si deja de recibir mensajes durante ~1,5 s (seguridad), así que
mientras algo deba seguir encendido se reenvía el mensaje periódicamente.
"""
from __future__ import annotations

import threading
import time

SIMULADO = "SIMULADO (sin Arduino)"
MAX_FOCOS = 9  # el protocolo usa un dígito por foco (1-9)


def construir_mensaje(numeros) -> str:
    """Convierte un conjunto de números de foco (1-9) en la línea que entiende el Arduino."""
    numeros = sorted({int(n) for n in numeros})
    for n in numeros:
        if not 1 <= n <= MAX_FOCOS:
            raise ValueError(f"Número de foco fuera de rango (1-{MAX_FOCOS}): {n}")
    return "0\n" if not numeros else "".join(str(n) for n in numeros) + "\n"


def listar_puertos() -> list[tuple[str, str]]:
    """Devuelve [(COM3, 'Standard Serial over Bluetooth link (COM3)'), ...]."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [(p.device, p.description) for p in sorted(list_ports.comports(), key=lambda p: p.device)]


class EnlaceArduino:
    """Envía mensajes al Arduino por un puerto serie (real, simulado o de prueba)."""

    def __init__(self) -> None:
        self._ser = None
        self._lock = threading.Lock()
        self.simulado = False
        self.puerto: str | None = None
        self.ultimo_mensaje: str | None = None
        self.historial_simulado: list[str] = []

    @property
    def conectado(self) -> bool:
        return self.simulado or (self._ser is not None and self._ser.is_open)

    def conectar(self, puerto: str, baudios: int = 9600) -> None:
        """Abre el puerto. Lanza RuntimeError con un mensaje claro si no se puede."""
        self.cerrar()
        if puerto == SIMULADO:
            self.simulado, self.puerto = True, puerto
            return
        try:
            import serial
        except ImportError as e:
            raise RuntimeError("Falta pyserial. Ejecuta: pip install -r requirements.txt") from e
        try:
            if "://" in puerto:  # p. ej. loop:// para pruebas
                ser = serial.serial_for_url(puerto, baudrate=baudios, timeout=0.1, write_timeout=1)
            else:
                ser = serial.Serial(puerto, baudios, timeout=0.1, write_timeout=1)
        except serial.SerialException as e:
            raise RuntimeError(
                f"No se pudo abrir {puerto}: {e}\n"
                "  - ¿El módulo está emparejado y encendido? (PIN habitual: 1234 o 0000)\n"
                "  - Usa el puerto COM SALIENTE del módulo (Más opciones de Bluetooth > Puertos COM).\n"
                "  - Cierra otros programas que usen ese puerto (monitor serie del Arduino IDE, etc.)."
            ) from e
        self._ser, self.puerto = ser, puerto

    def enviar(self, mensaje: str) -> bool:
        """Envía un mensaje. Devuelve False (y cierra el enlace) si falla la escritura."""
        with self._lock:
            if self.simulado:
                self.historial_simulado.append(mensaje.strip())
                del self.historial_simulado[:-50]
                self.ultimo_mensaje = mensaje.strip()
                return True
            if self._ser is None or not self._ser.is_open:
                return False
            try:
                self._ser.write(mensaje.encode("ascii"))
                self._ser.flush()
                self.ultimo_mensaje = mensaje.strip()
                return True
            except Exception:
                self._cerrar_sin_lock()
                return False

    def leer_disponible(self) -> str:
        """Lee lo que haya enviado el Arduino (solo para depuración)."""
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                return ""
            try:
                return self._ser.read(self._ser.in_waiting or 0).decode("ascii", errors="replace")
            except Exception:
                return ""

    def _cerrar_sin_lock(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self.simulado = False

    def cerrar(self) -> None:
        with self._lock:
            self._cerrar_sin_lock()
            self.puerto = None


if __name__ == "__main__":  # prueba rápida: python scripts/arduino_bt.py COM5
    import sys

    puertos = listar_puertos()
    print("Puertos disponibles:", puertos or "ninguno")
    if len(sys.argv) > 1:
        enlace = EnlaceArduino()
        enlace.conectar(sys.argv[1])
        for n in range(1, 9):
            enlace.enviar(construir_mensaje({n}))
            print(f"foco {n} encendido")
            time.sleep(0.8)
        enlace.enviar(construir_mensaje(set()))
        enlace.cerrar()
