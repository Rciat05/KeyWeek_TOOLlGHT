"""
verificar_gpu.py - Diagnóstico real de GPU/CUDA para PyTorch.

No basta con que el equipo tenga una tarjeta NVIDIA: este script comprueba
que PyTorch puede USARLA ejecutando operaciones reales en ella.

Uso:
    python scripts/verificar_gpu.py
    python scripts/verificar_gpu.py --dispositivo 0
    python scripts/verificar_gpu.py --tamano 6000

Código de salida: 0 si la GPU funciona, 1 si no.
Otros scripts pueden importar `diagnostico_gpu()` y `seleccionar_dispositivo()`.
"""
from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import time

MENSAJE_INSTALACION = """
--------------------------------------------------------------------
CÓMO INSTALAR PyTorch CON SOPORTE CUDA (Windows, PowerShell)
--------------------------------------------------------------------
1) Mira la versión de CUDA que admite tu controlador:
       nvidia-smi
   (arriba a la derecha aparece "CUDA Version: X.Y"). Si el comando no
   existe, instala/actualiza el controlador NVIDIA:
       https://www.nvidia.com/Download/index.aspx
   No necesitas instalar el "CUDA Toolkit": los paquetes de PyTorch
   ya incluyen las librerías CUDA necesarias.

2) Elige el comando exacto para tu equipo en el selector oficial:
       https://pytorch.org/get-started/locally/
   (Stable -> Windows -> Pip -> Python -> la plataforma CUDA que
   no supere la versión de tu controlador).

3) Desinstala la versión actual (probablemente solo CPU) e instala la
   de CUDA. EJEMPLO con CUDA 12.8 (confirma el sufijo cuXXX en pytorch.org):
       pip uninstall -y torch torchvision torchaudio
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

   Las GPU RTX serie 50 (Blackwell) requieren compilaciones recientes
   (cu128 o superior). Las series 20/30/40 funcionan con cu118/cu126/cu128.

4) Repite este diagnóstico:
       python scripts/verificar_gpu.py
--------------------------------------------------------------------
"""


def _ejecutar(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def info_nvidia_smi() -> dict:
    """Consulta el controlador NVIDIA con nvidia-smi (si existe)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"disponible": False}
    datos = {"disponible": True}
    salida = _ejecutar([exe, "--query-gpu=name,driver_version,memory.total,memory.used",
                        "--format=csv,noheader"])
    if salida:
        datos["gpus"] = [linea.strip() for linea in salida.splitlines()]
    texto = _ejecutar([exe])
    if texto:
        m = re.search(r"CUDA Version:\s*([\d.]+)", texto)
        if m:
            datos["cuda_maxima_controlador"] = m.group(1)
    return datos


def seleccionar_dispositivo(dispositivo: str = "auto"):
    """
    Devuelve (device_str, torch) según la petición:
      - "auto": GPU 0 si CUDA funciona; si no, CPU (con aviso).
      - "cpu": CPU explícita.
      - "0", "1", "cuda:0"...: esa GPU; error si no está disponible.
    """
    import torch

    pedido = str(dispositivo).lower().strip()
    if pedido == "cpu":
        return "cpu", torch
    if pedido == "auto":
        if torch.cuda.is_available():
            return "cuda:0", torch
        print("[AVISO] CUDA no disponible: se usará la CPU (será muy lento).")
        return "cpu", torch

    indice = pedido.replace("cuda:", "").replace("cuda", "") or "0"
    if not indice.isdigit():
        raise ValueError(f"Dispositivo no válido: '{dispositivo}'. Usa auto, cpu, 0, 1, ...")
    if not torch.cuda.is_available():
        raise RuntimeError("Pediste la GPU pero PyTorch no puede usar CUDA.\n" + MENSAJE_INSTALACION)
    if int(indice) >= torch.cuda.device_count():
        raise RuntimeError(f"La GPU {indice} no existe. GPU detectadas: {torch.cuda.device_count()}.")
    return f"cuda:{indice}", torch


def diagnostico_gpu(dispositivo: str = "auto", tamano: int = 4096, verboso: bool = True) -> bool:
    """Ejecuta el diagnóstico completo. Devuelve True si se puede entrenar en GPU."""
    def p(*args):
        if verboso:
            print(*args)

    p("=" * 68)
    p("DIAGNÓSTICO DE GPU PARA ENTRENAMIENTO")
    p("=" * 68)
    p(f"Sistema operativo : {platform.system()} {platform.release()}")
    p(f"Python            : {sys.version.split()[0]} ({platform.architecture()[0]})")

    # 1) Controlador NVIDIA
    smi = info_nvidia_smi()
    if smi["disponible"]:
        p("nvidia-smi       : encontrado")
        for g in smi.get("gpus", []):
            p(f"  GPU (nombre, controlador, VRAM total, VRAM usada): {g}")
        if "cuda_maxima_controlador" in smi:
            p(f"  CUDA máxima admitida por el controlador: {smi['cuda_maxima_controlador']}")
    else:
        p("nvidia-smi       : NO encontrado (¿controlador NVIDIA instalado y en el PATH?)")

    # 2) PyTorch
    try:
        import torch
    except ImportError:
        p("\n[ERROR] PyTorch no está instalado en este entorno.")
        p(MENSAJE_INSTALACION)
        return False

    p(f"PyTorch           : {torch.__version__}")
    p(f"CUDA de PyTorch   : {torch.version.cuda}  (None = compilación solo CPU)")
    p(f"cuDNN             : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'no disponible'}")
    p(f"CUDA disponible   : {torch.cuda.is_available()}")

    # 3) Diagnóstico de causas si CUDA no está disponible
    if not torch.cuda.is_available():
        p("\n[ERROR] PyTorch NO puede usar la GPU. Causas probables:")
        if torch.version.cuda is None:
            p("  - Tienes instalado PyTorch de solo CPU (muy habitual si se instaló")
            p("    ultralytics antes que PyTorch).")
        if not smi["disponible"]:
            p("  - Falta el controlador NVIDIA o nvidia-smi no está en el PATH.")
        else:
            p("  - El controlador puede ser demasiado antiguo para la versión CUDA de PyTorch.")
        p("  - La GPU puede no ser NVIDIA (AMD/Intel integrada no sirve para CUDA).")
        p(MENSAJE_INSTALACION)
        return False

    # 4) Dispositivo elegido
    try:
        dev, _ = seleccionar_dispositivo(dispositivo)
    except (RuntimeError, ValueError) as e:
        p(f"\n[ERROR] {e}")
        return False
    if dev == "cpu":
        p("\n[AVISO] Se pidió CPU explícitamente; no se probará la GPU.")
        return False

    idx = int(dev.split(":")[1])
    props = torch.cuda.get_device_properties(idx)
    p(f"\nGPU seleccionada  : {dev} -> {props.name}")
    p(f"Capacidad cómputo : {props.major}.{props.minor}")
    p(f"VRAM total        : {props.total_memory / 1024**3:.2f} GB")
    p(f"Multiprocesadores : {props.multi_processor_count}")

    # 5) Prueba real: operaciones en la GPU
    try:
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        a = torch.randn(tamano, tamano, device=dev)
        b = torch.randn(tamano, tamano, device=dev)
        _ = a @ b  # calentamiento
        torch.cuda.synchronize(idx)
        t0 = time.perf_counter()
        for _ in range(5):
            c = a @ b
        torch.cuda.synchronize(idx)
        t_fp32 = (time.perf_counter() - t0) / 5
        assert c.device.type == "cuda", "El resultado no está en la GPU"
        p(f"\nPrueba FP32 ({tamano}x{tamano}): {t_fp32 * 1000:.1f} ms por multiplicación  -> OK en {c.device}")

        amp_ok = False
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = a @ b
                torch.cuda.synchronize(idx)
                t0 = time.perf_counter()
                for _ in range(5):
                    d = a @ b
                torch.cuda.synchronize(idx)
                t_amp = (time.perf_counter() - t0) / 5
            amp_ok = d.dtype == torch.float16
            p(f"Prueba AMP FP16               : {t_amp * 1000:.1f} ms por multiplicación -> "
              f"{'OK (precisión mixta disponible)' if amp_ok else 'sin efecto'}")
        except Exception as e:
            p(f"Prueba AMP: no disponible ({e})")

        usada = torch.cuda.memory_allocated(idx) / 1024**2
        p(f"Memoria reservada en la prueba: {usada:.0f} MB")
        del a, b, c
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        p("\n[ERROR] Sin memoria de GPU incluso en la prueba. Cierra otras aplicaciones "
          "que usen la GPU o reduce --tamano.")
        return False
    except Exception as e:
        p(f"\n[ERROR] Falló una operación real en la GPU: {e}")
        p("Suele indicar incompatibilidad entre controlador, GPU y compilación de PyTorch.")
        p(MENSAJE_INSTALACION)
        return False

    # 6) Ultralytics (informativo)
    try:
        import ultralytics
        p(f"\nUltralytics       : {ultralytics.__version__}")
    except ImportError:
        p("\nUltralytics       : no instalado (pip install -r requirements.txt)")

    p("\n[OK] PyTorch puede ejecutar operaciones reales en tu GPU.")
    p("     Ajusta el batch size según la VRAM: si ves 'CUDA out of memory',")
    p("     reduce batch (16 -> 8 -> 4) o imgsz (640 -> 512 -> 416).")
    p("=" * 68)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnóstico de GPU/CUDA con PyTorch")
    ap.add_argument("--dispositivo", default="auto", help="auto, cpu, 0, 1, ... (por defecto auto)")
    ap.add_argument("--tamano", type=int, default=4096, help="Lado de la matriz de prueba")
    args = ap.parse_args()
    return 0 if diagnostico_gpu(args.dispositivo, args.tamano) else 1


if __name__ == "__main__":
    sys.exit(main())
