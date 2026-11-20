
import os
import sys
import urllib.request
from pathlib import Path


# app/ es la carpeta base del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]

# Ubicación esperada por main.py
MODEL_PATH = BASE_DIR / "modelo" / "modelo.joblib"


def descargar_modelo():
    model_url = os.getenv("MODEL_URL")

    if not model_url:
        raise RuntimeError(
            "Falta configurar la variable de entorno MODEL_URL."
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = MODEL_PATH.with_suffix(".joblib.tmp")

    request = urllib.request.Request(
        model_url,
        headers={"User-Agent": "ReciClass-Model-Downloader"}
    )

    print("Descargando modelo desde Hugging Face...")

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with open(temp_path, "wb") as archivo:
                while True:
                    bloque = response.read(1024 * 1024)
                    if not bloque:
                        break
                    archivo.write(bloque)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("La descarga produjo un archivo vacío.")

        temp_path.replace(MODEL_PATH)

        print("Modelo descargado correctamente.")
        print(f"Ubicación: {MODEL_PATH}")
        print(f"Tamaño: {MODEL_PATH.stat().st_size} bytes")

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


if __name__ == "__main__":
    try:
        descargar_modelo()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
