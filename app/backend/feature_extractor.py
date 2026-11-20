"""Extracción HOG + LBP + HSV idéntica a la sección 4 del notebook ReciClass."""

from io import BytesIO

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

TAMANO = (128, 128)  # (alto, ancho)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_P = 8
LBP_R = 1
HSV_BINS = (8, 8, 4)


def extraer_caracteristicas_desde_bytes(image_bytes: bytes) -> np.ndarray:
    """Recibe bytes de una imagen y devuelve matriz (1, 8366)."""
    with Image.open(BytesIO(image_bytes)) as im:
        im = im.convert("RGB").resize(
            (TAMANO[1], TAMANO[0]), Image.Resampling.BILINEAR
        )
        rgb = np.asarray(im, dtype=np.float32) / 255.0

    gris = rgb2gray(rgb)

    vector_hog = hog(
        gris,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    gris_uint8 = np.clip(gris * 255, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(
        gris_uint8, P=LBP_P, R=LBP_R, method="uniform"
    )
    hist_lbp, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_P + 3),
        density=True,
    )

    hsv = rgb2hsv(rgb)
    hist_hsv, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=HSV_BINS,
        range=((0, 1), (0, 1), (0, 1)),
        density=True,
    )
    hist_hsv = hist_hsv.ravel()

    vector = np.concatenate([
        vector_hog,
        hist_lbp.astype(np.float32),
        hist_hsv.astype(np.float32),
    ]).astype(np.float32)

    return vector.reshape(1, -1)
