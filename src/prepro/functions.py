import cv2
import numpy as np
from PIL import Image


def _build_text_mask(
    pil_img: Image.Image,
    blue_blur_ksize: int = 5,
    bg_median_ksize: int = 31,
    min_comp_area: int = 40,
    conn: int = 8,
    close_ksize: int = 3,
    close_iterations: int = 2,
) -> np.ndarray:
    """
    Construye una máscara binaria donde 255 ≈ texto y 0 ≈ fondo.

    Parámetros a tunear:
        blue_blur_ksize: tamaño del kernel (gaussiano) sobre el canal azul (suavizar tinta/reverso).
        bg_median_ksize: tamaño del kernel de medianBlur para estimar el fondo.
        min_comp_area: área mínima de componente conectado para considerarlo texto.
        conn: conectividad en connectedComponentsWithStats (4 u 8).
        close_ksize: tamaño de kernel para MORPH_CLOSE (rellena huecos en la máscara).
        close_iterations: número de iteraciones de MORPH_CLOSE.
    """

    # Asegurar que los tamaños de kernel sean impares y >= 1
    def _odd(k):
        k = max(1, int(k))
        return k if k % 2 == 1 else k + 1

    blue_blur_ksize = _odd(blue_blur_ksize)
    bg_median_ksize = _odd(bg_median_ksize)
    close_ksize = max(1, int(close_ksize))

    img = np.array(pil_img.convert("RGB"))
    b, g, r = cv2.split(img)

    # 1. Canal azul suavizado
    blue = cv2.GaussianBlur(b, (blue_blur_ksize, blue_blur_ksize), 0)

    # 2. Aplanar fondo (para localizar texto)
    bg = cv2.medianBlur(blue, bg_median_ksize)
    flat = cv2.divide(blue, bg, scale=255)
    flat = np.clip(flat, 0, 255).astype("uint8")

    # 3. Umbral global de Otsu → texto = blanco, fondo = negro
    _, bin_img = cv2.threshold(
        flat, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    bin_img = 255 - bin_img  # texto blanco

    # 4. Filtrar componentes por área
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_img, connectivity=conn
    )
    mask = np.zeros_like(bin_img)
    for i in range(1, num_labels):  # 0 = fondo
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_comp_area:
            mask[labels == i] = 255

    # 5. Cierre morfológico para unir trazos
    kernel_close = np.ones((close_ksize, close_ksize), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=close_iterations)

    return mask


def _apply_mask_keep_strokes(
    pil_img: Image.Image,
    mask: np.ndarray,
    dilate_ksize: int = 3,
    dilate_iterations: int = 1,
    use_hist_eq: bool = True,
) -> Image.Image:
    """
    Aplica la máscara sobre la imagen original en gris para mantener el trazado fino.

    Parámetros a tunear:
        dilate_ksize: tamaño del kernel de dilatación de la máscara (ensanchar región de texto).
        dilate_iterations: número de iteraciones de dilatación.
        use_hist_eq: si True, aplica equalizeHist al gris dentro de la máscara.
    """

    gray = np.array(pil_img.convert("L"))

    # Asegurar tamaño de máscara
    mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    dilate_ksize = max(1, int(dilate_ksize))
    kernel_dil = np.ones((dilate_ksize, dilate_ksize), np.uint8)

    # 1. Ensanchar ligeramente la máscara
    soft_mask = cv2.dilate(mask, kernel_dil, iterations=dilate_iterations)

    # 2. Hist EQ opcional
    eq_gray = cv2.equalizeHist(gray) if use_hist_eq else gray

    # 3. Componer: gris (mejorado) donde hay texto, blanco fuera
    text_only = np.where(soft_mask == 255, eq_gray, 255).astype("uint8")

    return Image.fromarray(text_only).convert("RGB")


def resize_long_side(pil_img: Image.Image, max_side: int) -> Image.Image:
    w, h = pil_img.size
    s = max(w, h)
    if s <= max_side: return pil_img
    scale = max_side / float(s)
    return pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)


def image_to_np_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"))


def preprocess_pipeline(
    pil_img: Image.Image,
    # Parámetros de la máscara
    blue_blur_ksize: int = 5,
    bg_median_ksize: int = 31,
    min_comp_area: int = 40,
    conn: int = 8,
    close_ksize: int = 3,
    close_iterations: int = 2,
    # Parámetros de aplicación de máscara
    dilate_ksize: int = 3,
    dilate_iterations: int = 1,
    use_hist_eq: bool = True
):
    """
    Preprocesado completo de carta manuscrita con bleed-through.

    Todos los parámetros son tunables; los valores por defecto son un buen punto de partida
    para la carta que has compartido.
    """

    mask = _build_text_mask(
        pil_img,
        blue_blur_ksize=blue_blur_ksize,
        bg_median_ksize=bg_median_ksize,
        min_comp_area=min_comp_area,
        conn=conn,
        close_ksize=close_ksize,
        close_iterations=close_iterations,
    )

    prepro_img = _apply_mask_keep_strokes(
        pil_img,
        mask,
        dilate_ksize=dilate_ksize,
        dilate_iterations=dilate_iterations,
        use_hist_eq=use_hist_eq,
    )

    return prepro_img
