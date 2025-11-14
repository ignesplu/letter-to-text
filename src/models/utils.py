from typing import Optional, Tuple
import numpy as np
from PIL import Image
import io
import base64


def _poly_to_xyxy_any(poly) -> tuple | None:
    """
    Accepts:
      - array/list of shape (N, 2)  -> N points
      - flat array/list [x1,y1,x2,y2,...] with even length >= 8
    Returns:
      (x1, y1, x2, y2) or None if invalid
    """
    arr = np.asarray(poly)
    if arr.size < 8:
        return None

    if arr.ndim == 2 and arr.shape[1] == 2:
        pts = arr
    elif arr.ndim == 1 and arr.size % 2 == 0:
        pts = arr.reshape(-1, 2)
    else:
        return None

    xs, ys = pts[:, 0], pts[:, 1]
    # quick sanity filter (non-finite)
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        return None

    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    # discard degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def extract_boxes_from_res(res: dict) -> list[tuple]:
    """
    Works with PaddleOCR 3.3.x result dict.
    Tries rec_boxes -> rec_polys -> dt_polys, handling shapes (N,4,2) or (N,8).
    """
    boxes = []

    # Preferred: text-line boxes from recognizer
    rb = res.get("rec_boxes", None)
    if isinstance(rb, np.ndarray) and rb.size > 0:
        # rb can be (N,4,2) or (N,8)
        if rb.ndim == 3 and rb.shape[1:] == (4, 2):
            for i in range(rb.shape[0]):
                xyxy = _poly_to_xyxy_any(rb[i])
                if xyxy: boxes.append(xyxy)
        elif rb.ndim == 2 and rb.shape[1] in (8, 16):  # sometimes flattened per row
            for i in range(rb.shape[0]):
                xyxy = _poly_to_xyxy_any(rb[i])
                if xyxy: boxes.append(xyxy)

    # Fallback: polygons lists from recognizer
    if not boxes and isinstance(res.get("rec_polys"), list) and len(res["rec_polys"]) > 0:
        for poly in res["rec_polys"]:
            xyxy = _poly_to_xyxy_any(poly)
            if xyxy: boxes.append(xyxy)

    # Fallback: detector polygons
    if not boxes and isinstance(res.get("dt_polys"), list) and len(res["dt_polys"]) > 0:
        for poly in res["dt_polys"]:
            xyxy = _poly_to_xyxy_any(poly)
            if xyxy: boxes.append(xyxy)

    return boxes


def sanitize_box(box: Tuple[int,int,int,int], img_w: int, img_h: int,
                  pad: int = 2, min_w: int = 6, min_h: int = 6) -> Optional[Tuple[int,int,int,int]]:
    """Clamp to image bounds, apply padding, and ensure left<right & top<bottom."""
    x1, y1, x2, y2 = map(int, box)

    # Ensure ordering first
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    # Apply padding
    x1 -= pad; y1 -= pad
    x2 += pad; y2 += pad

    # Clip to image bounds
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    # Re-enforce ordering after clipping
    if x1 >= x2 or y1 >= y2:
        return None

    # Discard tiny boxes (often noise)
    if (x2 - x1) < min_w or (y2 - y1) < min_h:
        return None

    return (x1, y1, x2, y2)


def crop_box_safe(pil_img, box: Tuple[int,int,int,int], pad: int = 2):
    """Sanitize then crop; returns None if invalid."""
    w, h = pil_img.size
    b = sanitize_box(box, w, h, pad=pad)
    if b is None:
        return None
    return pil_img.crop(b)


def sort_and_merge_boxes(boxes, y_tol=8):
    """
    boxes: list of (x_min,y_min,x_max,y_max)
    Returns groups (lines) sorted top->bottom then left->right.
    """
    items = []
    for b in boxes:
        x1,y1,x2,y2 = b
        y_mean = 0.5*(y1+y2)
        x_mean = 0.5*(x1+x2)
        items.append((y_mean, x_mean, b))
    items.sort(key=lambda t: (t[0], t[1]))

    lines, cur = [], []
    for it in items:
        if not cur:
            cur.append(it); continue
        if abs(it[0] - cur[-1][0]) <= y_tol:
            cur.append(it)
        else:
            cur.sort(key=lambda t: t[1])
            lines.append([t[2] for t in cur])
            cur = [it]
    if cur:
        cur.sort(key=lambda t: t[1])
        lines.append([t[2] for t in cur])
    return lines


def image_to_bytes(image: Image.Image, image_format: str = "JPEG"):
    img_buffer = io.BytesIO()
    image.save(img_buffer, format=image_format)
    img_bytes = img_buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")
