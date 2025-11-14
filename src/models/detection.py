from paddleocr import PaddleOCR


def load_detection_model(lang: str = 'es'):
    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,  # no corrige orientación global
        use_doc_unwarping=False,  # no intenta “desdoblar” el documento
        use_textline_orientation=False  # fuerza a NO usar orientación de líneas
    )
