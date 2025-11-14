import numpy as np
from typing import List, Dict


def _levenshtein(a: List[str], b: List[str]) -> int:
    """
    Distancia de Levenshtein entre dos listas (de chars o de palabras).
    Implementación estándar O(len(a)*len(b)).
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[0, :] = np.arange(m + 1)
    dp[:, 0] = np.arange(n + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # borrado
                dp[i, j - 1] + 1,      # inserción
                dp[i - 1, j - 1] + cost  # sustitución
            )
    return int(dp[n, m])


def _compute_cer(gt_text: str, pred_text: str) -> float:
    """
    Character Error Rate: errores totales a nivel carácter.

    CER = Levenshtein(gold_chars, pred_chars) / |gold_chars|

    Valores típicos en OCR manuscrito:
      - Excelente: < 0.05
      - Muy bueno: 0.05 – 0.10
      - Aceptable: 0.10 – 0.20
      - Pobre: > 0.20
    """
    gt_chars = list(gt_text)
    pred_chars = list(pred_text)
    dist = _levenshtein(gt_chars, pred_chars)
    return dist / max(1, len(gt_chars))


def _compute_wer(gt_text: str, pred_text: str) -> float:
    """
    Word Error Rate: errores a nivel palabra.

    WER = Levenshtein(gold_words, pred_words) / |gold_words|

    Valores típicos en manuscrito:
      - Excelente: < 0.10
      - Muy bueno: 0.10 – 0.20
      - Aceptable: 0.20 – 0.35
      - Pobre: > 0.35
    """
    gt_words = gt_text.split()
    pred_words = pred_text.split()
    dist = _levenshtein(gt_words, pred_words)
    return dist / max(1, len(gt_words))


def _compute_line_cer(gt_text: str, pred_text: str) -> float:
    """
    CER promedio por línea: evalúa errores estructurales y de alineación.

    Una métrica útil para:

      - detectar si el modelo está mezclando líneas
      - ver si ciertos renglones salen peor
      - mejorar line-by-line prompting

    Interpretación:
      - Excelente: < 0.07
      - Buena: 0.07 – 0.15
      - Mala: > 0.15
    """
    gt_lines = gt_text.splitlines()
    pred_lines = pred_text.splitlines()

    # Alinear cantidad de líneas para emparejar
    max_len = max(len(gt_lines), len(pred_lines))
    gt_lines += [""] * (max_len - len(gt_lines))
    pred_lines += [""] * (max_len - len(pred_lines))

    cer_list = []
    for gt, pr in zip(gt_lines, pred_lines):
        cer_list.append(_compute_cer(gt, pr))

    return float(np.mean(cer_list)), float(np.std(cer_list))


def _compute_ser(gt_text: str, pred_text: str) -> float:
    """
    SER — Structural Error Rate

    Mide errores de ESTRUCTURA:
      - saltos de línea faltantes o extra
      - espacios dobles vs simples
      - merges o splits de líneas
      - guiones finales borrados
      - reflow

    SER = Levenshtein(gt_structure_tokens, pred_structure_tokens) / |gt_structure_tokens|

    Normaliza la estructura en tokens:
      - <NL> para saltos de línea
      - <SP> para espacios

    Buenas prácticas:
      - Excelente: < 0.05
      - Aceptable: 0.05 – 0.15
      - Pobre: > 0.15
    """
    def tokenize_structure(text):
        tokens = []
        for line in text.split("\n"):
            if line == "":
                tokens.append("<NL>")
            else:
                # Insertar <SP> entre palabras
                for w in line.split(" "):
                    tokens.append(w)
                    tokens.append("<SP>")
                tokens.append("<NL>")
        return tokens

    gt_tokens = tokenize_structure(gt_text)
    pred_tokens = tokenize_structure(pred_text)
    dist = _levenshtein(gt_tokens, pred_tokens)
    return dist / max(1, len(gt_tokens))


def evaluate_document(gt_text: str, pred_text: str) -> Dict[str, float]:
    """
    Evalúa un documento manuscrito usando múltiples métricas específicas:

    MÉTRICAS INCLUIDAS:
    -------------------
    1. CER  (Character Error Rate)
       - errores en caracteres individuales
       - cuanto más bajo mejor
         (Excelente < 0.05, Aceptable 0.10–0.20, Pobre > 0.20)

    2. WER  (Word Error Rate)
       - errores en palabras completas
       - captura normalización, splits, merges 
         (Excelente < 0.10, Aceptable < 0.25)

    3. SER  (Structural Error Rate)
       - errores de estructura: saltos de línea, espacios, guiones
       - especialmente importante en paleografía
         (Excelente < 0.05, Aceptable < 0.15)

    4. CER por línea
       - evalúa estabilidad línea a línea
         (Ideal < 0.10)

    Cómo interpretar:
    -----------------
    - CER alto → errores visuales; mejorar resolución o prompt literal.
    - WER alto → normalizaciones o palabras inventadas; revisar reglas estrictas.
    - SER alto → problemas en saltos de línea o estructura; revisar prompt line-by-line.
    - CER por línea variable → problemas localizados (sombras, inclinación, recortes).

    Retorna:
    --------
    Diccionario con todas las métricas (floats) y matriz de confusión.
    """
    line_cer_mean, line_cer_std = _compute_line_cer(gt_text, pred_text)
    return {
        "CER": _compute_cer(gt_text, pred_text),
        "WER": _compute_wer(gt_text, pred_text),
        "SER": _compute_ser(gt_text, pred_text),
        "Line_CER_mean": line_cer_mean,
        "Line_CER_std": line_cer_std,
    }


def explain_results(metrics: dict):
    """
    Explica los resultados de evaluación de OCR manuscrito, incluyendo:

    MÉTRICAS:
    ---------
    - CER (Character Error Rate)
        * Excelente < 0.05
        * Muy bueno: 0.05-0.10
        * Aceptable: 0.10-0.20
        * Pobre > 0.20
      → Mide errores a nivel carácter.

    - WER (Word Error Rate)
        * Excelente < 0.10
        * Muy bueno < 0.20
        * Aceptable < 0.35
        * Malo > 0.35
      → Evalúa errores a nivel palabra (splits, merges, normalizaciones).

    - SER (Structural Error Rate)
        * Excelente < 0.05
        * Aceptable < 0.15
        * Pobre > 0.15
      → Captura errores de estructura: saltos de línea, espacios, guiones, reflow.

    - Line_CER
        * Ideal: < 0.10
        * Mide consistencia línea por línea (importante para paleografía).

    Cuándo un resultado es bueno o malo:
    -----------------------------------
    - CER bajo = el modelo copia bien los caracteres visuales.
    - WER bajo = el modelo no inventa palabras ni corrige ortografía.
    - SER bajo = respeta saltos de línea y estructura original.
    - Line_CER bajo y estable = renglones uniformes, sin fallos localizados.

    Parámetros:
    -----------
    - metrics: diccionario de evaluate_document_ext
    """

    CER = metrics["CER"]
    WER = metrics["WER"]
    SER = metrics["SER"]
    Line_CER = metrics["Line_CER_mean"]

    # Accuracy
    chr_OK = 1 - CER
    word_OK = 1 - WER
    struct_OK = 1 - SER
    line_OK = 1 - Line_CER

    print("===== MÉTRICAS DE OCR MANUSCRITO =====")
    print(metrics)
    print()

    print(f"% Caracteres Correctos (1-CER):             {chr_OK*100:.2f}%")
    print(f"% Palabras Correctas (1-WER):               {word_OK*100:.2f}%")
    print(f"% Estructura Correcta (1-SER):              {struct_OK*100:.2f}%")
    print(f"% Caracteres Correctos x Línea (1-LineCER): {line_OK*100:.2f}%")
    print()

    print("===== INTERPRETACIÓN RÁPIDA =====")

    # CER
    if CER < 0.05:
        print(f"CER excelente ({CER:.3f}) — transcripción muy fiel a nivel carácter.")
    elif CER < 0.10:
        print(f"CER muy bueno ({CER:.3f}).")
    elif CER < 0.20:
        print(f"CER aceptable ({CER:.3f}). Algunos errores visuales presentes.")
    else:
        print(f"CER pobre ({CER:.3f}) — revisar prompt y preprocesado.")

    # WER
    if WER < 0.10:
        print(f"WER excelente ({WER:.3f}).")
    elif WER < 0.20:
        print(f"WER muy bueno ({WER:.3f}).")
    elif WER < 0.35:
        print(f"WER aceptable ({WER:.3f}).")
    else:
        print(f"WER alto ({WER:.3f}) — puede estar normalizando o inventando palabras.")

    # SER
    if SER < 0.05:
        print(f"SER excelente ({SER:.3f}) — respeta muy bien saltos de línea y estructura.")
    elif SER < 0.15:
        print(f"SER aceptable ({SER:.3f}).")
    else:
        print(f"SER pobre ({SER:.3f}) — mezcla líneas o rompe la estructura.")

    # Line CER
    if Line_CER < 0.10:
        print(f"Line CER bueno ({Line_CER:.3f}).")
    else:
        print(f"Line CER elevado ({Line_CER:.3f}) — problemas localizados en líneas específicas.")

    print()
