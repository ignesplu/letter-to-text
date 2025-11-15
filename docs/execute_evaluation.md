
```py
from src import Letter2TextEvaluation
```

```py
real_text = """
Querido Carlos:

¡Cuánto tiempo sin saber de ti! Espero que cuando recibas
esta carta todo te esté yendo de maravilla por el norte.
Yo aquí sigo aunque con muchas ganas de que llegue el día.
"""
pred_text = """
Querido Carlos:

Cuánto tiempo sin caber de tii! Espero que cuando recivas
esta tarta todo te esté llendo de maravilla por el corte.
Yo aqui sigo aunque con muchas ganas de que llegue el día.
"""

metrics = Letter2TextEvaluation.evaluate(pred_text=pred_text, real_text=real_text, explain_metrics=True)
```

```
===== MÉTRICAS DE OCR MANUSCRITO =====
{'CER': 0.046875, 'WER': 0.22857142857142856, 'SER': 0.1038961038961039, 'Line_CER_mean': 0.026474231555901246, 'Line_CER_std': 0.03193375575627031}

% Caracteres Correctos (1-CER):             95.31%
% Palabras Correctas (1-WER):               77.14%
% Estructura Correcta (1-SER):              89.61%
% Caracteres Correctos x Línea (1-LineCER): 97.35%

===== INTERPRETACIÓN RÁPIDA =====
CER excelente (0.047) — transcripción muy fiel a nivel carácter.
WER aceptable (0.229).
SER aceptable (0.104).
Line CER bueno (0.026).
```
