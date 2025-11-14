
```py
from src import Letter2TextInference
```

# LLM Inference

```py
import os
from PIL import Image
from src.models.llm import set_api_key

client = set_api_key(api_key=os.getenv('OPENAI_API_KEY'))
infer = Letter2TextInference(inference_type='LLM', llm_openai_client=client)
```

```py
PATH_IMAGE = "/content/drive/MyDrive/.../letter.jpg"
pil_img = Image.open(PATH_IMAGE)

system_prompt = """
You are an expert paleographer specialized in reading old Spanish handwriting from letters and manuscripts from 1960s. Your task is to **transcribe** the handwritten content in this image into **plain Spanish text**, maintaining accuracy and fidelity.

Guidelines:
1. **Accuracy first** - transcribe exactly what you see. Do not infer or guess missing words.
2. **Language** - the text is in Spanish; preserve the original spelling, accents, and punctuation, even if they appear old-fashioned or inconsistent.
3. **Line structure** - maintain natural **line breaks** as they appear in the image. Each original line of text should correspond to one line in your output.
4. **Illegible parts** - if a word or character cannot be confidently read, write `[?]` in its place. For partially legible words, write the readable part followed by `[?]` (e.g., 'querid[?]').
5. **Do not translate, interpret, or modernize** the content.
6. **Do not include explanations, metadata, or comments** — return **only** the transcribed text, nothing else
"""
user_prompt = "Transcribe the following handwritten letter exactly as written."

text = infer.run(
    img=pil_img,
    img_format='jpeg',
    img_resize=False,
    img_prepro=False,
    llm_model='gpt-5.1',
    llm_user_prompt=system_prompt,
    llm_sys_prompt=user_prompt,
    llm_report=True
)
print(text)
```

```
Beaspo 3-3-62

Mi querida esposa: Empezare por decirte que hemos
dado otro paso camino de la libertad, pues por lo visto
es que ya me han comunicado el dia que cumplo
con la sentencia. Dices bien es el Uno de Mayo asi es
que el citado dia - si estoy aqui - a las ocho de la
mañana, podras darte un abrazo a la puerta de
esta carcel.
Por todo ello, cuando recibas esta carta, podras
apreciar que son menos de dos meses lo que nos
falta para reunirnos definitivamente. Como
comprenderas, ya no es cosa de que te devuelvas mucho
y es conveniente que poco a poco vayas preparando
todas tus cosillas, con poco que lo demores, se te echa
el tiempo encima y en los ultimos momentos
siempre se corre con un poco de trastorno, aun
aun es posible que en los ultimos dias, suceda
un poco mas de bulla y no sepas donde acudir.
Por otra razon por la que he venido diciendote
tiempo para que mandes todo lo mio y dejar
el tiempo libre para lo tuyo, sin otra clase de
preocupaciones. Por ejemplo, si es lo que me vas
a mandar hubiese que cambiar alguna cosa,
casi no habria tiempo material de devolverte
para cambiarla, pues por pronto que llegues
sera al rededor del dia veinte, y a esas alturas,
el tiempo esta ya muy limitado.
Ya te decia en mi anterior lo que pensaba
que debias de decirle a Pedro, esta proxima se-
mana le escribes a ver carta, el dinero, cuando
lo recibas ya te dire yo como lo tienes que
mandar.

Celebro que te haya gustado la carta
```

# OCR Inference

```py
infer = Letter2TextInference(inference_type='OCR')
```

```py
PATH_IMAGE = "/content/drive/MyDrive/.../letter.jpg"
pil_img = Image.open(PATH_IMAGE)

prepro_args = {
    'blue_blur_ksize': 5,
    'bg_median_ksize': 21,
    'min_comp_area': 10,
    'conn': 8,
    'close_ksize': 3,
    'close_iterations': 1,
    'dilate_ksize': 3,
    'dilate_iterations': 1,
    'use_hist_eq': True
}

text = infer.run(
    img=pil_img,
    img_format='jpeg',
    img_resize=True,
    img_resize_max_side=2000,
    img_prepro=True,
    img_prepro_args=prepro_args,
    rec_model='qantev/trocr-large-spanish'
)
print(text)
```
```
1,
Berrío, 3- 3-69
1.2
12ª, se ema éxima: Bencherare hor desirle que kmum
dad año tromiro camino de la liberlod.  Fue haniró eñemmich
En que era "more" con "comunicado" el oka avec "Wemirlo
con 2 xaercinul.  Dreño  obía. es el Uro del Meno, una
que en "cermoso sir" - "«" aun aquec" - a los ocho de
1.4 mm avivana.  Izadre "barre en" a brero a la "huerher de
para carcer ».
Ver rock, el lo, cercado reciber enle carke, höbre
a creerías que con memor de por una ser de que ser
Uz.º a  para recemiruo, delimivivamente, homo
I aunse cercameía. Ya año es como de que le dessección mucha
un es  comunicacione que hoco a hoco veyar preberando
roción, ren comúlter, con rodo que le demure, se h sebe
sí riemizo excecima y surler exchiemen aunmunden
marra-Tener que-er “poco”atroheblade, mm
úcero, a7 hombre que en los sechimón, dúan áceder
1.1.ªn "2.000" 12250 reas resulta, y que ceben" chomale, acender",
You enter rezomor (por fue que ke vecuido ejerseñó
reicuelo Here Rue semendere, todo de mio y deferen
el riem ho höbre 1-are, lo huyo, ciurrohn (elme de
(rereciración, Tor ejcm'ido, ci en lo que une ve,
a amaúnder Newkiene que cambiar ademme era
coní uno tendre him fu unaherúal de devolaria
Nore escubiérle, buen, hor, bronb que llegme
vera aí realedor del día Uweuke, y a unen el kbre,
el hémico en la nuevy, identibado,
Y a le okeún en múi encheñor de que femek
que de bien de alecirle a Pedro, enk irmecima se
mona le embrlor a reu exerla «el deinero, exen
clo lo reciben ya le deic y o como de himen que
un ám dar
be lebro que le lueye puede de cark
```
