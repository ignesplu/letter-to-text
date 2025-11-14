from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image


def load_recognition_model(model_name: str = "qantev/trocr-large-spanish", device: str = 'cpu'):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device).eval()

    return processor, model


@torch.inference_mode()
def recognize_text(pil_img: Image.Image, model, processor, device) -> str:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    ids = model.generate(pixel_values, num_beams=5, max_length=256, early_stopping=True)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
