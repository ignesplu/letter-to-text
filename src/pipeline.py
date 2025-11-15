from PIL import Image
from openai import OpenAI

# Constants
from .cons import (
    AVAILABLE_INFERENCE_TYPES,
    DEFAULT_PREPRO_ARGS,
    DEFAULT_SYS_PROMPT,
    DEFAULT_USER_PROMPT
)
# Preprocessing
from .prepro import resize_long_side, preprocess_pipeline, image_to_np_rgb
# Model Inference
from .models.detection import load_detection_model
from .models.recognition import load_recognition_model, recognize_text
from .models.llm import req_api, response_tokens_cost
from .models.utils import image_to_bytes, extract_boxes_from_res, sanitize_box, sort_and_merge_boxes, crop_box_safe
# Validation
from .validation import evaluate_document, explain_results


class Letter2TextInference:

    def __init__(
            self,
            inference_type: str,
            llm_openai_client: OpenAI,
    ) -> None:

        if inference_type not in AVAILABLE_INFERENCE_TYPES:
            raise ValueError(f'Param `inference_type` must be one of the follwoing: {AVAILABLE_INFERENCE_TYPES}.')

        self.inference_type = inference_type
        self.llm_client = llm_openai_client

        # Pipeline has only been executed with these default params
        self.text_lang = 'es'
        self.device = 'cpu'

    def __infer_OCR(self) -> None:

        # Load models
        det_model = load_detection_model(self.text_lang)
        rec_processor, rec_model = load_recognition_model(model_name=self.rec_model, device=self.device)

        np_img = image_to_np_rgb(self.img)
        res_list = det_model.predict(np_img)
        res = res_list[0]
        raw_boxes = extract_boxes_from_res(res)

        # Sanitize all boxes once here
        w, h = self.img.size
        boxes = []
        for b in raw_boxes:
            sb = sanitize_box(b, w, h, pad=0)
            if sb:
                boxes.append(sb)

        # Group boxes into lines and recognize each crop with TrOCR
        lines = sort_and_merge_boxes(boxes, y_tol=8)  # Parametrizar y_tol

        # Recognition inference
        page_lines = []
        for line_boxes in lines:
            line_text_parts = []

            for b in line_boxes:
                crop = crop_box_safe(self.img, b, pad=2)
                if crop is None:
                    continue
                txt = recognize_text(
                    pil_img=crop,
                    model=rec_model,
                    processor=rec_processor,
                    device=self.device
                )
                if txt and txt.strip():
                    line_text_parts.append(txt.strip())
            if line_text_parts:
                text_line = " ".join(line_text_parts)
                page_lines.append(text_line)

        self.pred_text = " ".join(page_lines).strip()

    def __infer_LLM(self) -> None:

        img_b64 = image_to_bytes(image=self.img, image_format=self.img_format.upper())

        response = req_api(
            client=self.llm_client,
            img_b64=img_b64,
            img_format=self.img_format.lower(),
            model=self.llm_model,
            system_msg=self.llm_sys_prompt,
            user_msg=self.llm_user_prompt
        )

        if self.llm_report:
            response_tokens_cost(
                response=response,
                price_per_token=self.llm_price_per_token,
                price_completion=self.llm_price_completion
            )

        self.pred_text = response.choices[0].message.content

    def run(
            self,
            img: Image.Image,
            img_format: str = 'jpeg',
            img_resize: bool = False,
            img_resize_max_side: int = 2000,
            img_prepro: bool = False,
            img_prepro_args: dict = DEFAULT_PREPRO_ARGS,
            rec_model: str = 'qantev/trocr-large-spanish',
            llm_model: str = 'gpt-5.1',
            llm_user_prompt: str = DEFAULT_USER_PROMPT,
            llm_sys_prompt: str = DEFAULT_SYS_PROMPT,
            llm_report: bool = True,
            llm_price_per_token: float = 0.000002,
            llm_price_completion: float = 0.000006,
    ) -> str:

        # Resize if required
        if img_resize:
            img = resize_long_side(pil_img=img, max_side=img_resize_max_side)

        # Prepro if required
        if img_prepro:
            img = preprocess_pipeline(pil_img=img, **img_prepro_args)  # Revisar

        self.img_format = img_format
        self.img = img

        # Inference by type
        if self.inference_type == 'OCR':
            self.rec_model = rec_model
            self.__infer_OCR()
        else:
            if not llm_user_prompt or not llm_model:
                raise ValueError('Params `llm_user_prompt` and `llm_model` are mandatory if using LLM inference.')

            self.llm_model = llm_model
            self.llm_user_prompt = llm_user_prompt
            self.llm_sys_prompt = llm_sys_prompt
            self.llm_report = llm_report
            self.llm_price_per_token = llm_price_per_token
            self.llm_price_completion = llm_price_completion
            self.__infer_LLM()

        return self.pred_text


class Letter2TextEvaluation:

    def evaluate(
            pred_text: str,
            real_text: str,
            explain_metrics: bool = True
    ) -> dict:

        metrics_dict = evaluate_document(gt_text=real_text, pred_text=pred_text)

        if explain_metrics:
            explain_results(metrics_dict)

        return metrics_dict
