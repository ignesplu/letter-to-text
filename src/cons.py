
AVAILABLE_INFERENCE_TYPES = ["LLM", "OCR"]

DEFAULT_PREPRO_ARGS = {
    'blue_blur_ksize': 5,
    'bg_median_ksize': 21,
    'min_comp_area': 10,
    'conn': 8,
    'close_ksize': 3,
    'close_iterations': 1,
    'dilate_ksize': 3,
    'dilate_iterations': 1,
    'use_hist_eq': True,
}
DEFAULT_SYS_PROMPT = None
DEFAULT_USER_PROMPT = "Transcribe the following handwritten letter exactly as written."
