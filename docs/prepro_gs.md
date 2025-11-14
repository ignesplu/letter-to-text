
```py
import itertools
import math

from PIL import Image
import matplotlib.pyplot as plt

from src.prepro import preprocess_pipeline
```

```py
PATH_IMAGE = "/content/drive/MyDrive/.../letter.jpg"
pil_img = Image.open(PATH_IMAGE)
pil_img
```
<img src="img/letter.jpg">

```py
param_grid = {
    "max_side":         [2000],
    "blue_blur_ksize":  [3, 5],
    "bg_median_ksize":  [21, 31],
    "min_comp_area":    [10, 30, 60],
    "close_iterations": [1, 2],
    "dilate_iterations":[0, 1],
    "close_ksize":      [3],
    "conn":             [8],
    "dilate_ksize":     [3],
    "use_hist_eq":      [True],
}

param_names = list(param_grid.keys())
param_values = [param_grid[name] for name in param_names]
combinations = list(itertools.product(*param_values))
```

```py
results = []

for _i, combo in enumerate(combinations):
    params = dict(zip(param_names, combo))
    print(f"[{_i+1}/{len(combinations)}] {params}")

    label = (
        f"{_i}"
        f"_bb{params['blue_blur_ksize']}"
        f"_bg{params['bg_median_ksize']}"
        f"_area{params['min_comp_area']}"
        f"_cl{params['close_iterations']}"
        f"_dil{params['dilate_iterations']}"
    )

    proc_img = preprocess_pipeline(
        pil_img,
        max_side=params["max_side"],
        blue_blur_ksize=params["blue_blur_ksize"],
        bg_median_ksize=params["bg_median_ksize"],
        min_comp_area=params["min_comp_area"],
        conn=params["conn"],
        close_ksize=params["close_ksize"],
        close_iterations=params["close_iterations"],
        dilate_ksize=params["dilate_ksize"],
        dilate_iterations=params["dilate_iterations"],
        use_hist_eq=params["use_hist_eq"],
        np_ouput=False
    )

    results.append((proc_img, label))
```

```py
n = len(results)
n_cols = 6
n_rows = math.ceil(n / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 8*n_rows))
axes = axes.ravel()

for idx, (img, label) in enumerate(results):
    ax = axes[idx]
    ax.imshow(img, cmap="gray")
    ax.set_title(label, fontsize=8)
    ax.axis("off")

for j in range(idx + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
````
<img src="img/letter_prepro_subplots.png">