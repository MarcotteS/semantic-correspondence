import json
import os
from pathlib import Path
import pandas as pd

#the methods of this files allow to build a comparison table of different metrics stored in json files.
#the results are available at the bottom of the file.
## the .json files are available on drive as well as the .pt files: https://drive.google.com/drive/u/0/folders/130C33edJ_vrh-boOOQ2n9IdUUNqlIvVi
def load_metrics(json_path: str) -> dict:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # unwrap payload
    if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
        data = data["metrics"]

    return data


def _get_overall_pck(overall: dict, t: float) -> float:
    """
    Get overall PCK at threshold t, handling keys like "0.1" (not "0.10").
    If not found, returns NaN.
    """
    if not isinstance(overall, dict) or not overall:
        return float("nan")

    candidates = [
        str(t),
        f"{t:.1f}",
        f"{t:.2f}",
        f"{t:.3f}",
    ]

    for k in candidates:
        if k in overall:
            return float(overall[k])

    return float("nan")


def build_metrics_comparison_table(metrics_paths: list[str]) -> pd.DataFrame:
    """
    Même comportement qu'avant: tu passes une liste de paths .json.
    """
    THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
    rows = []

    for path in metrics_paths:
        path = Path(path)
        metrics = load_metrics(path)
        overall = metrics.get("overall", {})

        row = {"name": path.stem}
        for t in THRESHOLDS:
            row[f"overall_pck@{t:.2f}"] = _get_overall_pck(overall, t)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values("name", key=lambda s: s.str.lower()).reset_index(drop=True)


def build_metrics_comparison_table_from_dir(metrics_dir: str) -> pd.DataFrame:
    """
    Nouveau: tu passes un dossier, ça prend tous les .json et fait le même tableau.
    """
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {metrics_dir}")

    json_paths = [str(p) for p in sorted(metrics_dir.glob("*.json"), key=lambda p: p.name.lower())]
    return build_metrics_comparison_table(json_paths)


# --- usage ---
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

#Compare with json file paths
paths = [
    r"C:\Users\guill\Downloads\metricsDinOV2with2epochsImages518with1Layers.json",
    r"C:\Users\guill\Downloads\metricsDinOV2with1epochsImages518with2Layers.json",
]
#df_files = build_metrics_comparison_table(paths)
#print(df_files.to_string(index=False))

#Compare with all json files in a directory, sorted by alphabetical order
metrics_dir = r"C:\Users\guill\Downloads\finetunedmodels\FineTunedModels"
df_dir = build_metrics_comparison_table_from_dir(metrics_dir)
print(df_dir.to_string(index=False))

"""                                    name  overall_pck@0.05  overall_pck@0.10  overall_pck@0.15  overall_pck@0.20

                       DINOv2StandardMetrics         36.923739         53.933068         63.384204         69.931392
DinOV2with1epochsImages518with1LayersMetrics         50.547958         64.004619         70.560864         74.803007
DinOV2with2epochsImages518with1Layersmetrics         41.701386         54.616883         61.758446         66.807807
DinOV2with1epochsImages518with2Layersmetrics         45.837107         59.510008         66.564396         71.314872

 
                       DINOv3StandardMetrics         35.411195         52.766960         61.733539         67.575401
DINOv3with1epochsImages518with1LayersMetrics         50.285300         66.644779         73.675392         78.459832
DINOv3with3epochsImages518with1LayersMetrics         50.287565         66.650439         73.682185         78.463228
DINOv3with3epochsImages518with2Layersmetrics         43.744905         60.729327         68.499230         73.656145

                          SamStandardMetrics         13.369486         22.668916         30.074269         36.442578
   SAMwith1epochsImages512with1LayersMetrics         29.691604         42.630876         51.106105         57.343085
   SAMwith4epochsImages512with1Layersmetrics         16.378725         26.800109         34.186215         40.527353
"""
   
