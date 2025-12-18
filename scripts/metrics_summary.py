from pathlib import Path
import csv
from collections import defaultdict


def _safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def main():
    log_dir = Path("logs")
    files = sorted(log_dir.glob("metrics_*.csv"))
    if not files:
        print("Nenhum arquivo metrics_*.csv encontrado em logs/")
        return

    latest = files[-1]
    print(f"Usando arquivo: {latest}")

    sum_iou = defaultdict(float)
    sum_fp = defaultdict(float)
    sum_fn = defaultdict(float)
    count = defaultdict(int)

    sum_iou_ref = defaultdict(float)
    sum_fp_ref = defaultdict(float)
    sum_fn_ref = defaultdict(float)
    count_ref = defaultdict(int)

    with latest.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scen = row.get("scenario", "unknown")

            iou = _safe_float(row.get("iou"))
            fp = _safe_float(row.get("fp"))
            fn = _safe_float(row.get("fn"))

            if iou is not None and fp is not None and fn is not None:
                sum_iou[scen] += iou
                sum_fp[scen] += fp
                sum_fn[scen] += fn
                count[scen] += 1

            iou_r = _safe_float(row.get("iou_vs_ref"))
            fp_r = _safe_float(row.get("fp_vs_ref"))
            fn_r = _safe_float(row.get("fn_vs_ref"))
            if iou_r is not None and fp_r is not None and fn_r is not None:
                sum_iou_ref[scen] += iou_r
                sum_fp_ref[scen] += fp_r
                sum_fn_ref[scen] += fn_r
                count_ref[scen] += 1

    print("\nResumo por cenário (RAW vs POST no mesmo frame):")
    for scen in sorted(count.keys()):
        n = count[scen]
        print(f"- {scen}:")
        print(f"    frames considerados: {n}")
        print(f"    IoU médio: {sum_iou[scen]/n:.3f}")
        print(f"    FP médios: {sum_fp[scen]/n:.2f}")
        print(f"    FN médios: {sum_fn[scen]/n:.2f}")

    if any(count_ref.values()):
        print("\nResumo extra (POST vs referência pré-ataque):")
        for scen in sorted(count_ref.keys()):
            n = count_ref[scen]
            print(f"- {scen}:")
            print(f"    frames considerados: {n}")
            print(f"    IoU médio (vs_ref): {sum_iou_ref[scen]/n:.3f}")
            print(f"    FP médios (vs_ref): {sum_fp_ref[scen]/n:.2f}")
            print(f"    FN médios (vs_ref): {sum_fn_ref[scen]/n:.2f}")


if __name__ == "__main__":
    main()
