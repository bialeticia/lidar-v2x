from pathlib import Path
import csv
from collections import defaultdict

def main():
    log_dir = Path("logs")
    files = sorted(log_dir.glob("metrics_*.csv"))
    if not files:
        print("Nenhum arquivo metrics_*.csv encontrado em logs/")
        return

    latest = files[-1]
    print(f"Usando arquivo: {latest}")

    sum_iou = defaultdict(float)
    sum_fp  = defaultdict(float)
    sum_fn  = defaultdict(float)
    count   = defaultdict(int)

    with latest.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scen = row.get("scenario", "unknown")
            try:
                iou = float(row["iou"])
                fp  = float(row["fp"])
                fn  = float(row["fn"])
            except (KeyError, ValueError):
                continue

            sum_iou[scen] += iou
            sum_fp[scen]  += fp
            sum_fn[scen]  += fn
            count[scen]   += 1

    print("\nResumo por cenário:")
    for scen in sorted(count.keys()):
        n = count[scen]
        iou_med = sum_iou[scen] / n if n else 0.0
        fp_med  = sum_fp[scen]  / n if n else 0.0
        fn_med  = sum_fn[scen]  / n if n else 0.0
        print(f"- {scen}:")
        print(f"    frames considerados: {n}")
        print(f"    IoU médio: {iou_med:.3f}")
        print(f"    FP médios: {fp_med:.2f}")
        print(f"    FN médios: {fn_med:.2f}")

if __name__ == "__main__":
    main()