# parse_epoch_log_sg.py
import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser(description="Parse 'SUMMARY OF EPOCH' from Super-Gradients console log.")
    ap.add_argument("--log", required=True, help="Caminho do arquivo console_*.txt")
    ap.add_argument("--out-csv", required=True, help="CSV de saída com métricas por época")
    ap.add_argument("--plot", default=None, help="(Opcional) caminho da figura PNG")
    return ap.parse_args()

# Regex básicas
RE_EPOCH_HDR   = re.compile(r"^\s*SUMMARY\s+OF\s+EPOCH\s+(\d+)\s*$", re.I)
RE_TRAIN_HDR   = re.compile(r"^\s*├──\s*Train\b", re.I)
RE_VAL_HDR     = re.compile(r"^\s*└──\s*Validation\b", re.I)

# Ex.: "Ppyoloeloss/loss_cls = 1.9448" (serve para loss, loss_cls, loss_iou, loss_dfl)
RE_LOSS_LINE   = re.compile(r"Ppyoloeloss/(loss(?:_cls|_iou|_dfl)?)\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.I)

# Ex.: "Precision@0.50 = 0.0", "Map@0.50 = 0.0001"
RE_METRIC_LINE = re.compile(r"\b(Precision|Recall|Map|F1)@0\.50\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.I)

# Ex.: "Best_score_threshold = 0.0"
RE_BEST_TH     = re.compile(r"Best_score_threshold\s*=\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.I)

def parse_console_log(path: Path) -> pd.DataFrame:
    rows = []
    current = None
    scope = None  # None | 'train' | 'val'

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            ln = raw.rstrip("\n")

            # Novo epoch
            m = RE_EPOCH_HDR.search(ln)
            if m:
                # salva anterior
                if current:
                    rows.append(current)
                current = {"epoch": int(m.group(1))}
                scope = None
                continue

            # Entrando no bloco Train / Validation
            if RE_TRAIN_HDR.search(ln):
                scope = "train"
                continue
            if RE_VAL_HDR.search(ln):
                scope = "val"
                continue

            if current is None:
                continue  # ainda não iniciou nenhum epoch

            # Coletar losses
            m = RE_LOSS_LINE.search(ln)
            if m and scope in ("train", "val"):
                key = m.group(1).lower()  # loss, loss_cls, loss_iou, loss_dfl
                val = float(m.group(2))
                current[f"{scope}_{key}"] = val
                continue

            # Coletar métricas (normalmente só aparecem no Validation)
            m = RE_METRIC_LINE.search(ln)
            if m:
                metric_name = m.group(1).lower()  # precision, recall, map, f1
                val = float(m.group(2))
                # prefixa com val_ para deixar claro
                current[f"val_{metric_name}"] = val
                continue

            # Best score threshold (fica no Validation)
            m = RE_BEST_TH.search(ln)
            if m:
                current["val_best_score_threshold"] = float(m.group(1))
                continue

    if current:
        rows.append(current)

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return df

def make_plot(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()

    # linhas de loss
    if "train_loss" in df.columns:
        ax1.plot(df["epoch"], df["train_loss"], label="Train loss")
    if "val_loss" in df.columns:
        ax1.plot(df["epoch"], df["val_loss"], label="Val loss")

    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.25)

    # mAP em eixo secundário
    if "val_map" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df["epoch"], df["val_map"], linestyle="--", label="mAP@0.50 (val)")
        ax2.set_ylabel("mAP@0.50")

        # juntar legendas dos dois eixos
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    args = parse_args()
    log_path = Path(args.log)
    out_csv  = Path(args.out_csv)
    df = parse_console_log(log_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if args.plot:
        make_plot(df, Path(args.plot))

if __name__ == "__main__":
    main()
