import argparse
from pathlib import Path
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from super_gradients.training import models

# ---------- utils ----------
def read_yolo_labels(label_path):
    """
    Retorna lista de (cls, xc, yc, w, h) em coordenadas relativas [0..1].
    Se o arquivo não existir, retorna [].
    """
    if not os.path.exists(label_path):
        return []
    items = []
    with open(label_path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) >= 5:
                c, xc, yc, w, h = parts[:5]
                items.append((int(float(c)), float(xc), float(yc), float(w), float(h)))
    return items

def yolo_rel_to_xyxy(xc, yc, w, h, W, H):
    """Converte bbox YOLO relativa para (x1,y1,x2,y2) absolutos (pixels)."""
    bw = w * W
    bh = h * H
    cx = xc * W
    cy = yc * H
    x1 = max(0.0, cx - bw / 2.0)
    y1 = max(0.0, cy - bh / 2.0)
    x2 = min(W - 1.0, cx + bw / 2.0)
    y2 = min(H - 1.0, cy + bh / 2.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def iou_xyxy(a, b):
    """IoU entre duas caixas (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union

def greedy_match(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    gt_boxes: [N,4], pred_boxes: [M,4]
    Retorna lista de (gt_idx, pred_idx, iou) para pares com IoU>=thr usando matching guloso.
    """
    matches = []
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return matches
    gt_used = set()
    pred_used = set()
    # calcula matriz IoU
    iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou_mat[i, j] = iou_xyxy(g, p)
    # ordena todas as combinações por IoU desc
    pairs = [(i, j, iou_mat[i, j]) for i in range(len(gt_boxes)) for j in range(len(pred_boxes))]
    pairs.sort(key=lambda x: x[2], reverse=True)
    for i, j, v in pairs:
        if v < iou_thresh:
            break
        if i in gt_used or j in pred_used:
            continue
        gt_used.add(i); pred_used.add(j)
        matches.append((i, j, float(v)))
    return matches

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Gráficos de IoU (YOLO-NAS / Super-Gradients)")
    ap.add_argument("--ckpt", required=True, help="Caminho do checkpoint .pth")
    ap.add_argument("--model-name", required=True, help="Ex.: yolo_nas_l")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--images-dir", required=True, help="Diretório de imagens (val/test)")
    ap.add_argument("--labels-dir", required=True, help="Diretório de labels YOLO correspondentes")
    ap.add_argument("--class-id", type=int, default=0, help="Classe alvo (ex.: 0 para 'fall')")
    ap.add_argument("--conf", type=float, default=0.20, help="Confiança mínima para filtrar detecções")
    ap.add_argument("--iou-thresh", type=float, default=0.50, help="Limiar para considerar match")
    ap.add_argument("--limit", type=int, default=0, help="Limitar nº de imagens (0 = tudo)")
    ap.add_argument("--out-dir", required=True, help="Saída dos gráficos/CSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carrega modelo
    print(f"[INFO] Carregando {args.model_name} de {args.ckpt} ...")
    model = models.get(args.model_name, num_classes=args.num_classes, checkpoint_path=args.ckpt)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Coleta imagens
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    img_paths = []
    for e in exts:
        img_paths += glob.glob(os.path.join(args.images_dir, e))
    img_paths = sorted(img_paths)
    if args.limit and args.limit > 0:
        img_paths = img_paths[:args.limit]
    print(f"[INFO] Imagens encontradas: {len(img_paths)}")

    rows = []
    with torch.inference_mode():
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            W, H = img.size

            # GT (somente classe alvo)
            stem = Path(p).stem
            gt_file = os.path.join(args.labels_dir, stem + ".txt")
            gt_ann = [r for r in read_yolo_labels(gt_file) if r[0] == args.class_id]
            gt_boxes = np.stack([yolo_rel_to_xyxy(xc, yc, w, h, W, H) for (_, xc, yc, w, h) in gt_ann], axis=0) if gt_ann else np.zeros((0,4), np.float32)

            # Predições (usa pipeline interno do SG)
            preds = model.predict(img, conf=args.conf)
            # às vezes vem lista
            if isinstance(preds, list):
                preds = preds[0]
            pred_struct = getattr(preds, "prediction", None)
            if pred_struct is None:
                pred_boxes = np.zeros((0,4), np.float32)
                pred_labels = np.zeros((0,), np.int32)
                pred_confs = np.zeros((0,), np.float32)
            else:
                boxes_xyxy = np.array(getattr(pred_struct, "bboxes_xyxy", []), dtype=np.float32)
                labels     = np.array(getattr(pred_struct, "labels", []), dtype=np.int32)
                confs      = np.array(getattr(pred_struct, "confidence", []), dtype=np.float32)
                m = labels == args.class_id
                pred_boxes = boxes_xyxy[m] if boxes_xyxy.size else np.zeros((0,4), np.float32)
                pred_confs = confs[m] if confs.size else np.zeros((0,), np.float32)

            # matching guloso por IoU
            matches = greedy_match(gt_boxes, pred_boxes, iou_thresh=args.iou_thresh)

            # salva linhas por match
            for (gi, pj, v) in matches:
                rows.append({
                    "image": Path(p).name,
                    "iou": v,
                    "conf": float(pred_confs[pj]) if len(pred_confs) > pj else np.nan,
                    "gt_idx": gi,
                    "pred_idx": pj
                })

            # opcional: GTs não casados (IoU=0) -> úteis para analisar falhas
            for gi in range(len(gt_boxes)):
                if not any(m[0] == gi for m in matches):
                    rows.append({
                        "image": Path(p).name,
                        "iou": 0.0,
                        "conf": np.nan,
                        "gt_idx": gi,
                        "pred_idx": -1
                    })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "iou_matches.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV salvo: {csv_path}")

    # ---- Gráficos ----
    if len(df) == 0:
        print("[AVISO] Nenhum match/GT encontrado. Nada para plotar.")
        return

    # 1) Histograma IoU
    plt.figure()
    vals = df["iou"].values
    plt.hist(vals, bins=30, edgecolor="black")
    plt.xlabel("IoU")
    plt.ylabel("Contagem")
    plt.title(f"Histograma de IoU (classe {args.class_id}, thr={args.iou_thresh})")
    plt.tight_layout()
    out_hist = out_dir / "hist_iou.png"
    plt.savefig(out_hist, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Hist salvo: {out_hist}")

    # 2) Dispersão Confiança x IoU (apenas pares com conf disponível)
    df_scatter = df.dropna(subset=["conf"])
    if not df_scatter.empty:
        plt.figure()
        plt.scatter(df_scatter["conf"].values, df_scatter["iou"].values, s=12, alpha=0.6)
        plt.xlabel("Confiança da predição")
        plt.ylabel("IoU do match")
        plt.title(f"Confiança × IoU (classe {args.class_id})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_sc = out_dir / "scatter_conf_iou.png"
        plt.savefig(out_sc, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] Scatter salvo: {out_sc}")

    # 3) Boxplot IoU
    plt.figure()
    plt.boxplot(vals, vert=True, showmeans=True)
    plt.ylabel("IoU")
    plt.title(f"Distribuição de IoU (classe {args.class_id})")
    plt.tight_layout()
    out_box = out_dir / "boxplot_iou.png"
    plt.savefig(out_box, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Boxplot salvo: {out_box}")

    # resumo estatístico
    summ = {
        "n": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75))
    }
    pd.DataFrame([summ]).to_csv(out_dir / "iou_summary.csv", index=False)
    print(f"[OK] Summary salvo: {out_dir/'iou_summary.csv'}")

if __name__ == "__main__":
    main()
