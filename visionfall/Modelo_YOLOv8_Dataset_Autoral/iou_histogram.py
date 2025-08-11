
#!/usr/bin/env python3
# iou_histogram.py
# Calcula histograma de IoU usando GT YOLO e predições YOLO (txts).
# Requisitos: pip install pillow matplotlib pandas numpy

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def yolo_to_xyxy(x, y, w, h, W, H):
    x1 = (x - w/2) * W
    y1 = (y - h/2) * H
    x2 = (x + w/2) * W
    y2 = (y + h/2) * H
    return x1, y1, x2, y2

def iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    union = area_a + area_b - inter + 1e-9
    return inter / union

def load_yolo_txt(path):
    arr = []
    if not os.path.isfile(path):
        return arr
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else None
            arr.append((cls, x, y, w, h, conf))
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, help="Pasta GT labels (YOLO)")
    ap.add_argument("--pred_dir", required=True, help="Pasta predições (YOLO)")
    ap.add_argument("--img_dir", required=True, help="Pasta imagens")
    ap.add_argument("--out", default="iou_hist.png", help="Figura saída")
    ap.add_argument("--iou_thr", type=float, default=0.5, help="IoU mínimo para TP (default 0.5)")
    args = ap.parse_args()

    ious = []
    gt_files = [f for f in os.listdir(args.labels_dir) if f.endswith(".txt")]
    for gt_name in gt_files:
        base = os.path.splitext(gt_name)[0]
        gt_path = os.path.join(args.labels_dir, gt_name)
        pred_path = os.path.join(args.pred_dir, base + ".txt")

        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            t = os.path.join(args.img_dir, base + ext)
            if os.path.isfile(t):
                img_path = t; break
        if img_path is None: continue

        W, H = Image.open(img_path).size
        gt = load_yolo_txt(gt_path)
        pr = load_yolo_txt(pred_path)

        gt_boxes = [(c, *yolo_to_xyxy(x, y, w, h, W, H)) for (c, x, y, w, h, _) in gt]
        pr_boxes = [(c, *yolo_to_xyxy(x, y, w, h, W, H)) for (c, x, y, w, h, _) in pr]

        for cls in sorted(set([g[0] for g in gt_boxes] + [p[0] for p in pr_boxes])):
            g = [g for g in gt_boxes if g[0] == cls]
            p = [p for p in pr_boxes if p[0] == cls]

            used_p = set()
            pairs = []
            for gi, gv in enumerate(g):
                for pi, pv in enumerate(p):
                    i = iou_xyxy(gv[1:], pv[1:])
                    pairs.append((i, gi, pi))
            pairs.sort(key=lambda x: x[0], reverse=True)

            matched_g = set()
            for i,i_g,i_p in pairs:
                if i <= 0: break
                if i_g in matched_g or i_p in used_p: continue
                if i >= args.iou_thr:
                    matched_g.add(i_g); used_p.add(i_p)
                    ious.append(i)

    if len(ious) == 0:
        print("Nenhum IoU coletado. Verifique paths e se existem TXT de predição.")
    else:
        import numpy as np
        plt.figure()
        bins = np.linspace(0,1,21)
        plt.hist(ious, bins=bins)
        plt.xlabel("IoU (TPs)")
        plt.ylabel("Contagem")
        plt.title("Histograma de IoU (true positives)")
        plt.tight_layout()
        plt.savefig(args.out, dpi=160)
        plt.close()
        print(f"Histograma salvo em: {args.out}")

if __name__ == "__main__":
    main()
