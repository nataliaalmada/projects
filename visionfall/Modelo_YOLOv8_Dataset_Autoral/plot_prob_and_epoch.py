
#!/usr/bin/env python3
# plot_prob_and_epoch.py
# Gera:
#   - Histograma de probabilidades por classe (scores_videos.csv + labels_videos.csv)
#   - Curvas de Loss por época e F1 por época (a partir de runs/detect/trainX/results.csv)

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def hist_probs(scores_csv, labels_csv, outpath):
    df_s = pd.read_csv(scores_csv)
    df_l = pd.read_csv(labels_csv)
    df = df_s.merge(df_l, on="video", how="inner")
    if "score_agg" not in df.columns:
        raise ValueError("scores CSV precisa ter coluna 'score_agg'.")
    if "label" not in df.columns:
        raise ValueError("labels CSV precisa ter coluna 'label'.")

    s0 = df.loc[df["label"]==0, "score_agg"].to_numpy()
    s1 = df.loc[df["label"]==1, "score_agg"].to_numpy()

    plt.figure()
    bins = np.linspace(0,1,21)
    plt.hist(s0, bins=bins, alpha=0.5, label="Negativos (0)")
    plt.hist(s1, bins=bins, alpha=0.5, label="Positivos (1)")
    plt.xlabel("Score (score_agg)")
    plt.ylabel("Contagem")
    plt.title("Histograma de Probabilidades por Classe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def f1_from_pr(p, r):
    if p is None or r is None:
        return None
    if (p + r) == 0:
        return 0.0
    return 2.0 * p * r / (p + r)

def curves_from_results(results_csv, out_loss, out_f1):
    df = pd.read_csv(results_csv)
    # Losses
    loss_cols = [c for c in df.columns if c.endswith("box_loss") or c.endswith("cls_loss") or c.endswith("dfl_loss")]
    if not loss_cols:
        loss_candidates = ["train/box_loss","train/cls_loss","train/dfl_loss","val/box_loss","val/cls_loss","val/dfl_loss"]
        loss_cols = [c for c in loss_candidates if c in df.columns]

    # P/R
    p_cols = [c for c in df.columns if "metrics/precision" in c]
    r_cols = [c for c in df.columns if "metrics/recall" in c]
    p_col = p_cols[0] if p_cols else None
    r_col = r_cols[0] if r_cols else None

    if loss_cols:
        plt.figure()
        for c in loss_cols:
            plt.plot(df.index.values, df[c].values, label=c)
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title("Loss por Época")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_loss, dpi=160)
        plt.close()

    if p_col and r_col:
        f1 = [f1_from_pr(p, r) for p, r in zip(df[p_col].values, df[r_col].values)]
        plt.figure()
        plt.plot(df.index.values, f1)
        plt.xlabel("Época")
        plt.ylabel("F1 (derivado)")
        plt.title("F1 por Época (a partir de P/R do validation)")
        plt.tight_layout()
        plt.savefig(out_f1, dpi=160)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores_videos.csv (com score_agg)")
    ap.add_argument("--labels", required=True, help="labels_videos.csv (com video,label)")
    ap.add_argument("--results", required=True, help="runs/detect/trainX/results.csv")
    ap.add_argument("--outdir", default="figs", help="Pasta de saída")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    hist_probs(args.scores, args.labels, os.path.join(args.outdir, "hist_probabilidades.png"))
    curves_from_results(args.results,
                        os.path.join(args.outdir, "losses_por_epoca.png"),
                        os.path.join(args.outdir, "f1_por_epoca.png"))
    print(f"Figuras salvas em: {args.outdir}")

if __name__ == "__main__":
    main()
