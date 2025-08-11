# hist_prob.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser(description="Histograma de probabilidades por imagem (ex.: prob_fall).")
    ap.add_argument("--csv", required=True, help="CSV com colunas: image, label_fall, prob_fall (ou especifique --col).")
    ap.add_argument("--col", default="prob_fall", help="Nome da coluna de probabilidade (default: prob_fall).")
    ap.add_argument("--label-col", default="label_fall", help="Coluna binária 0/1 para overlay por classe (default: label_fall).")
    ap.add_argument("--bins", type=int, default=30, help="Número de bins do histograma (default: 30).")
    ap.add_argument("--out", required=True, help="Caminho do PNG de saída.")
    ap.add_argument("--overlay", action="store_true", help="Plota histogramas separados por classe (0 vs 1) sobrepostos.")
    ap.add_argument("--vline", type=float, default=None, help="Desenha linha vertical no limiar (ex.: --vline 0.5).")
    ap.add_argument("--stats-out", default=None, help="Se definido, salva CSV com estatísticas descritivas.")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    if args.col not in df.columns:
        raise SystemExit(f"Coluna '{args.col}' não encontrada. Colunas disponíveis: {list(df.columns)}")

    probs = df[args.col].astype(float).to_numpy()

    # Estatísticas simples
    stats = {
        "count": len(probs),
        "mean": float(np.mean(probs)) if len(probs) else np.nan,
        "std": float(np.std(probs)) if len(probs) else np.nan,
        "min": float(np.min(probs)) if len(probs) else np.nan,
        "25%": float(np.percentile(probs, 25)) if len(probs) else np.nan,
        "50%": float(np.percentile(probs, 50)) if len(probs) else np.nan,
        "75%": float(np.percentile(probs, 75)) if len(probs) else np.nan,
        "max": float(np.max(probs)) if len(probs) else np.nan,
    }

    # Plot
    plt.figure(figsize=(9, 5))

    if args.overlay and args.label_col in df.columns:
        # Overlay: histogramas separados para 0 e 1
        m0 = df[args.label_col].astype(int) == 0
        m1 = df[args.label_col].astype(int) == 1

        # Usamos o mesmo range/bins para serem comparáveis
        common_range = (0.0, 1.0) if np.all((probs >= 0) & (probs <= 1)) else (float(np.min(probs)), float(np.max(probs)))
        bins = args.bins

        plt.hist(df.loc[m0, args.col].astype(float), bins=bins, range=common_range, alpha=0.5, label=f"{args.label_col}=0", density=False)
        plt.hist(df.loc[m1, args.col].astype(float), bins=bins, range=common_range, alpha=0.5, label=f"{args.label_col}=1", density=False)
        plt.legend()
        plt.title(f"Histograma de {args.col} (overlay por {args.label_col})")
    else:
        # Um único histograma
        plt.hist(probs, bins=args.bins, alpha=0.8)
        plt.title(f"Histograma de {args.col}")

    if args.vline is not None:
        plt.axvline(args.vline, linestyle="--")

    plt.xlabel("Probabilidade")
    plt.ylabel("Contagem")
    plt.grid(True, alpha=0.25)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()

    # Exporta stats opcionais
    if args.stats_out:
        pd.DataFrame([stats]).to_csv(args.stats_out, index=False)

if __name__ == "__main__":
    main()
