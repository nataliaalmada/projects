# lossaccu_direct.py
# Plota Loss (train) e mAP@0.50 (val) por época usando um RUN específico (caminho fixo)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== AJUSTE AQUI (RUN fixo) ======
RUN_DIR = Path("/mnt/c/Users/Nat Natalia/Desktop/P2_YOLO_NAS/checkpoints/fall_author_dataset/RUN_20250810_010447_075005")
OUT_FIG = Path("/mnt/c/Users/Nat Natalia/Desktop/P2_YOLO_NAS/plots/loss_map.png")

# Tags (ajuste se seu CSV tiver nomes diferentes)
TAG_LOSS_TRAIN = "ppyoloeloss/loss"
TAG_MAP_VAL    = "map@0.50"      # pode trocar por precision@0.50 / recall@0.50 / f1@0.50

# ====== nada abaixo deve precisar mudar ======
CSV_PATH = RUN_DIR / "train_history_tb.csv"
if not CSV_PATH.exists():
    raise SystemExit(f"CSV não encontrado: {CSV_PATH}")

hist = pd.read_csv(CSV_PATH)

# checar tags
metrics = set(hist["metric"].unique())
for tag in (TAG_LOSS_TRAIN, TAG_MAP_VAL):
    if tag not in metrics:
        sample = ", ".join(sorted(list(metrics))[:15])
        raise SystemExit(f"Tag '{tag}' não encontrada. Algumas disponíveis: {sample} ...")

# preparar séries
loss = (hist[hist.metric == TAG_LOSS_TRAIN]
        .sort_values("step")[["step", "scalar"]]
        .rename(columns={"scalar": "loss"})
        .reset_index(drop=True))
val  = (hist[hist.metric == TAG_MAP_VAL]
        .sort_values("step")[["step", "scalar"]]
        .rename(columns={"scalar": "map"})
        .reset_index(drop=True))

if val.empty:
    raise SystemExit(f"Nenhuma linha para '{TAG_MAP_VAL}' no CSV.")

# --- ALINHAMENTO ROBUSTO ---
# para cada step de validação s, pega o último loss conhecido com step <= s
loss_steps = loss["step"].to_numpy()
loss_vals  = loss["loss"].to_numpy()
val_steps  = val["step"].to_numpy()

# índices do "último <= s"
idx = np.searchsorted(loss_steps, val_steps, side="right") - 1
aligned_loss = np.where(idx >= 0, loss_vals[np.clip(idx, 0, len(loss_vals)-1)], np.nan)

# (opcional) suavizar loss
# aligned_loss = pd.Series(aligned_loss).rolling(3, min_periods=1).mean().to_numpy()

# montar DataFrame por época
val["epoch"] = np.arange(1, len(val) + 1)
loss_epoch = pd.DataFrame({"epoch": val["epoch"], "loss": aligned_loss})

# debug rápido se ainda não aparecer linha
if np.all(np.isnan(loss_epoch["loss"].to_numpy())):
    print("[AVISO] Todos os pontos de loss por época ficaram NaN.")
    print("Ex.: primeiros steps de loss:", loss_steps[:10])
    print("Ex.: steps de validação   :", val_steps[:10])

# plot
plt.figure()
plt.plot(loss_epoch["epoch"], loss_epoch["loss"], label="Train loss (alinhado por época)")
plt.plot(val["epoch"], val["map"], label=f"{TAG_MAP_VAL} (val)")
plt.xlabel("Época")
plt.title("Loss (train) e Métrica de Validação por Época")
plt.legend()

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figura salva em: {OUT_FIG}")
