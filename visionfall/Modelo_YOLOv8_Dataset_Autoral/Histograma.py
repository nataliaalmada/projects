import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

scores_csv = r"C:\Users\Nat Natalia\Desktop\Natalia\vision_temporaria\P2_YOLOv8\scores_videos.csv"
labels_csv = r"C:\Users\Nat Natalia\Desktop\Natalia\vision_temporaria\P2_YOLOv8\dataset\labels_videosYOLOMEU.csv"

df_s = pd.read_csv(scores_csv)
df_l = pd.read_csv(labels_csv)
df = df_s.merge(df_l, on="video", how="inner")

bins = np.linspace(0,1,21)
plt.figure(figsize=(7,4))
plt.hist(df.loc[df.label==0, "score_agg"], bins=bins, alpha=0.6, label="Negativos (0)")
plt.hist(df.loc[df.label==1, "score_agg"], bins=bins, alpha=0.6, label="Positivos (1)")
plt.xlabel("Probabilidade"); plt.ylabel("Quantidade de Imagens"); plt.title("Histograma de Probabilidades - Classe 'Fall'")
plt.legend(); plt.tight_layout(); plt.show()
