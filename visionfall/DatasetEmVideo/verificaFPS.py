import cv2
import os

# Caminho da pasta com os vídeos
folder = r"C:\Users\Nat Natalia\Desktop\TCC2\P1_OpenPifPaf_LSTM\videos"

# Verifica todos os arquivos de vídeo da pasta
for filename in os.listdir(folder):
    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        path = os.path.join(folder, filename)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps:
            status = ">= 30 FPS" if fps >= 30 else "< 30 FPS"
            print(f"{filename}: {fps:.2f} FPS ({status})")
        else:
            print(f"{filename}: Não foi possível ler o FPS.")
