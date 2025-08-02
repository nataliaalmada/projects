import os
import cv2
import subprocess
import csv
from shutil import copy2

# Pasta com os vídeos originais
input_folder = r"C:\Users\Nat Natalia\Desktop\TCC2\P1_OpenPifPaf_LSTM\videos"

# Pasta para salvar os vídeos modificados
output_folder = r"C:\Users\Nat Natalia\Desktop\TCC2\P1_OpenPifPaf_LSTM\videos_convertidos"
os.makedirs(output_folder, exist_ok=True)

# Arquivo de log
log_csv = os.path.join(output_folder, "log_fps.csv")

# Lista de registros para o log
log_data = [["Arquivo", "FPS Original", "Ação"]]

# Loop pelos vídeos
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps < 29.9 or fps > 30.1:
            print(f"Ajustando '{filename}' de {fps:.2f} FPS para 30 FPS...")
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-filter:v", "fps=fps=30",
                "-c:a", "copy",
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log_data.append([filename, f"{fps:.2f}", "Convertido para 30 FPS"])
        else:
            print(f" '{filename}' já está em 30 FPS. Copiando...")
            copy2(input_path, output_path)
            log_data.append([filename, f"{fps:.2f}", "Copiado sem alteração"])

# Salva o log CSV
with open(log_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(log_data)

print(f"\nTodos os vídeos foram processados.")
print(f"Log salvo em: {log_csv}")
print(f"Vídeos salvos em: {output_folder}")
