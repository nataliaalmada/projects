
#!/usr/bin/env python3
# gradcam_yolov8.py
# Gera Grad-CAM/saliency para imagens usando YOLOv8.
# Requisitos: pip install ultralytics torch torchvision grad-cam

import argparse
import numpy as np
import torch
import cv2
from ultralytics import YOLO

def overlay_heatmap(img_bgr, cam, alpha=0.4):
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_uint8 = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_bgr, 1.0, heat, alpha, 0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best.pt do YOLOv8")
    ap.add_argument("--image", required=True, help="caminho da imagem")
    ap.add_argument("--out", default="gradcam_out.jpg", help="arquivo de saída")
    ap.add_argument("--target-class", type=int, default=0, help="classe-alvo (default: 0)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    m = model.model  # nn.Module

    # Escolher uma camada conv "profunda"
    target_layer = None
    for mod in reversed(list(m.modules())):
        if isinstance(mod, torch.nn.Conv2d):
            target_layer = mod
            break
    if target_layer is None:
        raise RuntimeError("Não foi possível localizar uma camada convolucional para Grad-CAM.")

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import preprocess_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except Exception:
        raise RuntimeError("Instale a dependência: pip install grad-cam")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = preprocess_image(rgb, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    targets = [ClassifierOutputTarget(args.target_class)]
    cam_extractor = GradCAM(model=m, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]

    out = overlay_heatmap(img_bgr, grayscale_cam)
    cv2.imwrite(args.out, out)
    print(f"Grad-CAM salvo em: {args.out}")

if __name__ == "__main__":
    main()
