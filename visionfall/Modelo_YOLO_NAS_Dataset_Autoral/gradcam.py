import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

from super_gradients.training import models as sg_models
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_image(path, force_size=640):
    img = Image.open(path).convert("RGB")
    if force_size and force_size > 0:
        img = img.resize((force_size, force_size), Image.LANCZOS)
    return np.array(img)

def preprocess_image(np_img):
    # [H,W,3] -> [1,3,H,W], float32 0..1
    tensor = torch.from_numpy(np_img / 255.0).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor

def find_last_conv2d(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

class DetectorWrapper(nn.Module):
    """Envolve o detector para devolver um tensor [B,1] que o EigenCAM aceita."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        # tenta pegar um tensor dentre as saídas (tuplas/listas)
        if isinstance(out, (list, tuple)):
            tensors = [o for o in out if torch.is_tensor(o)]
            if len(tensors) == 0:
                return torch.zeros(x.size(0), 1, device=x.device)
            t = tensors[0]
        elif torch.is_tensor(out):
            t = out
        else:
            return torch.zeros(x.size(0), 1, device=x.device)
        # colapsa dimensões espaciais/canais para [B,1]
        while t.dim() > 2:
            t = t.mean(dim=-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        # média nos canais -> [B,1]
        return t.mean(dim=1, keepdim=True)

def main():
    ap = argparse.ArgumentParser(description="EigenCAM para YOLO-NAS (SuperGradients)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-name", required=True, help="ex.: yolo_nas_l")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--images", required=True, help="arquivo ou pasta")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--size", type=int, default=640, help="NxN (múltiplo de 32); 0 para manter")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Carregando {args.model_name} de {args.ckpt} ...")
    base_model = sg_models.get(args.model_name, num_classes=args.num_classes, checkpoint_path=args.ckpt)
    base_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)

    # camada alvo: última Conv2d
    target_layer = find_last_conv2d(base_model)
    if target_layer is None:
        raise SystemExit("Nenhuma nn.Conv2d encontrada no modelo para aplicar CAM.")

    # wrapper que devolve [B,1]
    wrapped = DetectorWrapper(base_model)
    wrapped.eval().to(device)

    cam = EigenCAM(model=wrapped, target_layers=[target_layer])

    p = Path(args.images)
    if p.is_file():
        img_paths = [p]
    elif p.is_dir():
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        img_paths = sorted([q for q in p.glob("*") if q.suffix.lower() in exts])
    else:
        raise FileNotFoundError(f"Caminho não encontrado: {p}")

    with torch.inference_mode():
        for ip in img_paths:
            print(f"[INFO] Processando {ip} ...")
            rgb = load_image(ip, force_size=args.size)
            rgb_f32 = (rgb.astype(np.float32) / 255.0).clip(0, 1)
            inp = preprocess_image(rgb).to(device)

            grayscale = cam(input_tensor=inp)[0]  # [H,W] 0..1
            vis = show_cam_on_image(rgb_f32, grayscale, use_rgb=True, image_weight=args.alpha)

            out_path = out_dir / f"cam_{ip.stem}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"[OK] Salvo: {out_path}")

if __name__ == "__main__":
    main()
