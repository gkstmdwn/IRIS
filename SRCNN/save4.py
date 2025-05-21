#!/usr/bin/env python3
# test_compare.py

import os
import numpy as np
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import to_tensor
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

# — 사용자 설정 부분 —
WEIGHTS_FILE = "./data/output/best.pth"
IMAGE_FILE   = "./data/HIT-UAV Dataset/images/test/0_60_30_0_01614.jpg"
SCALE        = 4
OUTPUT_DIR   = "results"
# — 여기까지 —

def make_lr(image: pil_image.Image, scale: int):
    w, h = image.size
    w_cropped = (w // scale) * scale
    h_cropped = (h // scale) * scale
    image = image.crop((0, 0, w_cropped, h_cropped))
    return image.resize((w_cropped // scale, h_cropped // scale),
                        resample=pil_image.BICUBIC)

def bicubic_up(lr: pil_image.Image, scale: int):
    w, h = lr.size
    return lr.resize((w * scale, h * scale), resample=pil_image.BICUBIC)

def srcnn_sr(model, lr_tensor: torch.Tensor, device):
    with torch.no_grad():
        out = model(lr_tensor.to(device)).clamp(0.0, 1.0)
    return out.cpu()

def main():
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 모델 로드
    model = SRCNN(num_channels=1).to(device)
    state = torch.load(WEIGHTS_FILE, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # 2) 원본 HR 읽기
    hr = pil_image.open(IMAGE_FILE).convert('RGB')

    # 3) LR, Bicubic, SRCNN 생성
    lr  = make_lr(hr, SCALE)
    bic = bicubic_up(lr, SCALE)

    hr_np  = np.array(bic).astype(np.float32)
    ycbcr  = convert_rgb_to_ycbcr(hr_np)
    y      = ycbcr[..., 0] / 255.0
    y_t    = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    sr_y   = srcnn_sr(model, y_t, device).squeeze().numpy() * 255.0

    out_np = np.stack([sr_y, ycbcr[...,1], ycbcr[...,2]], axis=-1)
    sr     = pil_image.fromarray(
                 np.clip(convert_ycbcr_to_rgb(out_np), 0, 255)
                   .astype(np.uint8)
             )

    # 4) PSNR 계산 & 출력
    psnr_bic = calc_psnr(to_tensor(bic.convert('YCbCr'))[0:1],
                         to_tensor(hr.convert('YCbCr'))[0:1])
    psnr_sr  = calc_psnr(torch.from_numpy(sr_y/255.0).unsqueeze(0),
                         to_tensor(hr.convert('YCbCr'))[0:1])
    print(f'PSNR(bicubic): {psnr_bic:.2f} dB')
    print(f'PSNR(srcnn) : {psnr_sr:.2f} dB')

    # 5) 결과물 각각 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(IMAGE_FILE))[0]

    hr_small = hr.resize((hr.width//SCALE, hr.height//SCALE))
    hr_small.save(os.path.join(OUTPUT_DIR, f'{base}_HR.png'))
    lr.save   (os.path.join(OUTPUT_DIR, f'{base}_LR_x{SCALE}.png'))
    bic.save  (os.path.join(OUTPUT_DIR, f'{base}_Bicubic_x{SCALE}.png'))
    sr.save   (os.path.join(OUTPUT_DIR, f'{base}_SRCNN_x{SCALE}.png'))

    # (선택) composite도 저장하려면
    widths, heights = zip(*(i.size for i in [hr_small, lr, bic, sr]))
    comp = pil_image.new('RGB', (sum(widths), max(heights)))
    x_off = 0
    for im in [hr_small, lr, bic, sr]:
        comp.paste(im, (x_off, 0)); x_off += im.width
    comp.save(os.path.join(OUTPUT_DIR, f'{base}_compare_x{SCALE}.png'))

    print(f"Saved all images under '{OUTPUT_DIR}/'")

if __name__ == '__main__':
    main()
