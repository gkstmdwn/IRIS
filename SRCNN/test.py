#!/usr/bin/env python3
# test_compare.py

import argparse
import os

import numpy as np
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import to_tensor, to_pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


def make_lr(image: pil_image.Image, scale: int):
    """HR → LR로 다운샘플"""
    w, h = image.size
    w_cropped = (w // scale) * scale
    h_cropped = (h // scale) * scale
    image = image.crop((0, 0, w_cropped, h_cropped))
    # Downsample
    lr = image.resize((w_cropped // scale, h_cropped // scale), resample=pil_image.BICUBIC)
    return lr


def bicubic_up(lr: pil_image.Image, scale: int):
    """LR → HR 크기로 업샘플 (bicubic)"""
    w, h = lr.size
    return lr.resize((w * scale, h * scale), resample=pil_image.BICUBIC)


def srcnn_sr(model, lr_tensor: torch.Tensor, device):
    """SRCNN 처리 (Y 채널만)"""
    # lr_tensor: [1,1,H,W] 0-1
    with torch.no_grad():
        out = model(lr_tensor.to(device)).clamp(0.0, 1.0)
    return out.cpu()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True,
                        help='SRCNN 학습된 .pth 파일 경로')
    parser.add_argument('--image-file', type=str, required=True,
                        help='원본 HR 이미지 파일 경로')
    parser.add_argument('--scale', type=int, default=3,
                        help='다운/업 스케일 배율 (예: 3)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='비교 이미지 저장 폴더')
    return parser.parse_args()


def main():
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    model = SRCNN(num_channels=1).to(device)
    state = torch.load(r"./data/output/best.pth", map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    # 원본 HR 이미지 읽기
    hr = pil_image.open(r"./data/HIT-UAV Dataset/images/test/0_60_30_0_01614.jpg").convert('RGB')

    # 단계별 이미지 생성
    lr = make_lr(hr, 4)                                # 1) LR
    bic = bicubic_up(lr, 4)                            # 2) bicubic
    # 3) SRCNN은 Y 채널만 처리하므로 YCbCr 변환
    hr_np = np.array(bic).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(hr_np)
    y = ycbcr[..., 0] / 255.
    y_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)     # [1,1,H,W]
    sr_y = srcnn_sr(model, y_tensor, device).squeeze(0).squeeze(0).numpy() * 255.0
    # YCbCr 합성 → RGB
    out_np = np.stack([sr_y, ycbcr[..., 1], ycbcr[..., 2]], axis=-1)
    sr = pil_image.fromarray(
        np.clip(convert_ycbcr_to_rgb(out_np), 0, 255).astype(np.uint8)
    )

    # PSNR 계산
    psnr_bic = calc_psnr(to_tensor(bic.convert('YCbCr'))[0:1], to_tensor(hr.convert('YCbCr'))[0:1])
    psnr_sr  = calc_psnr(torch.from_numpy(sr_y/255.0).unsqueeze(0), 
                         to_tensor(hr.convert('YCbCr'))[0:1])
    print(f'PSNR(bicubic): {psnr_bic:.2f} dB')
    print(f'PSNR(srcnn) : {psnr_sr:.2f} dB')

    # 4장 좌우 결합
    imgs = [
        hr.resize((hr.width//4, hr.height//4)),  # 보기 쉽도록 축소
        lr.resize((hr.width//4, hr.height//4)),
        bic.resize((hr.width//4, hr.height//4)),
        sr.resize((hr.width//4, hr.height//4))
    ]
    widths, heights = zip(*(i.size for i in imgs))
    total_w = sum(widths)
    max_h = max(heights)
    comp = pil_image.new('RGB', (total_w, max_h))
    x_offset = 0
    for im in imgs:
        comp.paste(im, (x_offset, 0))
        x_offset += im.width

    # 저장
    base = os.path.splitext(os.path.basename(r"./data/HIT-UAV Dataset/images/test/0_60_30_0_01614.jpg"))[0]
    out_path = os.path.join(r"./data/out", f'{base}_compare_x{4}.png')
    comp.save(out_path)
    print('Saved comparison image to', out_path)
    
    


if __name__ == '__main__':
    main()
