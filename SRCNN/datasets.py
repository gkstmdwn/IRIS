import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self, hr_dir:str, lr_size = (120, 60), hr_crop_size = (480, 240), transform=None):
        self.hr_paths = sorted(Path(hr_dir).rglob("*.[pj][pn]g"))
        assert len(self.hr_paths) > 0, f"SRCNN/datsets.py:10 \n HR Image not found: {hr_dir}"
        self.lr_size = lr_size
        self.hr_crop_size = hr_crop_size
        self.transform = transform if transform is not None else transforms.ToTensor()
    
    def __len__(self):
        return len(self.hr_paths)
    
    def __getitem__(self, idx):
        # HR 이미지 로드 및 RGB 변환
        hr = Image.open(self.hr_paths[idx]).convert("L")
        # HR Center Crop
        crop_w, crop_h = self.hr_crop_size
        w, h = hr.size
        left = max((w - crop_w) // 2, 0)
        top = max((h - crop_h) // 2, 0)
        right = left + crop_w
        bottom = top + crop_h
        hr_cropped = hr.crop((left, top, right, bottom))
        # LR 생성: 지정된 크기로 다운샘플링 -> 다시 HR 크기로 업샘플링
        lr = hr_cropped.resize(self.lr_size, resample=Image.BICUBIC)
        lr_up = lr.resize((crop_w, crop_h), resample=Image.BICUBIC)
        # Tensor 변환
        hr_tensor = self.transform(hr_cropped)
        lr_tensor = self.transform(lr_up)
        return lr_tensor, hr_tensor

class EvalDataset(Dataset):
    def __init__(self, hr_dir: str, lr_size=(120, 60), hr_crop_size=(480, 240), transform=None):
        """
        Eval용 Dataset: SRDataset와 동일하게 동작하지만, 순차적 접근 보장
        :param hr_dir: High-resolution 이미지가 저장된 디렉토리 경로
        :param lr_size: 생성할 Low-resolution 이미지 크기 (width, height)
        :param hr_crop_size: HR 이미지를 자를 크기 (width, height)
        :param transform: PIL Image -> Tensor 변환 함수
        """
        self.hr_paths = sorted(Path(hr_dir).rglob("*.[pj][pn]g"))
        assert len(self.hr_paths) > 0, f"HR 이미지가 없습니다: {hr_dir}"
        self.lr_size = lr_size
        self.hr_crop_size = hr_crop_size
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        # SRDataset과 동일 처리
        hr = Image.open(self.hr_paths[idx]).convert("L")
        crop_w, crop_h = self.hr_crop_size
        w, h = hr.size
        left = max((w - crop_w) // 2, 0)
        top = max((h - crop_h) // 2, 0)
        hr_cropped = hr.crop((left, top, left + crop_w, top + crop_h))
        lr = hr_cropped.resize(self.lr_size, resample=Image.BICUBIC)
        lr_up = lr.resize((crop_w, crop_h), resample=Image.BICUBIC)
        hr_tensor = self.transform(hr_cropped)
        lr_tensor = self.transform(lr_up)
        return lr_tensor, hr_tensor
