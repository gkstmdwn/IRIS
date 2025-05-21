import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptual(nn.Module):
    def __init__(self, layer_indices=(3, 8, 17, 26), use_input_norm=True, device='cuda'):
        super().__init__()
        self.device = device

        # 1) VGG19 features 모듈 불러오기 (weights API 사용)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg_features = vgg
        self.layer_indices = set(layer_indices)

        # 2) ImageNet 정규화 상수
        if use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
            std  = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        else:
            self.mean = None
            self.std  = None

    def forward(self, x):
        # (1) 흑백→RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # (2) 정규화
        if self.mean is not None:
            x = (x - self.mean) / self.std

        features = []
        # (3) 레이어 순회하며 지정된 인덱스일 때만 저장
        for idx, layer in enumerate(self.vgg_features):
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)
        return features


class VGGLoss(nn.Module):
    def __init__(self, layer_indices=(3,8,17,26), layer_weights=None, device='cuda'):
        super().__init__()
        self.device = device
        self.feature_extractor = VGGPerceptual(layer_indices, device=device).to(device)
        num = len(layer_indices)
        if layer_weights is None:
            layer_weights = [1.0/num]*num
        assert len(layer_weights) == num
        self.layer_weights = layer_weights
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        f_pred   = self.feature_extractor(pred)
        # target은 그래디언트 제외
        f_target = self.feature_extractor(target.detach())
        loss = 0
        for w, p, t in zip(self.layer_weights, f_pred, f_target):
            loss = loss + w * self.criterion(p, t)
        return loss