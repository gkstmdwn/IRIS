import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 디바이스 설정 (GPU가 있으면 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. VGG 특징 추출기 정의
class VGGPerceptual(nn.Module):
    def __init__(self, layers=('relu2_2', 'relu3_3')):
        super().__init__()
        # 사전학습된 VGG19 feature 부분 로드
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        # 이름 → 인덱스 매핑
        self.layer_map = {
            'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 17, 'relu4_4': 28
        }
        max_idx = max(self.layer_map.values())
        # 필요한 레이어까지만 잘라낸 Sequential
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max_idx+1)])
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.layers = layers

    def forward(self, x):
        feats = {}
        out = x
        for idx, layer in enumerate(self.vgg):
            out = layer(out)
            # 해당 인덱스가 우리가 뽑고 싶은 레이어면 저장
            for name, li in self.layer_map.items():
                if idx == li and name in self.layers:
                    feats[name] = out.clone()
        return feats

# 2. 1채널→3채널 복제 + ImageNet 정규화
def gray_to_vgg_input(x_gray):
    # x_gray: [B,1,H,W], 0~1 범위
    x_rgb = x_gray.repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485,0.456,0.406], device=x_rgb.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=x_rgb.device).view(1,3,1,1)
    return (x_rgb - mean) / std

# 3. 이미지 로드 및 전처리
img = Image.open(r"./data/HIT-UAV Dataset/images/test/0_100_70_0_08227.jpg").convert("L")  # 그레이스케일로 로드
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),   # [0,1]
])
x_gray = transform(img).unsqueeze(0).to(device)  # [1,1,224,224]
x_vgg = gray_to_vgg_input(x_gray)                # [1,3,224,224]

# 4. 특징 맵 추출
vgg_net = VGGPerceptual(layers=('relu2_2',)).to(device)
features = vgg_net(x_vgg)

# 5. relu2_2 첫 4개 채널 플롯
feat_map = features['relu2_2'][0]  # [C, H, W]
for c in range(4):
    plt.figure()
    plt.imshow(feat_map[c].detach().cpu().numpy(), cmap='viridis')
    plt.title(f"relu2_2 Channel {c}")
    plt.axis('off')
    plt.show()

# 1. 특징 맵 불러오기 (예: relu2_2)
#    feats: [1, C, H, W]
feats = features['relu2_2']  # 이미 GPU/CPU에 있는 텐서

# 2. 채널 평균 맵 계산: [1, C, H, W] → [1, 1, H, W]
mean_map = feats.mean(dim=1, keepdim=True)

# 3. (선택) 해상도 키우기: 112×112 → 224×224
mean_up = F.interpolate(mean_map, size=(224,224), mode='bilinear', align_corners=False)

# 4. 2D 배열로 변환
heatmap = mean_up[0,0].detach().cpu().numpy()  # [H, W]

# 5. 시각화
plt.figure(figsize=(5,5))
plt.imshow(heatmap, cmap='magma')
plt.title("Mean over all relu2_2 channels")
plt.axis('off')
plt.show()