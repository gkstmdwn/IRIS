import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGGPerceptual(nn.Module):
    def __init__(self, layers=('relu2_2', 'relu3_3')):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        self.selected_idxs = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 17,
            'relu4_4': 28,
        }
        max_idx = max(self.selected_idxs.values())
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max_idx + 1)])
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.layers = layers
    
    def forward(self, x):
        feats = {}
        out = x
        for idx, layer in enumerate(self.vgg):
            out = layer(out)
            for name, layer_idx in self.selected_idxs.items():
                if idx == layer_idx and name in self.layers:
                    feats[name] = out
        return feats

if __name__ == "__main__":
    pass