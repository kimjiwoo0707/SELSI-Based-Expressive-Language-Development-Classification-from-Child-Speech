import torch.nn as nn
import torchvision.models as models

class CustomSwinTransformer(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Linear(self.swin.head.in_features, num_classes)

    def forward(self, x):
        return self.swin(x)
