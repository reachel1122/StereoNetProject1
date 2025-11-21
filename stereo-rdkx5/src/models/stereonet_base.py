import torch
import torch.nn as nn

class StereoNetBase(nn.Module):
    """极简可跑版：拼接左右图卷积回归视差；支持可选 refine 分支。"""
    def __init__(self, max_disp: int = 192, feature_ch: int = 32, refine: bool = False):
        super().__init__()
        self.max_disp = max_disp
        self.net = nn.Sequential(
            nn.Conv2d(6, feature_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(feature_ch, feature_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(feature_ch, 1, 3, padding=1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1+3, feature_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(feature_ch, 1, 3, padding=1),
        ) if refine else None

    def forward(self, img_l: torch.Tensor, img_r: torch.Tensor):
        x = torch.cat([img_l, img_r], dim=1)   # (B,6,H,W)
        disp = self.net(x)                     # (B,1,H,W)
        if self.refine is not None:
            disp = disp + self.refine(torch.cat([img_l, disp], dim=1))
        return disp
