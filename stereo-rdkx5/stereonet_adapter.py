import torch
from stereonet.model import StereoNet

class TeacherStereoNet:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = StereoNet(in_channels=1).to(device).eval()

    @torch.no_grad()
    def infer(self, left_right_tensor):
        return self.model(left_right_tensor.to(self.device))
