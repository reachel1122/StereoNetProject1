import torch, torch.nn.functional as F
from torch import optim
from src.models.stereonet_base import StereoNetBase

def main(steps=10, h=64, w=128, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    m = StereoNetBase().to(device).train()
    opt = optim.Adam(m.parameters(), lr=1e-3)
    for t in range(steps):
        imgL = torch.rand(2,3,h,w, device=device)
        imgR = torch.rand(2,3,h,w, device=device)
        gt   = torch.rand(2,1,h,w, device=device) * 10.0
        pred = m(imgL, imgR)
        loss = F.smooth_l1_loss(pred, gt)
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"step {t+1}/{steps}  loss={loss.item():.4f}")
    print("✅ sanity train done")

if __name__ == "__main__":
    main()
