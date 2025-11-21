from pathlib import Path
from PIL import Image
import torch, numpy as np, random
from torch.utils.data import Dataset
import torch.nn.functional as F

def read_png_16_as_float(path: Path) -> torch.Tensor:
    disp_u16 = np.array(Image.open(path), dtype=np.uint16)
    disp = disp_u16.astype(np.float32) / 256.0
    return torch.from_numpy(disp).unsqueeze(0)  # (1,H,W)

class Kitti2015(Dataset):
    def __init__(self, root, split="train", crop_hw=(256,512)):
        self.root = Path(root)
        self.crop_h, self.crop_w = crop_hw
        # 以 GT 列表为准（只包含 *_10.png，数量约 200）
        disp_dir = self.root/"training"/"disp_occ_0"
        self.disps = sorted(disp_dir.glob("*.png"))
        assert len(self.disps) > 0, f"no GT disparity found in {disp_dir}"
        self.triples = []
        for pD in self.disps:
            name = pD.name
            pL = self.root/"training"/"image_2"/name
            pR = self.root/"training"/"image_3"/name
            if pL.exists() and pR.exists():
                self.triples.append((pL, pR, pD))
            else:
                raise FileNotFoundError(f"missing L/R for {name}")
        if split == "train":
            self.triples = self.triples[:160]
        elif split == "val":
            self.triples = self.triples[160:]

    def __len__(self): return len(self.triples)

    def _random_crop(self, L, R, D):
        _,H,W = L.shape
        th, tw = self.crop_h, self.crop_w
        if H < th or W < tw:
            pad_h = max(0, th-H); pad_w = max(0, tw-W)
            L = F.pad(L, (0,pad_w,0,pad_h))
            R = F.pad(R, (0,pad_w,0,pad_h))
            D = F.pad(D, (0,pad_w,0,pad_h))
            _,H,W = L.shape
        import random
        y = random.randint(0, H-th)
        x = random.randint(0, W-tw)
        return L[:,y:y+th,x:x+tw], R[:,y:y+th,x:x+tw], D[:,y:y+th,x:x+tw]

    def __getitem__(self, idx):
        pL,pR,pD = self.triples[idx]
        L = torch.from_numpy(np.array(Image.open(pL), dtype=np.uint8)).permute(2,0,1).float()/255.0
        R = torch.from_numpy(np.array(Image.open(pR), dtype=np.uint8)).permute(2,0,1).float()/255.0
        D = read_png_16_as_float(pD)
        L,R,D = self._random_crop(L,R,D)
        return L, R, D
