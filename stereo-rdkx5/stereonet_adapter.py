# stereonet_adapter.py
# Generic wrapper to use a GitHub StereoNet (or fallback to a simple placeholder).
# - Adjust IMPORT_PATH and MODEL_CLASS according to the repo you added.
# - Ensures disparity is returned in *pixel units* at the original image size.
import os, cv2 as cv, numpy as np, torch

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TODO: CHANGE THESE TWO LINES to your GitHub repo's model entrypoint
# Example: from stereonet.model import StereoNet as GitStereoNet
IMPORT_PATH = "disparity.models.stereonet_disp"
MODEL_CLASS = "StereoNet"


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def _dynamic_import(import_path, class_name):
    if import_path is None or class_name is None:
        return None
    mod = __import__(import_path, fromlist=[class_name])
    return getattr(mod, class_name)

class StereoNetAdapter(torch.nn.Module):
    def __init__(self, max_disp=192, input_size=(352, 640), mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                 repo_args: dict=None, checkpoint: str=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.max_disp = max_disp
        self.input_h, self.input_w = input_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1,1,3)
        self.std = np.array(std, dtype=np.float32).reshape(1,1,3)
        self.device = device

        ModelClass = _dynamic_import(IMPORT_PATH, MODEL_CLASS)
        if ModelClass is None:
            # Fallback to a very simple placeholder so wiring can be tested
            from models.stereonet_baseline import SimpleStereoNet as ModelClass  # assumes your project has it
        self.net = ModelClass(max_disp=max_disp, **(repo_args or {}))
        self.net.to(self.device).eval()

        if checkpoint and os.path.isfile(checkpoint):
            ckpt = torch.load(checkpoint, map_location=self.device)
            # Try common keys
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.net.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                self.net.load_state_dict(ckpt, strict=False)

    @torch.inference_mode()
    def forward(self, left_bgr: np.ndarray, right_bgr: np.ndarray):
        """Inputs: left/right BGR uint8 HxWx3
           Returns: disp (float32, HxW) in *original pixel* units.
        """
        H0, W0 = left_bgr.shape[:2]
        assert right_bgr.shape[:2] == (H0, W0), "Left/Right must have same size"
        # Resize to network input
        inpL = cv.resize(left_bgr, (self.input_w, self.input_h), interpolation=cv.INTER_LINEAR)
        inpR = cv.resize(right_bgr, (self.input_w, self.input_h), interpolation=cv.INTER_LINEAR)
        # BGR->RGB, to float [0,1], normalize
        L = inpL[:, :, ::-1].astype(np.float32) / 255.0
        R = inpR[:, :, ::-1].astype(np.float32) / 255.0
        L = (L - self.mean) / self.std
        R = (R - self.mean) / self.std
        tenL = torch.from_numpy(L).permute(2,0,1).unsqueeze(0).to(self.device)  # 1x3xHxW
        tenR = torch.from_numpy(R).permute(2,0,1).unsqueeze(0).to(self.device)

        disp_small = self.net(tenL, tenR)  # expected Bx1xHxW or BxHxW
        if isinstance(disp_small, (list, tuple)):
            disp_small = disp_small[-1]
        if disp_small.ndim == 4:
            disp_small = disp_small[:,0]
        disp_small = disp_small.float().squeeze(0).detach().cpu().numpy()  # H_in x W_in, pixel units (network scale)

        # Upsample back to original size; scale disparity with width ratio
        disp_up = cv.resize(disp_small, (W0, H0), interpolation=cv.INTER_LINEAR)
        scale = float(W0) / float(self.input_w)
        disp = disp_up * scale
        return disp
