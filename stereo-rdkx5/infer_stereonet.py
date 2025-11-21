# infer_stereonet.py
# Run GitHub StereoNet via the adapter, output disparity and (optional) point cloud.
import argparse, yaml, cv2 as cv, numpy as np, open3d as o3d
from stereonet_adapter import StereoNetAdapter

def load_calib(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    K = np.array(cfg['left']['K']).reshape(3,3).astype(np.float32)
    fx = float(K[0,0]); cx = float(K[0,2])
    # T is right->left, baseline = |tx|
    T = cfg['T']
    if isinstance(T, dict):
        tx = float(T.get('x', 0.0))
    else:
        tx = float(T[0])
    baseline_m = abs(tx)
    return K, fx, baseline_m

def disparity_to_depth(disp, fx, baseline_m, eps=1e-6):
    Z = fx * baseline_m / np.maximum(disp, eps)
    return Z

def depth_to_pcd(depth_m, K, color_bgr=None):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    h, w = depth_m.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    X = (xs - cx) * depth_m / fx
    Y = (ys - cy) * depth_m / fy
    Z = depth_m
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1,3)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if color_bgr is not None:
        col = color_bgr.reshape(-1,3)[:, ::-1] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(col)
    return pcd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--ckpt", required=False, help="Path to .pth/.pt checkpoint")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--pcd", default=None, help="Optional: save point cloud .ply")
    ap.add_argument("--h", type=int, default=352, help="Network input height")
    ap.add_argument("--w", type=int, default=640, help="Network input width")
    ap.add_argument("--max_disp", type=int, default=192)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    K, fx, baseline_m = load_calib(args.calib)
    L = cv.imread(args.left, cv.IMREAD_COLOR)
    R = cv.imread(args.right, cv.IMREAD_COLOR)
    assert L is not None and R is not None, "Cannot read images"

    net = StereoNetAdapter(max_disp=args.max_disp, input_size=(args.h, args.w), checkpoint=args.ckpt)
    disp = net(L, R)  # float32 HxW in pixel units

    # Save disparity (EXR) + color preview
    exr_path = os.path.join(args.out_dir, "disp.exr")
    cv.imwrite(exr_path, disp.astype(np.float32))
    disp_norm = (disp / max(1e-6, disp.max())).clip(0,1)
    disp_color = cv.applyColorMap((disp_norm*255).astype(np.uint8), cv.COLORMAP_JET)
    cv.imwrite(os.path.join(args.out_dir, "disp_color.png"), disp_color)

    # Optional point cloud
    if args.pcd:
        depth_m = disparity_to_depth(disp, fx, baseline_m)
        pcd = depth_to_pcd(depth_m, K, color_bgr=L)
        o3d.io.write_point_cloud(args.pcd, pcd)

    print("Saved disparity:", exr_path)
