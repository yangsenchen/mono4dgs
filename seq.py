import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from PIL import Image
from torch import optim
from tqdm import tqdm
import glob  # 补上了这个

try:
    from gsplat import rasterization
except ImportError:
    print("Error: gsplat not installed.")
    exit()

# === Utils ===
def ssim(img1, img2, window_size=11, size_average=True):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    channel = img1.size(1)
    window = create_window(window_size, channel)
    if img1.is_cuda: window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))).mean()

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class SequentialSolver:
    def __init__(self, data_path: str, focal_ratio: float = 0.8):
        self.device = torch.device("cuda:0")
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)

        if 'images' in data: self.images = torch.from_numpy(data['images'].astype(np.float32) / 255.0).to(self.device)
        else: self.images = torch.from_numpy(data['image'].astype(np.float32) / 255.0).to(self.device)
        
        depth_key = next((k for k in ['d', 'depth', 'depths'] if k in data), None)
        self.depths = torch.from_numpy(data[depth_key].astype(np.float32)).to(self.device)
        
        pose_key = 'cam_c2w' if 'cam_c2w' in data else 'c2w'
        self.c2ws = torch.from_numpy(data[pose_key].astype(np.float32)).to(self.device)
        
        self.num_frames, self.H, self.W, _ = self.images.shape
        print(f"Data Loaded: {self.num_frames} frames, {self.W}x{self.H}")

        self.focal = float(self.W) * focal_ratio
        self.K = torch.tensor([[self.focal, 0, self.W / 2.0], [0, self.focal, self.H / 2.0], [0, 0, 1]], device=self.device)
        
        # Load Masks
        mask_root = "/root/autodl-tmp/learn-genmojo/data/car-turn/Annotations"
        self.gt_masks = self._load_mask_from_folder(os.path.join(mask_root, "001"))
        if self.gt_masks is None: print("Error: Masks not found."); exit()
        self.gt_masks = torch.from_numpy(self.gt_masks).to(self.device)

        self.grad_accum = None
        self.denom = None
        
        # Init Frame 0
        self._init_frame_zero()

    def _load_mask_from_folder(self, folder_path):
        if not os.path.exists(folder_path): return None
        files = sorted(glob.glob(os.path.join(folder_path, "*")))[:self.num_frames]
        loaded = []
        for f in files:
            img = Image.open(f).convert('L').resize((self.W, self.H), Image.NEAREST)
            loaded.append((np.array(img).astype(np.float32) / 255.0 > 0.5).astype(np.float32))
        return np.stack(loaded, axis=0)[..., None] if loaded else None

    def _align_shape(self, tensor):
        if tensor.shape[0] == self.W and tensor.shape[1] == self.H: return tensor.permute(1, 0, 2)
        return tensor

    def _init_frame_zero(self):
        print("Initializing Frame 0 Geometry...")
        idx = 0
        # Use Mask + Depth to initialize points
        mask = self.gt_masks[idx, ..., 0] > 0.5
        depth = self.depths[idx]
        
        ys, xs = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        ys, xs = ys.to(self.device), xs.to(self.device)
        
        # Filter valid depth
        valid = (depth > 0.01) & (depth < 100.0)
        
        ys, xs, d = ys[valid], xs[valid], depth[valid]
        x_c = (xs - self.W/2.0) * d / self.focal
        y_c = (ys - self.H/2.0) * d / self.focal
        xyz_cam = torch.stack([x_c, y_c, d], dim=-1)
        
        c2w = self.c2ws[idx]
        self.means = (c2w[:3, :3] @ xyz_cam.T).T + c2w[:3, 3]
        
        N = self.means.shape[0]
        self.scales = torch.ones((N, 3), device=self.device) * -4.0 
        self.quats = torch.rand((N, 4), device=self.device); self.quats[:, 0] = 1.0
        
        rgb_vals = self.images[idx][valid]
        self.rgbs = torch.log(torch.clamp(rgb_vals, 0.01, 0.99) / (1 - rgb_vals))
        # Start visible!
        self.opacities = torch.ones((N), device=self.device) * 4.0 
        
        self.means.requires_grad_(True); self.scales.requires_grad_(True)
        self.quats.requires_grad_(True); self.rgbs.requires_grad_(True); self.opacities.requires_grad_(True)
        
        self.grad_accum = torch.zeros(N, device=self.device)
        self.denom = torch.zeros(N, device=self.device)
        print(f"[*] Frame 0 Initialized with {N} points.")

    def _densify_and_prune(self):
        with torch.no_grad():
            grads = self.grad_accum / self.denom
            grads[self.denom == 0] = 0.0
            
            # Clone based on gradients
            to_clone = grads >= 0.0002
            
            if to_clone.sum() > 0:
                self.means = torch.cat([self.means, self.means[to_clone]]).detach().requires_grad_(True)
                self.scales = torch.cat([self.scales, self.scales[to_clone]]).detach().requires_grad_(True)
                self.rgbs = torch.cat([self.rgbs, self.rgbs[to_clone]]).detach().requires_grad_(True)
                self.quats = torch.cat([self.quats, self.quats[to_clone]]).detach().requires_grad_(True)
                self.opacities = torch.cat([self.opacities, self.opacities[to_clone]]).detach().requires_grad_(True)
            
            self.grad_accum = torch.zeros(self.means.shape[0], device=self.device)
            self.denom = torch.zeros(self.means.shape[0], device=self.device)

    
    def run(self):
        out_dir = "results_sequential"
        os.makedirs(out_dir, exist_ok=True)
        frames = []
        
        print("=== Starting Sequential Training ===")
        
        for t in range(self.num_frames):
            # Frame 0 needs more time to Densify geometry
            # Subsequent frames just need to track (move existing points)
            iters = 2000 if t == 0 else 300
            densify = True if t == 0 else False
            
            rgb_np = self.train_frame(t, iters, densify)
            frames.append(Image.fromarray(rgb_np))
            
            # Save debug image
            frames[-1].save(f"{out_dir}/frame_{t:03d}.png")
            
        frames[0].save(f"{out_dir}/result.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"Done! Saved to {out_dir}/result.gif")

if __name__ == "__main__":
    tyro.cli(lambda: SequentialSolver("/root/autodl-tmp/learn-genmojo/data/car-turn/MegaSAM_Outputs/car-turn_sgd_cvd_hr.npz").run())