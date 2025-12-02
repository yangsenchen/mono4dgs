import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
from PIL import Image
from torch import optim
from tqdm import tqdm
import cv2 
from gsplat import rasterization

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


class GSSolver:
    def __init__(self, data_path: str, sam2_video_path: str, output_dir: str, focal_ratio: float = 0.8):
        self.device = torch.device("cuda:0")
        self.output_dir = output_dir
        
        # [NEW] 创建分类子文件夹
        self.dir_params = os.path.join(self.output_dir, "params")
        self.dir_images = os.path.join(self.output_dir, "images")
        self.dir_depths = os.path.join(self.output_dir, "depths")
        os.makedirs(self.dir_params, exist_ok=True)
        os.makedirs(self.dir_images, exist_ok=True)
        os.makedirs(self.dir_depths, exist_ok=True)
        
        print(f"Output Directory set to: {self.output_dir}")

        print(f"Loading data from {data_path}...")
        data = np.load(data_path)

        # 加载图像
        if 'images' in data: self.images = torch.from_numpy(data['images'].astype(np.float32) / 255.0).to(self.device)
        else: self.images = torch.from_numpy(data['image'].astype(np.float32) / 255.0).to(self.device)
        
        # 加载深度
        depth_key = next((k for k in ['d', 'depth', 'depths'] if k in data), None)
        self.depths = torch.from_numpy(data[depth_key].astype(np.float32)).to(self.device)
        
        # 加载位姿
        pose_key = 'cam_c2w' if 'cam_c2w' in data else 'c2w'
        self.c2ws = torch.from_numpy(data[pose_key].astype(np.float32)).to(self.device)
        
        # [Critical] 坐标系翻转: OpenCV -> OpenGL
        self.c2ws[..., 0:3, 1:3] *= -1
        
        self.num_frames, self.H, self.W, _ = self.images.shape
        print(f"Data Loaded: {self.num_frames} frames, {self.W}x{self.H}")

        if 'K' in data:
            self.K = torch.from_numpy(data['K']).to(self.device)
            self.focal = self.K[0, 0].item()
        elif 'intrinsic' in data:
            self.K = torch.from_numpy(data['intrinsic']).to(self.device)
            self.focal = self.K[0, 0].item()
        else:
            self.focal = float(self.W) * focal_ratio
            self.K = torch.tensor([[self.focal, 0, self.W / 2.0], [0, self.focal, self.H / 2.0], [0, 0, 1]], device=self.device)
            print(f"Warning: Using guessed focal length {self.focal:.2f}")

        # 从 SAM2 视频加载 Mask
        self.gt_masks, self.core_masks = self._load_masks_from_video(sam2_video_path)
        
        if self.gt_masks is None:
            print("Error: Masks failed to load from video.")
            exit()

        self.means_prev_all = None
        self.means_prev_2 = None
        self.knn_rigid_indices = None
        self.knn_rigid_weights = None

        self._init_spheres()

    def _load_masks_from_video(self, video_path):
        print(f"Loading masks from video: {video_path}")
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return None, None
            
        cap = cv2.VideoCapture(video_path)
        full_masks = []
        core_masks = []
        kernel = np.ones((15, 15), np.uint8) 
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if count >= self.num_frames: break 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if (gray.shape[1] != self.W) or (gray.shape[0] != self.H):
                gray = cv2.resize(gray, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            
            mask_np = (gray > 127).astype(np.uint8)
            core_np = cv2.erode(mask_np, kernel, iterations=1)
            
            full_masks.append(torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(-1))
            core_masks.append(torch.from_numpy(core_np.astype(np.float32)).unsqueeze(-1))
            count += 1
            
        cap.release()
        
        if len(full_masks) < self.num_frames:
            print(f"Warning: Video frames ({len(full_masks)}) < NPZ frames ({self.num_frames}). Padding.")
            last_full = full_masks[-1]
            last_core = core_masks[-1]
            for _ in range(self.num_frames - len(full_masks)):
                full_masks.append(last_full)
                core_masks.append(last_core)

        print(f"Loaded {len(full_masks)} masks from video.")
        return torch.stack(full_masks).to(self.device), torch.stack(core_masks).to(self.device)

    def _align_shape(self, tensor):
        if tensor.shape[0] == self.W and tensor.shape[1] == self.H: return tensor.permute(1, 0, 2)
        return tensor

    def _init_spheres(self):
        print("Initializing Gaussian Ellipsoids (Anisotropic)...")
        idx = 0
        depth = self.depths[idx]
        mask = self.gt_masks[idx, ..., 0] > 0.5
        
        ys, xs = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing='ij')
        ys, xs = ys.to(self.device), xs.to(self.device)
        
        valid = (depth > 0.01) & (depth < 100.0) & mask
        
        ys, xs, d = ys[valid], xs[valid], depth[valid]
        x_c = (xs - self.W/2.0) * d / self.focal
        y_c = (ys - self.H/2.0) * d / self.focal
        xyz_cam = torch.stack([x_c, y_c, d], dim=-1)
        
        c2w = self.c2ws[idx]
        self.means = (c2w[:3, :3] @ xyz_cam.T).T + c2w[:3, 3]
        
        N = self.means.shape[0]
        self.radii = torch.ones((N, 3), device=self.device) * -5.0 
        self.quats = torch.rand((N, 4), device=self.device); self.quats[:, 0] = 1.0
        
        rgb_vals = self.images[idx][valid]
        self.rgbs = torch.log(torch.clamp(rgb_vals, 0.01, 0.99) / (1 - rgb_vals))
        self.opacities = torch.ones((N), device=self.device) * 4.0 
        
        self.means.requires_grad_(True)
        self.radii.requires_grad_(True)
        self.quats.requires_grad_(True)
        self.rgbs.requires_grad_(True)
        self.opacities.requires_grad_(True)
        
        self.grad_accum = torch.zeros(N, device=self.device)
        self.denom = torch.zeros(N, device=self.device)
        print(f"[*] Initialized {N} Gaussians (Ellipsoids).")

    def _compute_knn(self, points, k=20):
        indices = []
        dists = []
        chunk_size = 2000
        N = points.shape[0]
        with torch.no_grad():
            for i in range(0, N, chunk_size):
                end = min(i + chunk_size, N)
                d = torch.cdist(points[i:end], points) 
                val, idx = d.topk(k, dim=1, largest=False)
                indices.append(idx)
                dists.append(val)
        knn_indices = torch.cat(indices, dim=0)
        dist_sq = torch.cat(dists, dim=0) ** 2
        knn_weights = torch.exp(-2000.0 * dist_sq)
        return knn_indices, knn_weights

    def _densify_and_prune(self):
        with torch.no_grad():
            grads = self.grad_accum / self.denom
            grads[self.denom == 0] = 0.0
            
            to_clone = grads >= 0.0004
            
            if to_clone.sum() > 0:
                self.means = torch.cat([self.means, self.means[to_clone]]).detach().requires_grad_(True)
                self.radii = torch.cat([self.radii, self.radii[to_clone]]).detach().requires_grad_(True)
                self.rgbs = torch.cat([self.rgbs, self.rgbs[to_clone]]).detach().requires_grad_(True)
                self.quats = torch.cat([self.quats, self.quats[to_clone]]).detach().requires_grad_(True)
                self.opacities = torch.cat([self.opacities, self.opacities[to_clone]]).detach().requires_grad_(True)
            
            self.grad_accum = torch.zeros(self.means.shape[0], device=self.device)
            self.denom = torch.zeros(self.means.shape[0], device=self.device)
            return to_clone.sum() > 0

    # [NEW] 保存高斯点参数的功能
    def save_params(self, frame_idx):
        save_path = os.path.join(self.dir_params, f"gaussians_{frame_idx:03d}.pt")
        torch.save({
            'means': self.means.detach().cpu(),
            'quats': self.quats.detach().cpu(),
            'radii': self.radii.detach().cpu(),
            'rgbs': self.rgbs.detach().cpu(),
            'opacities': self.opacities.detach().cpu(),
            'num_points': self.means.shape[0]
        }, save_path)

    def train_frame(self, frame_idx, iterations):
        allow_densify = True if frame_idx == 0 else (frame_idx % 5 == 0)

        optimizer = optim.Adam([
            {'params': [self.means], 'lr': 0.00016},
            {'params': [self.quats], 'lr': 0.001},
            {'params': [self.rgbs], 'lr': 0.0025},
            {'params': [self.radii], 'lr': 0.005},
            {'params': [self.opacities], 'lr': 0.05}
        ], lr=0.001)

        l1_loss = nn.L1Loss()
        
        gt_mask = self.gt_masks[frame_idx]
        core_mask = self.core_masks[frame_idx]
        
        gt_img_raw = self.images[frame_idx]
        gt_img = gt_img_raw * gt_mask + (1.0 - gt_mask) * 1.0 
        
        gt_d = self.depths[frame_idx].unsqueeze(-1)
        viewmat = torch.inverse(self.c2ws[frame_idx])
        bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        # Depth Loss 权重
        lambda_depth = 0.2

        pbar = tqdm(range(iterations), desc=f"Frame {frame_idx}", leave=False)
        
        render_d_final = None # 用于最后返回

        for i in pbar:
            scales = self.radii 
            colors_precomp = torch.cat([torch.sigmoid(self.rgbs), torch.ones_like(self.rgbs[:, :1])], dim=1)

            # --- 1. RGB Rendering ---
            meta = rasterization(
                self.means, F.normalize(self.quats, dim=-1), 
                torch.exp(scales), 
                torch.sigmoid(self.opacities), 
                colors_precomp, 
                viewmat[None], self.K[None], self.W, self.H, 
                packed=False
            )
            render_rgba = self._align_shape(meta[0][0])
            render_rgb_fg = render_rgba[..., :3]
            render_alpha = render_rgba[..., 3:4]
            render_rgb = render_rgb_fg + bg_color * (1.0 - render_alpha)

            # Losses
            loss_rgb = 0.8 * l1_loss(render_rgb, gt_img) + 0.2 * (1.0 - ssim(render_rgb.permute(2,0,1).unsqueeze(0), gt_img.permute(2,0,1).unsqueeze(0)))
            loss = loss_rgb

            if core_mask.sum() > 0:
                loss += 2.0 * l1_loss(render_alpha * core_mask, core_mask)

            bg_mask = 1.0 - gt_mask
            if bg_mask.sum() > 0:
                loss += 5.0 * l1_loss(render_alpha * bg_mask, torch.zeros_like(bg_mask))
            
            # --- 2. Depth Rendering & Loss ---
            # 只有当 Mask 存在内容时才计算深度 Loss
            if gt_mask.sum() > 0:
                # 将高斯中心变换到相机坐标系
                means_cam = self.means @ viewmat[:3, :3].T + viewmat[:3, 3]
                # 提取 Z 轴作为 "颜色" 传入光栅化器
                d_cols = means_cam[:, 2:3].expand(-1, 3)
                
                meta_d = rasterization(
                    self.means, F.normalize(self.quats, dim=-1), 
                    torch.exp(scales), 
                    torch.sigmoid(self.opacities), 
                    d_cols, # 这里传入的是深度值
                    viewmat[None], self.K[None], self.W, self.H, packed=False
                )
                render_d = self._align_shape(meta_d[0][0][..., 0:1])
                
                # [Updated] Depth Loss - 使用 GT Mask 过滤
                loss += lambda_depth * l1_loss(render_d * gt_mask, gt_d * gt_mask)
                
                # 记录最后一次迭代的深度图用于可视化
                if i == iterations - 1:
                    render_d_final = render_d.detach()

            # --- 3. Rigidity Loss ---
            if frame_idx > 0 and self.knn_rigid_indices is not None:
                n_prev = self.means_prev_all.shape[0]
                curr_rigid = self.means[:n_prev]
                neighbors_curr = curr_rigid[self.knn_rigid_indices]
                curr_exp = curr_rigid.unsqueeze(1).expand(-1, 20, -1)
                dist_curr = torch.norm(neighbors_curr - curr_exp, dim=-1)
                neighbors_prev = self.means_prev_all[self.knn_rigid_indices]
                prev_exp = self.means_prev_all.unsqueeze(1).expand(-1, 20, -1)
                dist_prev = torch.norm(neighbors_prev - prev_exp, dim=-1)
                loss += 10.0 * (self.knn_rigid_weights * torch.abs(dist_curr - dist_prev)).mean()

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            if allow_densify and i > 100 and i < iterations - 100 and i % 300 == 0:
                if self.means.grad is not None:
                    self.grad_accum += torch.norm(self.means.grad[:, :2], dim=-1)
                    self.denom += 1.0
                added = self._densify_and_prune()
                if added:
                    optimizer = optim.Adam([
                        {'params': [self.means], 'lr': 0.00016},
                        {'params': [self.quats], 'lr': 0.001},
                        {'params': [self.rgbs], 'lr': 0.0025},
                        {'params': [self.radii], 'lr': 0.005},
                        {'params': [self.opacities], 'lr': 0.05}
                    ], lr=0.001)

        # 准备返回数据
        rgb_np = (render_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        
        depth_np = None
        if render_d_final is not None:
            # 简单的深度可视化: 假设深度范围在 0-5m 之间 (可根据数据调整)
            d_vis = render_d_final.cpu().numpy()
            d_vis = (d_vis / 5.0 * 255).clip(0, 255).astype(np.uint8)
            depth_np = np.repeat(d_vis, 3, axis=-1)
        else:
            depth_np = np.zeros_like(rgb_np)

        return rgb_np, depth_np

    def run(self):
        frames = []
        depth_frames = []
        
        print(f"=== Starting Ellipsoidal Reconstruction ===")
        print(f"=== Saving to: {self.output_dir} ===")
        
        for t in range(self.num_frames):
            iters = 3000 if t == 0 else 3000
            
            if t > 0:
                self.knn_rigid_indices, self.knn_rigid_weights = self._compute_knn(self.means_prev_all, k=20)
                if t > 1:
                    with torch.no_grad():
                        n_curr = self.means.shape[0]
                        n_prev_2 = self.means_prev_2.shape[0]
                        min_n = min(n_curr, n_prev_2)
                        velocity = self.means_prev_all[:min_n] - self.means_prev_2[:min_n]
                        self.means.data[:min_n] += velocity

            # [Modified] 接收 RGB 和 Depth
            rgb_np, depth_np = self.train_frame(t, iters)
            
            # [NEW] 保存高斯参数
            self.save_params(t)
            
            # 保存图像
            img_pil = Image.fromarray(rgb_np)
            depth_pil = Image.fromarray(depth_np)
            
            frames.append(img_pil)
            depth_frames.append(depth_pil)
            
            self.means_prev_all = self.means.detach().clone()
            if t > 0: self.means_prev_2 = self.means_prev_all.clone()
            else: self.means_prev_2 = self.means.detach().clone()

            # 保存单帧图片
            img_pil.save(f"{self.dir_images}/frame_{t:03d}.png")
            depth_pil.save(f"{self.dir_depths}/depth_{t:03d}.png")
            
        # 保存 GIF
        frames[0].save(f"{self.output_dir}/result_rgb.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
        depth_frames[0].save(f"{self.output_dir}/result_depth.gif", save_all=True, append_images=depth_frames[1:], duration=40, loop=0)
        
        print(f"Done! Results saved to {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Splatting Solver")
    
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/learn-genmojo/data/car-turn/MegaSAM_Outputs/car-turn_sgd_cvd_hr.npz", help="Path to the npz data file")
    parser.add_argument("--video_path", type=str, default="sam2.mp4", help="Path to the SAM2 video mask file")
    parser.add_argument("--exp_name", type=str, default="default_exp", help="Name of the experiment")
    parser.add_argument("--output_root", type=str, default="results", help="Root directory for results")
    parser.add_argument("--focal_ratio", type=float, default=0.8, help="Focal length estimation ratio")

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    folder_name = f"{current_time}_{args.exp_name}"
    full_output_dir = os.path.join(args.output_root, folder_name)

    solver = GSSolver(
        data_path=args.data_path, 
        sam2_video_path=args.video_path, 
        output_dir=full_output_dir,
        focal_ratio=args.focal_ratio
    )
    solver.run()