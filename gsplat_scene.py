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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import (
    DDPMScheduler, 
    UNet2DConditionModel, 
    AutoencoderKL
)
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os

# --- 修复 1: 输入维度改为 772 (768图像特征 + 4姿态) ---
class CLIPCameraProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(772, 768) 
        self.norm = nn.Identity()

    def forward(self, x):
        return self.proj(x)

class Zero123Guide(nn.Module):
    def __init__(self, device, model_key="ashawkey/zero123-xl-diffusers"):
        super().__init__()
        self.device = device
        self.dtype = torch.float16
        
        print(f"[SDS] Initializing Zero123 Guidance...")
        print(f"[SDS] Loading components from: {model_key}...")

        try:
            # 1. Load VAE
            self.vae = AutoencoderKL.from_pretrained(
                model_key, subfolder="vae", torch_dtype=self.dtype
            ).to(device)
            
            # 2. Load UNet
            self.unet = UNet2DConditionModel.from_pretrained(
                model_key, subfolder="unet", torch_dtype=self.dtype
            ).to(device)
            
            # 3. Load Image Encoder
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_key, subfolder="image_encoder", torch_dtype=self.dtype
            ).to(device)
            
            # 4. Load Scheduler
            self.scheduler = DDPMScheduler.from_pretrained(
                model_key, subfolder="scheduler"
            )
            
            # 5. Load Camera Projection
            self.cc_projection = CLIPCameraProjection().to(device, dtype=self.dtype)
            
            print("[SDS] Downloading camera projection weights (safetensors)...")
            
            cc_path = hf_hub_download(
                repo_id=model_key, 
                filename="diffusion_pytorch_model.safetensors", 
                subfolder="clip_camera_projection"
            )
            
            state_dict = load_file(cc_path)
            
            # 映射权重 Key
            new_state_dict = {}
            for k, v in state_dict.items():
                if "proj" in k or "linear" in k:
                    new_state_dict["proj.weight"] = v if "weight" in k else new_state_dict.get("proj.weight")
                    new_state_dict["proj.bias"] = v if "bias" in k else new_state_dict.get("proj.bias")
                else:
                    new_state_dict[k] = v
            
            if new_state_dict.get("proj.weight") is not None:
                self.cc_projection.load_state_dict(new_state_dict, strict=True) # 这里现在可以开启 strict=True
                print("[SDS] Camera projection loaded successfully.")
            else:
                raise RuntimeError("Weight mismatch in camera projection.")

        except Exception as e:
            print(f"[Error] Failed to load components: {e}")
            raise RuntimeError("Could not load Zero123 components.")

        # Cleanup
        import gc; gc.collect(); torch.cuda.empty_cache()

        self.min_step = 0.02
        self.max_step = 0.98
        self.ref_embeddings = None
        self.c2w_ref = None

    # --- 修复 2: 拼接图像特征和姿态 ---
    def get_cam_embeddings(self, elevation, azimuth, radius):
        # 姿态: [B, 4]
        zero_tensor = torch.zeros_like(radius)
        camera_pose = torch.stack([elevation, azimuth, radius, zero_tensor], dim=-1).to(self.device, dtype=self.dtype)
        
        # 图像特征: [1, 768] -> 扩展到 [B, 768]
        if self.ref_embeddings is None:
             raise RuntimeError("Reference embeddings not initialized. Call prepare_condition first.")
        
        # 注意: self.ref_embeddings 形状通常是 [1, 768] (来自 image_embeds)
        # 我们需要将其重复以匹配 camera_pose 的 batch size
        batch_size = camera_pose.shape[0]
        ref_emb_expanded = self.ref_embeddings.repeat(batch_size, 1)
        
        # 拼接: [B, 768] + [B, 4] -> [B, 772]
        mlp_input = torch.cat([ref_emb_expanded, camera_pose], dim=-1)
        
        return self.cc_projection(mlp_input)

    @torch.no_grad()
    def prepare_condition(self, ref_image_tensor, c2w_ref):
        # --- 1. CLIP 分支 (需要 224x224, CLIP 归一化) ---
        ref_img_224 = F.interpolate(ref_image_tensor, (224, 224), mode='bilinear', align_corners=False)
        ref_img_norm_clip = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(ref_img_224)
        self.ref_embeddings = self.image_encoder(ref_img_norm_clip).image_embeds.to(dtype=self.dtype)

        # --- 2. VAE 分支 (需要 256x256, [-1, 1] 归一化) ---
        # 必须把参考图也编码成 latent，用于后续和噪声拼接
        ref_img_256 = F.interpolate(ref_image_tensor, (256, 256), mode='bilinear', align_corners=False)
        ref_img_norm_vae = (ref_img_256 - 0.5) * 2
        ref_img_norm_vae = ref_img_norm_vae.to(dtype=self.dtype)
        
        # 编码并存储，注意乘上缩放系数 0.18215
        self.ref_latents = self.vae.encode(ref_img_norm_vae).latent_dist.mode() * 0.18215
        
        self.c2w_ref = c2w_ref

    def compute_relative_pose(self, c2w_curr):
        T_ref = self.c2w_ref[:3, 3]
        T_curr = c2w_curr[:3, 3]

        def cartesian_to_spherical(xyz):
            xy = torch.sqrt(xyz[0]**2 + xyz[2]**2)
            elevation = torch.atan2(xyz[1], xy)
            azimuth = torch.atan2(xyz[0], xyz[2])
            radius = torch.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
            return elevation, azimuth, radius

        el_ref, az_ref, r_ref = cartesian_to_spherical(T_ref)
        el_curr, az_curr, r_curr = cartesian_to_spherical(T_curr)
        
        d_azimuth = torch.rad2deg(az_curr - az_ref)
        d_elevation = torch.rad2deg(el_curr - el_ref)
        d_radius = torch.tensor(0.0, device=self.device)
        
        d_azimuth = (d_azimuth + 180) % 360 - 180
        
        return d_elevation.unsqueeze(0), d_azimuth.unsqueeze(0), d_radius.unsqueeze(0)

    def sds_loss(self, pred_rgb, relative_pose):
        # 1. 准备渲染图 Latents
        pred_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        
        # 将 Float32 的渲染图转为 Float16 (Half) 给 VAE
        vae_input = (pred_256 - 0.5) * 2
        vae_input = vae_input.to(dtype=self.dtype) 
        
        # VAE 编码 (全程 Half)
        latents = self.vae.encode(vae_input).latent_dist.sample()
        latents = latents * 0.18215
        
        # 2. 加噪
        noise = torch.randn_like(latents)
        t = torch.randint(
            int(self.min_step * self.scheduler.config.num_train_timesteps), 
            int(self.max_step * self.scheduler.config.num_train_timesteps), 
            [1], device=self.device
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # 3. 拼接 Latents
        latent_model_input = torch.cat([noisy_latents, self.ref_latents], dim=1)
        
        # 4. 获取相机条件
        d_el, d_az, d_r = relative_pose
        camera_embeddings = self.get_cam_embeddings(d_el, d_az, d_r)
        
        # 5. UNet 预测 (Half)
        encoder_hidden_states = self.ref_embeddings.unsqueeze(1)
        noise_pred = self.unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=encoder_hidden_states,
            class_labels=camera_embeddings
        ).sample

        # 6. 计算梯度 (Half)
        w = 1 - (t / self.scheduler.config.num_train_timesteps)
        grad = w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        # [核心修复]: 在这里将 Half 转为 Float32 进行 Loss 计算
        # 这样 Backward 时，梯度会以 Float32 传到这里，然后安全转回 Half 给 VAE
        latents_float = latents.float()
        grad_float = grad.float()
        
        target = (latents_float - grad_float).detach()
        loss = 0.5 * F.mse_loss(latents_float, target, reduction="sum")
        
        return loss

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

def crop_and_resize_differentiable(img_tensor, mask_tensor, target_size=256, padding=0.1):
    """
    img_tensor: [1, 3, H, W]
    mask_tensor: [1, 1, H, W] (用于计算 BBox)
    """
    # 1. 从 Mask 计算 BBox (这里为了简单，假设 batch_size=1)
    # 注意：为了梯度反向传播，BBox 的坐标获取最好不要打断计算图，
    # 但通常我们用 GT mask 或当前渲染的 detach mask 来定位置是没问题的。
    
    nonzero = torch.nonzero(mask_tensor[0, 0] > 0.5)
    if nonzero.shape[0] == 0:
        return F.interpolate(img_tensor, (target_size, target_size), mode='bilinear')

    y_min, x_min = torch.min(nonzero, dim=0)[0]
    y_max, x_max = torch.max(nonzero, dim=0)[0]
    
    # 2. 变成正方形 BBox
    center_y = (y_min + y_max) / 2.0
    center_x = (x_min + x_max) / 2.0
    height = y_max - y_min
    width = x_max - x_min
    side_length = max(height, width) * (1 + padding) # 加上一点 padding
    
    # 3. 构建 Grid 进行采样 (Crop)
    # 我们需要构建一个变换矩阵，把 BBox 区域映射到 [-1, 1]
    # Grid Sample 需要 normalized coordinates
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    
    # 计算 scale
    s_x = side_length / W
    s_y = side_length / H
    
    # 计算 translation (归一化坐标系下，中心是 0,0)
    # 图像坐标 (cx, cy) -> 归一化坐标 (-1 + 2*cx/W, -1 + 2*cy/H)
    t_x = -1 + 2 * center_x / W
    t_y = -1 + 2 * center_y / H
    
    # 构建仿射矩阵 [B, 2, 3]
    theta = torch.tensor([[
        [s_x, 0,   t_x],
        [0,   s_y, t_y]
    ]], device=img_tensor.device, dtype=img_tensor.dtype)
    
    grid = F.affine_grid(theta, torch.Size([1, 3, target_size, target_size]), align_corners=False)
    cropped_img = F.grid_sample(img_tensor, grid, align_corners=False)
    
    return cropped_img
    

def get_orbit_camera(c2w, angle_x, angle_y, center=None, device='cuda'):
    """
    基于当前c2w，沿水平(azimuth)和垂直(elevation)方向旋转生成新视角
    angle_x, angle_y: 角度（度数）
    """
    if center is None:
        center = torch.zeros(3, device=device)
    
    # 提取旋转和平移
    rot = c2w[:3, :3]
    pos = c2w[:3, 3]
    
    # 计算当前相机相对于中心的半径向量
    dir_vec = pos - center
    radius = torch.norm(dir_vec)
    
    # 构建转换矩阵：先移回原点 -> 旋转 -> 移回原来的半径距离
    # 这里简化处理：直接假设物体在原点附近，使用LookAt逻辑重新构建
    
    # 1. 将笛卡尔坐标转为球坐标 (r, theta, phi)
    x, y, z = dir_vec[0], dir_vec[1], dir_vec[2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(y / r) # 极角 (与Y轴夹角, OpenGL坐标系Y通常为上/下)
    phi = torch.atan2(z, x)   # 方位角
    
    # 2. 施加偏移 (注意弧度转换)
    theta_new = theta + torch.deg2rad(torch.tensor(angle_y, device=device))
    phi_new = phi + torch.deg2rad(torch.tensor(angle_x, device=device))
    
    # 限制 theta 防止万向节死锁
    theta_new = torch.clamp(theta_new, 0.1, 3.1)
    
    # 3. 转回笛卡尔坐标 (新位置)
    x_new = r * torch.sin(theta_new) * torch.cos(phi_new)
    y_new = r * torch.cos(theta_new)
    z_new = r * torch.sin(theta_new) * torch.sin(phi_new)
    pos_new = torch.stack([x_new, y_new, z_new]) + center
    
    # 4. 构建 LookAt 矩阵 (新位置看向中心)
    forward = F.normalize(center - pos_new, dim=0)
    up = torch.tensor([0.0, 1.0, 0.0], device=device) # 假设Y轴向上
    
    # 如果forward和up平行，重新选择up
    if torch.abs(torch.dot(forward, up)) > 0.99:
        up = torch.tensor([0.0, 0.0, 1.0], device=device)
        
    right = F.normalize(torch.cross(forward, up), dim=0)
    new_up = F.normalize(torch.cross(right, forward), dim=0)
    
    # 构建 c2w
    new_c2w = torch.eye(4, device=device)
    new_c2w[:3, 0] = right
    new_c2w[:3, 1] = new_up # 注意: 有些系统是 -new_up，Zero123通常适应 OpenGL
    new_c2w[:3, 2] = -forward # OpenGL 相机看向 -Z
    new_c2w[:3, 3] = pos_new
    
    return new_c2w

class GSSolver:
    def __init__(self, data_path: str, sam2_video_path: str, output_dir: str, focal_ratio: float = 0.8, use_sds: bool = True):
        self.device = torch.device("cuda:0")
        self.output_dir = output_dir
        self.use_sds = use_sds # [NEW] Switch
        
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

        # [NEW] SDS Initialization
        if self.use_sds:
            print("[SDS] Initializing Zero123 Guidance...")
            self.guidance = Zero123Guide(self.device)
            # Encode Reference Image (Frame 0)
            # Permute to [1, 3, H, W]
            ref_img = self.images[0].permute(2, 0, 1).unsqueeze(0)
            ref_mask = self.gt_masks[0].permute(2, 0, 1).unsqueeze(0)
            # Apply Mask to Ref Image (Black background usually better for Zero123)
            ref_img_masked = ref_img * ref_mask
            self.guidance.prepare_condition(ref_img_masked, self.c2ws[0])
            print("[SDS] Reference image encoded.")

        self._init_spheres()

    def _load_masks_from_video(self, video_path):
        # ... (Keep existing mask loading code) ...
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
        # ... (Keep existing initialization code) ...
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
        # ... (Keep existing KNN code) ...
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
        # ... (Keep existing densify code) ...
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
        
        # 获取当前帧的GT位姿
        c2w_curr = self.c2ws[frame_idx]
        viewmat_gt = torch.inverse(c2w_curr)
        
        bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        lambda_depth = 0.2
        
        # [Config] SDS 配置
        lambda_sds_base = 0.05  # 基础权重
        max_angle_deg = 45.0    # 最大采样偏转角度
        sds_interval = 10       # 每多少次迭代做一次新视角SDS
        
        pbar = tqdm(range(iterations), desc=f"Frame {frame_idx}", leave=False)
        render_d_final = None 

        for i in pbar:
            optimizer.zero_grad() # 显式清零
            
            scales = self.radii 
            colors_precomp = torch.cat([torch.sigmoid(self.rgbs), torch.ones_like(self.rgbs[:, :1])], dim=1)

            # ==============================
            # 1. 真实视角渲染 (GT View)
            # ==============================
            meta = rasterization(
                self.means, F.normalize(self.quats, dim=-1), 
                torch.exp(scales), 
                torch.sigmoid(self.opacities), 
                colors_precomp, 
                viewmat_gt[None], self.K[None], self.W, self.H, 
                packed=False
            )
            render_rgba = self._align_shape(meta[0][0])
            render_rgb_fg = render_rgba[..., :3]
            render_alpha = render_rgba[..., 3:4]
            render_rgb = render_rgb_fg + bg_color * (1.0 - render_alpha)

            # --- GT View Losses (L1 + SSIM + Mask) ---
            loss_rgb = 0.8 * l1_loss(render_rgb, gt_img) + 0.2 * (1.0 - ssim(render_rgb.permute(2,0,1).unsqueeze(0), gt_img.permute(2,0,1).unsqueeze(0)))
            loss = loss_rgb

            if core_mask.sum() > 0:
                loss += 2.0 * l1_loss(render_alpha * core_mask, core_mask)

            bg_mask = 1.0 - gt_mask
            if bg_mask.sum() > 0:
                loss += 5.0 * l1_loss(render_alpha * bg_mask, torch.zeros_like(bg_mask))
            
            # --- Depth Loss ---
            if gt_mask.sum() > 0:
                means_cam = self.means @ viewmat_gt[:3, :3].T + viewmat_gt[:3, 3]
                d_cols = means_cam[:, 2:3].expand(-1, 3)
                meta_d = rasterization(
                    self.means, F.normalize(self.quats, dim=-1), 
                    torch.exp(scales), 
                    torch.sigmoid(self.opacities), 
                    d_cols, 
                    viewmat_gt[None], self.K[None], self.W, self.H, packed=False
                )
                render_d = self._align_shape(meta_d[0][0][..., 0:1])
                loss += lambda_depth * l1_loss(render_d * gt_mask, gt_d * gt_mask)
                
                if i == iterations - 1:
                    render_d_final = render_d.detach()

            # --- Rigidity Loss (保持不变) ---
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

            # ===============================================
            # 2. 新视角 SDS + 正则化 (Novel View Regularization)
            # ===============================================
            # 逻辑：不在GT视角做SDS。随机采样一个角度，角度越大权重越高。
            
            if self.use_sds and i % sds_interval == 0:
                # A. 随机采样偏移角度 (环绕采样)
                # 避开非常小的角度，因为那和GT太像了
                min_deg = 5.0
                delta_azimuth = (torch.rand(1).item() * 2 - 1) * max_angle_deg # [-max, max]
                delta_elevation = (torch.rand(1).item() * 2 - 1) * (max_angle_deg / 2.0) # [-max/2, max/2]
                
                # 确保偏离足够大，否则不做SDS
                total_angle_diff = np.sqrt(delta_azimuth**2 + delta_elevation**2)
                
                if total_angle_diff > min_deg:
                    # B. 生成新相机位姿
                    c2w_novel = get_orbit_camera(c2w_curr, delta_azimuth, delta_elevation, device=self.device)
                    viewmat_novel = torch.inverse(c2w_novel)
                    
                    # C. 渲染新视角
                    meta_novel = rasterization(
                        self.means, F.normalize(self.quats, dim=-1), 
                        torch.exp(scales), 
                        torch.sigmoid(self.opacities), 
                        colors_precomp, 
                        viewmat_novel[None], self.K[None], self.W, self.H, 
                        packed=False
                    )
                    rgba_novel = self._align_shape(meta_novel[0][0])
                    rgb_novel = rgba_novel[..., :3] + bg_color * (1.0 - rgba_novel[..., 3:4])
                    
                    # D. 准备 SDS 输入
                    pred_img = rgb_novel.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
                    pred_alpha = rgba_novel[..., 3:4].permute(2, 0, 1).unsqueeze(0)
    
                    # 对预测图进行裁剪，使其物体居中
                    pred_img_centered = crop_and_resize_differentiable(pred_img, pred_alpha)
                    # E. 计算动态权重
                    # 距离GT越远，越依赖SDS。公式: weight = base * (diff / max_diff)
                    # 加上一个基础bias确保有梯度
                    dynamic_weight = lambda_sds_base * (0.5 + 1.5 * (total_angle_diff / max_angle_deg))
                    
                    # F. 计算 SDS Loss
                    # 注意: compute_relative_pose 应该计算 (Novel - Ref_Image_Frame0) 的关系
                    # 这里把 c2w_novel 传进去即可
                    rel_pose = self.guidance.compute_relative_pose(c2w_novel)
                    loss_sds_val = self.guidance.sds_loss(pred_img_centered, rel_pose)
                    
                    loss += dynamic_weight * loss_sds_val
                    
                    # ===============================================
                    # 3. 新视角图像质量正则化 (Opacity Sparsity)
                    # ===============================================
                    # 在新视角下，抑制半透明的“云雾”伪影。
                    # 迫使 Opacity 趋向于 0 或 1
                    opacity_novel = rgba_novel[..., 3]
                    # loss_sparsity = lambda * (o * log(o) + (1-o) * log(1-o)) 的简化版 -> mean(o * (1-o))
                    # 或者简单的 L1 正则化鼓励稀疏
                    loss_sparsity = opacity_novel.mean() * 0.5 + (opacity_novel * (1.0 - opacity_novel)).mean() * 0.5
                    
                    loss += 0.01 * loss_sparsity * dynamic_weight # 权重随视角偏移增加

            loss.backward()
            optimizer.step()
            
            # --- Densify Logic ---
            if allow_densify and i > 100 and i < iterations - 100 and i % 300 == 0:
                if self.means.grad is not None:
                    self.grad_accum += torch.norm(self.means.grad[:, :2], dim=-1)
                    self.denom += 1.0
                added = self._densify_and_prune()
                if added:
                    # Re-init optimizer params (keep logic simple)
                    optimizer = optim.Adam([
                        {'params': [self.means], 'lr': 0.00016},
                        {'params': [self.quats], 'lr': 0.001},
                        {'params': [self.rgbs], 'lr': 0.0025},
                        {'params': [self.radii], 'lr': 0.005},
                        {'params': [self.opacities], 'lr': 0.05}
                    ], lr=0.001)

        # 结束处理 (渲染最终图用于可视化)
        rgb_np = (render_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        
        depth_np = None
        if render_d_final is not None:
            d_vis = render_d_final.cpu().numpy()
            d_vis = (d_vis / 5.0 * 255).clip(0, 255).astype(np.uint8)
            depth_np = np.repeat(d_vis, 3, axis=-1)
        else:
            depth_np = np.zeros_like(rgb_np)

        return rgb_np, depth_np

    def run(self):
        # ... (Same as original) ...
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

            rgb_np, depth_np = self.train_frame(t, iters)
            
            self.save_params(t)
            
            img_pil = Image.fromarray(rgb_np)
            depth_pil = Image.fromarray(depth_np)
            
            frames.append(img_pil)
            depth_frames.append(depth_pil)
            
            self.means_prev_all = self.means.detach().clone()
            if t > 0: self.means_prev_2 = self.means_prev_all.clone()
            else: self.means_prev_2 = self.means.detach().clone()

            img_pil.save(f"{self.dir_images}/frame_{t:03d}.png")
            depth_pil.save(f"{self.dir_depths}/depth_{t:03d}.png")
            
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
    # [NEW] Argument
    parser.add_argument("--use_sds", action="store_true", help="Enable Zero123 SDS Guidance")

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    folder_name = f"{current_time}_{args.exp_name}"
    full_output_dir = os.path.join(args.output_root, folder_name)

    solver = GSSolver(
        data_path=args.data_path, 
        sam2_video_path=args.video_path, 
        output_dir=full_output_dir,
        focal_ratio=args.focal_ratio,
        use_sds=args.use_sds 
    )
    solver.run()