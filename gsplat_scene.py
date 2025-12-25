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
from torchvision.utils import save_image
import trimesh
import torchvision.transforms as T
from diffusers import (
    DDPMScheduler, 
    UNet2DConditionModel, 
    AutoencoderKL
)
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ==========================================
# 1. 辅助工具函数
# ==========================================

def crop_image_by_mask(image_tensor, mask_tensor, target_size=256, padding=0.1):
    """
    [核心修复] 非微分裁剪，仅用于处理参考图 Condition。
    确保 Zero123 看到的物体是居中且放大的。
    image_tensor: [C, H, W]
    mask_tensor: [1, H, W]
    """
    # 找 BBox
    nonzero = torch.nonzero(mask_tensor[0] > 0.5)
    if nonzero.shape[0] == 0:
        # 如果 Mask 为空，直接缩放原图（兜底）
        return F.interpolate(image_tensor.unsqueeze(0), (target_size, target_size), mode='bilinear').squeeze(0)
    
    y_min, x_min = torch.min(nonzero, dim=0)[0]
    y_max, x_max = torch.max(nonzero, dim=0)[0]
    
    center_y = (y_min + y_max) / 2.0
    center_x = (x_min + x_max) / 2.0
    height = y_max - y_min
    width = x_max - x_min
    
    # 变成正方形
    side_length = max(height, width) * (1 + padding)
    
    # 计算裁剪坐标
    top = int(max(0, center_y - side_length / 2))
    left = int(max(0, center_x - side_length / 2))
    bottom = int(min(image_tensor.shape[1], center_y + side_length / 2))
    right = int(min(image_tensor.shape[2], center_x + side_length / 2))
    
    # Crop
    crop = image_tensor[:, top:bottom, left:right]
    
    # Resize 到 256x256
    crop_resized = F.interpolate(crop.unsqueeze(0), (target_size, target_size), mode='bilinear', align_corners=False).squeeze(0)
    
    return crop_resized

def crop_and_resize_differentiable(img_tensor, mask_tensor, target_size=256, padding=0.1):
    """
    [保留] 微分裁剪，用于 SDS 过程中对渲染出的预测图进行裁剪
    img_tensor: [1, 3, H, W]
    mask_tensor: [1, 1, H, W]
    """
    nonzero = torch.nonzero(mask_tensor[0, 0] > 0.5)
    if nonzero.shape[0] == 0:
        return F.interpolate(img_tensor, (target_size, target_size), mode='bilinear')

    y_min, x_min = torch.min(nonzero, dim=0)[0]
    y_max, x_max = torch.max(nonzero, dim=0)[0]
    
    center_y = (y_min + y_max) / 2.0
    center_x = (x_min + x_max) / 2.0
    height = y_max - y_min
    width = x_max - x_min
    side_length = max(height, width) * (1 + padding) 
    
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    s_x = side_length / W
    s_y = side_length / H
    t_x = -1 + 2 * center_x / W
    t_y = -1 + 2 * center_y / H
    
    theta = torch.tensor([[
        [s_x, 0,   t_x],
        [0,   s_y, t_y]
    ]], device=img_tensor.device, dtype=img_tensor.dtype)
    
    grid = F.affine_grid(theta, torch.Size([1, 3, target_size, target_size]), align_corners=False)
    cropped_img = F.grid_sample(img_tensor, grid, align_corners=False)
    
    return cropped_img

def get_orbit_camera(c2w, angle_x, angle_y, center=None, device='cuda'):
    if center is None:
        center = torch.zeros(3, device=device)
    
    # 1. 计算当前位置和半径
    pos_curr = c2w[:3, 3]
    vec = pos_curr - center
    radius = torch.norm(vec) + 1e-7
    
    x, y, z = vec[0], vec[1], vec[2]
    
    # 2. 转球坐标 (保持不变)
    cos_theta = torch.clamp(y / radius, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    phi = torch.atan2(z, x)
    
    # 3. 应用旋转 (保持不变)
    theta_new = theta - torch.deg2rad(torch.tensor(angle_y, device=device))
    theta_new = torch.clamp(theta_new, 0.1, 3.0)
    phi_new = phi - torch.deg2rad(torch.tensor(angle_x, device=device))
    
    # 4. 转回笛卡尔坐标
    y_new = radius * torch.cos(theta_new)
    h     = radius * torch.sin(theta_new)
    x_new = h * torch.cos(phi_new)
    z_new = h * torch.sin(phi_new)
    
    pos_new = torch.stack([x_new, y_new, z_new]) + center
    
    # 5. LookAt 矩阵构建 (关键修复)
    
    # 【Z轴修复】：改回指向物体 (解决白屏问题)
    # Forward (Z-axis): Points TO Object (OpenCV convention)
    z_axis = F.normalize(center - pos_new, dim=0) 
    
    # Global Up
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    # 【X轴修复】：强制取反 (解决镜像问题)
    # 原本是 cross(world_up, z_axis)，现在加个负号
    x_axis = -F.normalize(torch.cross(world_up, z_axis), dim=0)
    
    # Y轴跟随变化
    y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=0)
    
    new_c2w = torch.eye(4, device=device)
    new_c2w[:3, 0] = x_axis
    new_c2w[:3, 1] = y_axis
    new_c2w[:3, 2] = z_axis
    new_c2w[:3, 3] = pos_new
    
    return new_c2w
    
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

# ==========================================
# 2. Zero123 模块
# ==========================================

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
            self.vae = AutoencoderKL.from_pretrained(
                model_key, subfolder="vae", torch_dtype=self.dtype
            ).to(device)
            self.unet = UNet2DConditionModel.from_pretrained(
                model_key, subfolder="unet", torch_dtype=self.dtype
            ).to(device)
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_key, subfolder="image_encoder", torch_dtype=self.dtype
            ).to(device)
            self.scheduler = DDPMScheduler.from_pretrained(
                model_key, subfolder="scheduler"
            )
            self.cc_projection = CLIPCameraProjection().to(device, dtype=self.dtype)
            
            print("[SDS] Downloading camera projection weights (safetensors)...")
            cc_path = hf_hub_download(
                repo_id=model_key, 
                filename="diffusion_pytorch_model.safetensors", 
                subfolder="clip_camera_projection"
            )
            state_dict = load_file(cc_path)
            new_state_dict = {}
            for k, v in state_dict.items():
                if "proj" in k or "linear" in k:
                    new_state_dict["proj.weight"] = v if "weight" in k else new_state_dict.get("proj.weight")
                    new_state_dict["proj.bias"] = v if "bias" in k else new_state_dict.get("proj.bias")
                else:
                    new_state_dict[k] = v
            
            if new_state_dict.get("proj.weight") is not None:
                self.cc_projection.load_state_dict(new_state_dict, strict=True)
                print("[SDS] Camera projection loaded successfully.")
            else:
                raise RuntimeError("Weight mismatch in camera projection.")

        except Exception as e:
            print(f"[Error] Failed to load components: {e}")
            raise RuntimeError("Could not load Zero123 components.")

        import gc; gc.collect(); torch.cuda.empty_cache()
        self.min_step = 0.02
        self.max_step = 0.98
        self.ref_embeddings = None
        self.c2w_ref = None

    def get_cam_embeddings(self, elevation, azimuth, radius):
        zero_tensor = torch.zeros_like(radius)
        camera_pose = torch.stack([elevation, azimuth, radius, zero_tensor], dim=-1).to(self.device, dtype=self.dtype)
        if self.ref_embeddings is None:
             raise RuntimeError("Reference embeddings not initialized.")
        batch_size = camera_pose.shape[0]
        ref_emb_expanded = self.ref_embeddings.repeat(batch_size, 1)
        mlp_input = torch.cat([ref_emb_expanded, camera_pose], dim=-1)
        return self.cc_projection(mlp_input)

    @torch.no_grad()
    def prepare_condition(self, ref_image_tensor, c2w_ref):
        """
        ref_image_tensor: [1, 3, 256, 256], Range [0, 1]
        """
        # CLIP Branch
        ref_img_224 = F.interpolate(ref_image_tensor, (224, 224), mode='bilinear', align_corners=False)
        ref_img_norm_clip = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(ref_img_224)
        self.ref_embeddings = self.image_encoder(ref_img_norm_clip).image_embeds.to(dtype=self.dtype)
        
        # [新增] 生成 Unconditional (Null) Embeddings 用于 CFG
        # 通常是用全零图或者黑图，这里直接构造全零 embedding 是最稳的做法
        self.null_embeddings = torch.zeros_like(self.ref_embeddings)

        # VAE Branch
        ref_img_256 = F.interpolate(ref_image_tensor, (256, 256), mode='bilinear', align_corners=False)
        ref_img_norm_vae = (ref_img_256 - 0.5) * 2
        ref_img_norm_vae = ref_img_norm_vae.to(dtype=self.dtype)
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


    def sds_loss(self, pred_rgb, relative_pose, guidance_scale=3.0):
        """
        guidance_scale: 通常设为 3.0 到 5.0，这决定了生成纹理的锐利程度。
        """
        pred_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        vae_input = (pred_256 - 0.5) * 2
        vae_input = vae_input.to(dtype=self.dtype) 
        
        # 1. Encode Latents
        latents = self.vae.encode(vae_input).latent_dist.sample()
        latents = latents * 0.18215
        
        # 2. Add Noise
        noise = torch.randn_like(latents)
        t = torch.randint(
            int(self.min_step * self.scheduler.config.num_train_timesteps), 
            int(self.max_step * self.scheduler.config.num_train_timesteps), 
            [1], device=self.device
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # 3. Prepare Batch for CFG (Cond + Uncond)
        # [Batch=2]: [Cond, Uncond]
        latent_model_input = torch.cat([noisy_latents] * 2)
        # Cond 使用 ref_latents, Uncond 使用 ref_latents (或者 zeros, 但 Zero123XL 主要是靠 CLIP embedding 做 cond)
        # Zero123 的 latent condition 通常保持一致
        latent_model_input = torch.cat([latent_model_input, torch.cat([self.ref_latents] * 2)], dim=1)
        
        d_el, d_az, d_r = relative_pose
        camera_embeddings = self.get_cam_embeddings(d_el, d_az, d_r)
        
        # 构建 Cond 和 Uncond 的 Embedding
        # Cond: 真实的参考图 embedding
        # Uncond: 全零 embedding
        encoder_hidden_states = torch.cat([self.ref_embeddings, self.null_embeddings], dim=0).unsqueeze(1)
        
        # Camera Pose Cond 也要复制一遍 (Uncond 通常也给同样的 pose，或者给 null pose，Zero123 主要是靠 Image Embedding)
        # 这里给同样的 Pose 是常见做法
        camera_embeddings = torch.cat([camera_embeddings] * 2, dim=0)

        # 4. Predict Noise
        noise_pred = self.unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=encoder_hidden_states,
            class_labels=camera_embeddings
        ).sample

        # 5. Apply CFG
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # 6. Compute SDS Gradients
        w = 1 - (t / self.scheduler.config.num_train_timesteps)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        latents_float = latents.float()
        grad_float = grad.float()
        target = (latents_float - grad_float).detach()
        loss = 0.5 * F.mse_loss(latents_float, target, reduction="sum")
        
        return loss

# ==========================================
# 3. Gaussian Splatting Solver
# ==========================================

class GaussianSplattingSolver:
    def __init__(self, data_path: str, sam2_video_path: str, output_dir: str, focal_ratio: float = 0.8, use_sds: bool = True):
        self.device = torch.device("cuda:0")
        self.output_dir = output_dir
        self.use_sds = use_sds 
        
        self.dir_params = os.path.join(self.output_dir, "params")
        self.dir_images = os.path.join(self.output_dir, "images")
        self.dir_depths = os.path.join(self.output_dir, "depths")
        self.dir_debug = os.path.join(self.output_dir, "debug_sds") # [NEW] Debug folder
        os.makedirs(self.dir_params, exist_ok=True)
        os.makedirs(self.dir_images, exist_ok=True)
        os.makedirs(self.dir_depths, exist_ok=True)
        os.makedirs(self.dir_debug, exist_ok=True)
        
        print(f"Output Directory set to: {self.output_dir}")
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)

        self.images = torch.from_numpy(data['images'].astype(np.float32) / 255.0).to(self.device)
        
        self.depths = torch.from_numpy(data['depths'].astype(np.float32)).to(self.device)
        
        self.c2ws = torch.from_numpy(data['cam_c2w'].astype(np.float32)).to(self.device)
        self.c2ws[..., 0:3, 1:3] *= -1 # OpenCV -> OpenGL
        
        self.num_frames, self.H, self.W, _ = self.images.shape
        print(f"Data Loaded: {self.num_frames} frames, {self.W}x{self.H}")

        if 'intrinsic' in data:
            self.K = torch.from_numpy(data['intrinsic']).to(self.device)
            self.focal = self.K[0, 0].item()
        else:
            self.focal = float(self.W) * focal_ratio
            self.K = torch.tensor([[self.focal, 0, self.W / 2.0], [0, self.focal, self.H / 2.0], [0, 0, 1]], device=self.device)
            print(f"Warning: Using guessed focal length {self.focal:.2f}")

        self.gt_masks, self.core_masks = self._load_masks_from_video(sam2_video_path)
        if self.gt_masks is None: exit()

        # means_prev_all 和 means_prev_2这两个变量记录了高斯球（Gaussians）在前几帧的位置坐标（XYZ）。
        # means_prev_all (上一帧位置): 存储 $t-1$ 帧时所有高斯球的中心点坐标。
        # eans_prev_2 (前两帧位置): 存储 $t-2$ 帧时所有高斯球的中心点坐标。
        self.means_prev_all = None
        self.means_prev_2 = None

        self.knn_rigid_indices = None
        self.knn_rigid_weights = None

        if self.use_sds:
            print("[SDS] Initializing Zero123 Guidance Model...")
            self.guidance = Zero123Guide(self.device)

        self._init_spheres()

    def _load_masks_from_video(self, video_path):
        # gt_mask (Ground Truth Mask): 
        # 来源： 直接从 SAM2 视频分割结果中提取，表示算法认为的物体完整轮廓。
        # 特征： 边界比较贴近物体边缘，包含了一些可能存在不确定性的边缘区域。
        # core_mask (Core Mask):
        # 来源： 对 gt_mask 进行 cv2.erode (腐蚀操作) 得到的。
        # 特征： 比 gt_mask 整整“瘦”了一圈（代码里用了 15x15 的卷积核）。它只保留了物体最中心的、绝对确定属于物体的部分。
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
            last_full = full_masks[-1]; last_core = core_masks[-1]
            for _ in range(self.num_frames - len(full_masks)):
                full_masks.append(last_full); core_masks.append(last_core)
        return torch.stack(full_masks).to(self.device), torch.stack(core_masks).to(self.device)

    def _align_shape(self, tensor):
        if tensor.shape[0] == self.W and tensor.shape[1] == self.H: return tensor.permute(1, 0, 2)
        return tensor

    def _init_spheres(self):
        print("Initializing Gaussian Ellipsoids (Anisotropic)...")
        idx = 0
        depth = self.depths[idx]
        mask = self.gt_masks[idx, ..., 0] > 0.5 # 获取第一帧的掩码。我们只希望在物体本身所在的位置生成高斯球，而不希望在背景（天空、地面）生成。
        
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
        return torch.cat(indices, dim=0), torch.exp(-2000.0 * torch.cat(dists, dim=0) ** 2)

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
        
        # 数据准备
        gt_mask = self.gt_masks[frame_idx]
        core_mask = self.core_masks[frame_idx]
        gt_img_raw = self.images[frame_idx]
        gt_img = gt_img_raw * gt_mask + (1.0 - gt_mask) * 1.0 
        
        # Zero123 Condition 准备
        if self.use_sds:
            # 1. 转为 [C, H, W]
            ref_img_chw = gt_img_raw.permute(2, 0, 1) 
            ref_mask_chw = gt_mask.permute(2, 0, 1)   
            
            # 2. 应用 Mask (背景置黑，更适合 Zero123)
            ref_img_masked_raw = ref_img_chw * ref_mask_chw
            
            # 3. 裁剪并缩放 (居中物体，填充画面)
            ref_img_crop = crop_image_by_mask(ref_img_masked_raw, ref_mask_chw)
            
            # 4. 传入 Condition (当前帧 Pose)
            self.guidance.prepare_condition(ref_img_crop.unsqueeze(0), self.c2ws[frame_idx])
            
            # [Debug] 保存一下这一帧的参考图，确保裁剪正确
            save_image(ref_img_crop, f"{self.dir_debug}/ref_crop_frame_{frame_idx:03d}.png")

        gt_d = self.depths[frame_idx].unsqueeze(-1)
        c2w_curr = self.c2ws[frame_idx]
        viewmat_gt = torch.inverse(c2w_curr)
        bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        lambda_depth = 0.2
        lambda_sds_base = 0.01
        sds_warmup = 1800
        max_angle_deg = 45.0
        sds_interval = 10
        
        pbar = tqdm(range(iterations), desc=f"Frame {frame_idx}", leave=False)
        render_d_final = None 

        for i in pbar:
            optimizer.zero_grad()
            scales = self.radii 
            colors_precomp = torch.cat([torch.sigmoid(self.rgbs), torch.ones_like(self.rgbs[:, :1])], dim=1)

            # ==============================
            # 1. GT View 渲染
            # ==============================
            meta = rasterization(
                self.means, F.normalize(self.quats, dim=-1), 
                torch.exp(scales), torch.sigmoid(self.opacities), 
                colors_precomp, 
                viewmat_gt[None], self.K[None], self.W, self.H, packed=False
            )
            render_rgba = self._align_shape(meta[0][0])
            render_rgb_fg = render_rgba[..., :3]
            render_alpha = render_rgba[..., 3:4]
            render_rgb = render_rgb_fg + bg_color * (1.0 - render_alpha)

            loss = 0.8 * l1_loss(render_rgb, gt_img) + 0.2 * (1.0 - ssim(render_rgb.permute(2,0,1).unsqueeze(0), gt_img.permute(2,0,1).unsqueeze(0)))

            if core_mask.sum() > 0:
                loss += 2.0 * l1_loss(render_alpha * core_mask, core_mask)
            bg_mask = 1.0 - gt_mask
            if bg_mask.sum() > 0:
                loss += 5.0 * l1_loss(render_alpha * bg_mask, torch.zeros_like(bg_mask))
            
            if gt_mask.sum() > 0:
                means_cam = self.means @ viewmat_gt[:3, :3].T + viewmat_gt[:3, 3]
                d_cols = means_cam[:, 2:3].expand(-1, 3)
                meta_d = rasterization(
                    self.means, F.normalize(self.quats, dim=-1), 
                    torch.exp(scales), torch.sigmoid(self.opacities), 
                    d_cols, viewmat_gt[None], self.K[None], self.W, self.H, packed=False
                )
                render_d = self._align_shape(meta_d[0][0][..., 0:1])
                loss += lambda_depth * l1_loss(render_d * gt_mask, gt_d * gt_mask)
                if i == iterations - 1: render_d_final = render_d.detach()

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



            # ==============================
            # 2. SDS & Novel View
            # ==============================
            if self.use_sds and i % sds_interval == 0 and i > sds_warmup:
                min_deg = 5.0
                delta_azimuth = (torch.rand(1).item() * 2 - 1) * max_angle_deg 
                delta_elevation = (torch.rand(1).item() * 2 - 1) * (max_angle_deg / 2.0)

                
                total_angle_diff = np.sqrt(delta_azimuth**2 + delta_elevation**2)
                
                if total_angle_diff > min_deg:
                    # [核心修复] 计算物体中心，防止相机看空

                    object_center = torch.median(self.means.detach(), dim=0)[0]
        
                    # [或者] 如果你的数据做过归一化，车应该在原点，可以直接强行设为 0
                    # object_center = torch.zeros(3, device=self.device) 
                    
                    # 调用新的 get_orbit_camera
                    c2w_novel = get_orbit_camera(c2w_curr, delta_azimuth, delta_elevation, center=object_center, device=self.device)
                    if torch.isnan(c2w_novel).any():
                        print("WARNING: NaN detected in novel camera pose! Skipping SDS.")
                        continue
                    # [DEBUG] 打印新旧相机位置距离，如果离得特别远（比如几百米），说明飞了
                    # dist_change = torch.norm(c2w_novel[:3,3] - c2w_curr[:3,3])

                    viewmat_novel = torch.inverse(c2w_novel)
                    
                    meta_novel = rasterization(
                        self.means, F.normalize(self.quats, dim=-1), 
                        torch.exp(scales), torch.sigmoid(self.opacities), 
                        colors_precomp, 
                        viewmat_novel[None], self.K[None], self.W, self.H, packed=False
                    )
                    rgba_novel = self._align_shape(meta_novel[0][0])
                    rgb_novel = rgba_novel[..., :3] + bg_color * (1.0 - rgba_novel[..., 3:4])
                    
                    pred_img = rgb_novel.permute(2, 0, 1).unsqueeze(0) 
                    pred_alpha = rgba_novel[..., 3:4].permute(2, 0, 1).unsqueeze(0)
    
                    # 对渲染结果也做 Crop，保证 Loss 计算时物体也是居中的
                    pred_img_centered = crop_and_resize_differentiable(pred_img, pred_alpha)
                    
                    dynamic_weight = lambda_sds_base * (0.5 + 1.5 * (total_angle_diff / max_angle_deg))
                    rel_pose = self.guidance.compute_relative_pose(c2w_novel)
                    loss_sds_val = self.guidance.sds_loss(pred_img_centered, rel_pose, guidance_scale=3.0)
                    loss += dynamic_weight * loss_sds_val
                    
                    opacity_novel = rgba_novel[..., 3]
                    loss_sparsity = opacity_novel.mean() * 0.5 + (opacity_novel * (1.0 - opacity_novel)).mean() * 0.5
                    loss += 0.01 * loss_sparsity * dynamic_weight

                    # [Debug] 定期保存 Zero123 的输入图，检查是否看空
                    if i % 500 == 0:
                        save_image(pred_img_centered, f"{self.dir_debug}/frame_{frame_idx:03d}_iter_{i:04d}_novel.png")
                        pcd = trimesh.points.PointCloud(self.means.detach().cpu().numpy())
                        pcd.export(os.path.join(self.dir_debug, f"debug_scene_{i}.ply"))

                        # 2. 保存相机位置 (画一个小球代表相机)
                        cam_pos = c2w_novel[:3, 3].detach().cpu().numpy()
                        # 也就是看看 cam_pos 离 pcd 的点云是不是十万八千里远
                        cam_sphere = trimesh.creation.icosphere(radius=0.2)
                        cam_sphere.apply_translation(cam_pos)
                        cam_sphere.export(os.path.join(self.dir_debug, f"debug_cam_{i}.ply"))

                        print(f"[DEBUG] Saved PLY files to {self.dir_debug}. Please visualize them!")

            loss.backward()
            optimizer.step()
            
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
        frames = []
        depth_frames = []
        
        print(f"=== Starting Reconstruction ===")
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
    
    # parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/learn-genmojo/data/car-turn/MegaSAM_Outputs/car-turn_sgd_cvd_hr.npz")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/mega-sam/outputs_cvd/breakdance-flare_sgd_cvd_hr.npz")
    # parser.add_argument("--video_path", type=str, default="/root/autodl-tmp/exp/input/car-turn-sam2.mp4")
    parser.add_argument("--video_path", type=str, default="/root/autodl-tmp/exp/input/breakdance-flare-sam2.mp4")
    parser.add_argument("--exp_name", type=str, default="breakdance-flare")
    parser.add_argument("--output_root", type=str, default="results")
    parser.add_argument("--focal_ratio", type=float, default=0.8)
    parser.add_argument("--use_sds", action="store_true")

    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    folder_name = f"{current_time}_{args.exp_name}"
    full_output_dir = os.path.join(args.output_root, folder_name)

    solver = GaussianSplattingSolver(
        data_path=args.data_path, 
        sam2_video_path=args.video_path, 
        output_dir=full_output_dir,
        focal_ratio=args.focal_ratio,
        use_sds=args.use_sds 
    )
    solver.run()