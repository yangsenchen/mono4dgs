"""
Gaussian Splatting Solver
主要的训练逻辑，包含高斯点的初始化、训练、渲染等功能
"""
import os
import math
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import cv2
from PIL import Image
from gsplat import rasterization
from torchvision.utils import save_image
import trimesh

from utils.camera import compute_lookat_c2w, get_orbit_camera
from utils.geometry import depth_to_normal, normal_to_quaternion, quaternion_to_axes
from utils.losses import (
    compute_per_point_depth_loss,
    compute_normal_alignment_loss,
    compute_depth_pull_loss,
    ssim
)
from utils.image import crop_image_by_mask, crop_and_resize_differentiable
from core.zero123_guide import Zero123Guide


class GaussianSplattingSolver:
    """
    Gaussian Splatting求解器
    
    实现基于深度图的高斯点云重建，支持：
    - 从深度图初始化高斯点
    - 多帧时序重建
    - Zero123 SDS新视角引导
    - 深度和法线约束
    """
    
    def __init__(self, data_path: str, sam2_video_path: str, output_dir: str, 
                 focal_ratio: float = 0.8, use_sds: bool = True):
        """
        初始化求解器
        
        Args:
            data_path: 数据文件路径 (.npz格式)
            sam2_video_path: SAM2分割mask视频路径
            output_dir: 输出目录
            focal_ratio: 焦距比例（默认0.8）
            use_sds: 是否使用SDS引导
        """
        self.device = torch.device("cuda:0")
        self.output_dir = output_dir
        self.use_sds = use_sds 
        
        # 创建输出目录
        self.dir_params = os.path.join(self.output_dir, "params")
        self.dir_images = os.path.join(self.output_dir, "images")
        self.dir_videos = os.path.join(self.output_dir, "videos")
        self.dir_depths = os.path.join(self.output_dir, "depths")
        self.dir_debug = os.path.join(self.output_dir, "debug_sds")
        os.makedirs(self.dir_params, exist_ok=True)
        os.makedirs(self.dir_images, exist_ok=True)
        os.makedirs(self.dir_depths, exist_ok=True)
        os.makedirs(self.dir_videos, exist_ok=True)
        os.makedirs(self.dir_debug, exist_ok=True)
        
        print(f"Output Directory set to: {self.output_dir}")
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)

        self.images = torch.from_numpy(data['images'].astype(np.float32) / 255.0).to(self.device)
        self.depths = torch.from_numpy(data['depths'].astype(np.float32)).to(self.device)
        self.c2ws = torch.from_numpy(data['cam_c2w'].astype(np.float32)).to(self.device)
        self.c2ws[..., 0:3, 1:3] *= -1  # OpenCV -> OpenGL
        
        self.num_frames, self.H, self.W, _ = self.images.shape
        print(f"Data Loaded: {self.num_frames} frames, {self.W}x{self.H}")

        # 相机内参
        if 'intrinsic' in data:
            self.K = torch.from_numpy(data['intrinsic']).to(self.device)
            self.focal = self.K[0, 0].item()
        else:
            self.focal = float(self.W) * focal_ratio
            self.K = torch.tensor([
                [self.focal, 0, self.W / 2.0], 
                [0, self.focal, self.H / 2.0], 
                [0, 0, 1]
            ], device=self.device)
            print(f"Warning: Using guessed focal length {self.focal:.2f}")

        # 加载mask
        self.gt_masks, self.core_masks = self._load_masks_from_video(sam2_video_path)
        if self.gt_masks is None:
            exit()

        # 时序追踪变量
        self.means_prev_all = None
        self.means_prev_2 = None
        self.knn_rigid_indices = None
        self.knn_rigid_weights = None

        # 初始化SDS引导
        if self.use_sds:
            print("[SDS] Initializing Zero123 Guidance Model...")
            self.guidance = Zero123Guide(self.device)

        # 初始化高斯点
        self._init_spheres()

    def _load_masks_from_video(self, video_path):
        """
        从视频加载分割mask
        
        gt_mask: 完整轮廓mask
        core_mask: 腐蚀后的核心mask（更稳定）
        """
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
            if not ret:
                break
            if count >= self.num_frames:
                break 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (gray.shape[1] != self.W) or (gray.shape[0] != self.H):
                gray = cv2.resize(gray, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            
            mask_np = (gray > 127).astype(np.uint8)
            core_np = cv2.erode(mask_np, kernel, iterations=1)
            full_masks.append(torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(-1))
            core_masks.append(torch.from_numpy(core_np.astype(np.float32)).unsqueeze(-1))
            count += 1
            
        cap.release()
        
        # 如果mask帧数不够，用最后一帧填充
        if len(full_masks) < self.num_frames:
            last_full = full_masks[-1]
            last_core = core_masks[-1]
            for _ in range(self.num_frames - len(full_masks)):
                full_masks.append(last_full)
                core_masks.append(last_core)
                
        return torch.stack(full_masks).to(self.device), torch.stack(core_masks).to(self.device)

    def _align_shape(self, tensor):
        """对齐张量形状"""
        if tensor.shape[0] == self.W and tensor.shape[1] == self.H:
            return tensor.permute(1, 0, 2)
        return tensor

    def _init_spheres(self):
        """
        初始化高斯椭球
        
        从第一帧深度图生成初始高斯点，并用表面法线初始化四元数
        """
        print("Initializing Gaussian Ellipsoids (Anisotropic) with Normal-aligned Orientation...")
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
        
        # 计算像素级大小的 radii
        avg_depth = d.mean().item()
        pixel_size_3d = avg_depth / self.focal
        target_radius = pixel_size_3d * 1.5
        target_radii = math.log(target_radius)
        
        print(f"[*] Avg depth: {avg_depth:.3f}, Pixel size 3D: {pixel_size_3d:.6f}")
        print(f"[*] Target radius: {target_radius:.6f}, Target radii (log): {target_radii:.3f}")
        
        # 初始化 radii
        self.radii = torch.ones((N, 3), device=self.device) * target_radii
        self.radii[:, 2] = target_radii - 1.5  # z轴更薄
        
        # 保存像素级 radii 边界
        self.pixel_radii_max = target_radii + 0.5
        self.pixel_radii_min = target_radii - 3.0
        
        # 用深度图法线初始化四元数
        print("[*] Computing surface normals from depth map...")
        normal_map_cam = depth_to_normal(depth, self.K, mask.float())
        normals_cam = normal_map_cam[valid]
        
        # 转换到世界坐标系
        R_c2w = c2w[:3, :3]
        normals_world = (R_c2w @ normals_cam.T).T
        
        # 归一化
        normal_norms = torch.norm(normals_world, dim=-1, keepdim=True)
        valid_normal = normal_norms.squeeze() > 0.1
        normals_world = F.normalize(normals_world, dim=-1)
        
        # 转换为四元数
        self.quats = normal_to_quaternion(normals_world)
        
        # 无效法线使用单位四元数
        invalid_normal = ~valid_normal
        if invalid_normal.sum() > 0:
            self.quats[invalid_normal, 0] = 1.0
            self.quats[invalid_normal, 1:] = 0.0
        
        print(f"[*] Initialized {N} quaternions from surface normals ({valid_normal.sum().item()} valid normals)")
        
        # 初始化颜色和透明度
        rgb_vals = self.images[idx][valid]
        self.rgbs = torch.log(torch.clamp(rgb_vals, 0.01, 0.99) / (1 - rgb_vals))
        self.opacities = torch.ones((N), device=self.device) * 4.0 
        
        # 设置梯度
        self.means.requires_grad_(True)
        self.radii.requires_grad_(True)
        self.quats.requires_grad_(True)
        self.rgbs.requires_grad_(True)
        self.opacities.requires_grad_(True)
        
        # 梯度累积
        self.grad_accum = torch.zeros(N, device=self.device)
        self.denom = torch.zeros(N, device=self.device)
        print(f"[*] Initialized {N} Gaussians (Ellipsoids) with Normal-aligned Orientation.")

    def _compute_knn(self, points, k=20):
        """计算K近邻"""
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

    def _densify_and_prune(self, frame_idx=0, prune_depth_outliers=True):
        """Densify高梯度点，并剔除深度异常值"""
        with torch.no_grad():
            grads = self.grad_accum / self.denom
            grads[self.denom == 0] = 0.0
            to_clone = grads >= 0.0002
            
            if to_clone.sum() > 0:
                self.means = torch.cat([self.means, self.means[to_clone]]).detach().requires_grad_(True)
                self.radii = torch.cat([self.radii, self.radii[to_clone]]).detach().requires_grad_(True)
                self.rgbs = torch.cat([self.rgbs, self.rgbs[to_clone]]).detach().requires_grad_(True)
                self.quats = torch.cat([self.quats, self.quats[to_clone]]).detach().requires_grad_(True)
                self.opacities = torch.cat([self.opacities, self.opacities[to_clone]]).detach().requires_grad_(True)
            
            # 深度异常值剔除
            if prune_depth_outliers and hasattr(self, 'depths') and hasattr(self, 'gt_masks'):
                depth_outlier_mask = self._detect_depth_outliers(frame_idx, threshold=0.5)
                if depth_outlier_mask.sum() > 0:
                    keep_mask = ~depth_outlier_mask
                    print(f"[Prune] Removing {depth_outlier_mask.sum().item()} depth outliers out of {self.means.shape[0]} points")
                    self.means = self.means[keep_mask].detach().requires_grad_(True)
                    self.radii = self.radii[keep_mask].detach().requires_grad_(True)
                    self.rgbs = self.rgbs[keep_mask].detach().requires_grad_(True)
                    self.quats = self.quats[keep_mask].detach().requires_grad_(True)
                    self.opacities = self.opacities[keep_mask].detach().requires_grad_(True)
            
            self.grad_accum = torch.zeros(self.means.shape[0], device=self.device)
            self.denom = torch.zeros(self.means.shape[0], device=self.device)
            return to_clone.sum() > 0
    
    def _detect_depth_outliers(self, frame_idx, threshold=0.5):
        """检测深度异常值"""
        N = self.means.shape[0]
        device = self.device
        
        gt_depth = self.depths[frame_idx].unsqueeze(-1)
        gt_mask = self.gt_masks[frame_idx]
        c2w = self.c2ws[frame_idx]
        viewmat = torch.inverse(c2w)
        
        # 变换到相机坐标系
        means_homo = torch.cat([self.means, torch.ones(N, 1, device=device)], dim=-1)
        means_cam = (viewmat @ means_homo.T).T[:, :3]
        z_cam = means_cam[:, 2]
        
        # 投影
        means_2d = means_cam @ self.K.T
        u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)
        v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)
        
        # 有效性检查
        valid_depth = z_cam > 0.01
        valid_u = (u >= 0) & (u < self.W - 1)
        valid_v = (v >= 0) & (v < self.H - 1)
        valid_proj = valid_depth & valid_u & valid_v
        
        # 采样GT深度
        u_norm = (2.0 * u / (self.W - 1) - 1.0)
        v_norm = (2.0 * v / (self.H - 1) - 1.0)
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
        
        gt_depth_4d = gt_depth.squeeze(-1).unsqueeze(0).unsqueeze(0)
        gt_mask_4d = gt_mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
        
        sampled_depth = F.grid_sample(gt_depth_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        sampled_mask = F.grid_sample(gt_mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        
        # 计算相对深度误差
        rel_error = torch.abs(z_cam - sampled_depth) / (sampled_depth + 1e-8)
        
        # 异常值判定
        is_outlier = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01) & (rel_error > threshold)
        
        return is_outlier
    
    def _snap_points_to_depth_hard(self, frame_idx, remove_outliers=True, depth_tolerance=0.05):
        """硬约束深度对齐：强制将所有高斯点完全对齐到GT深度上"""
        N = self.means.shape[0]
        device = self.device
        
        gt_depth = self.depths[frame_idx].unsqueeze(-1)
        gt_mask = self.gt_masks[frame_idx]
        c2w = self.c2ws[frame_idx]
        viewmat = torch.inverse(c2w)
        
        # 变换到相机坐标系
        means_homo = torch.cat([self.means.data, torch.ones(N, 1, device=device)], dim=-1)
        means_cam = (viewmat @ means_homo.T).T[:, :3]
        z_cam = means_cam[:, 2]
        
        # 投影
        means_2d = means_cam @ self.K.T
        u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)
        v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)
        
        # 有效性检查
        valid_depth = z_cam > 0.01
        valid_u = (u >= 0) & (u < self.W - 1)
        valid_v = (v >= 0) & (v < self.H - 1)
        valid_proj = valid_depth & valid_u & valid_v
        
        # 采样GT深度
        u_norm = (2.0 * u / (self.W - 1) - 1.0)
        v_norm = (2.0 * v / (self.H - 1) - 1.0)
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
        
        gt_depth_4d = gt_depth.squeeze(-1).unsqueeze(0).unsqueeze(0)
        gt_mask_4d = gt_mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
        
        sampled_depth = F.grid_sample(gt_depth_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        sampled_mask = F.grid_sample(gt_mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        
        # Step 1: 100%对齐mask内的点到GT深度
        snap_mask = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01)
        
        if snap_mask.sum() > 0:
            target_z = sampled_depth[snap_mask]
            current_z = z_cam[snap_mask]
            scale_factor = target_z / (current_z + 1e-8)
            
            means_cam_snap = means_cam[snap_mask] * scale_factor.unsqueeze(-1)
            means_cam_snap_homo = torch.cat([means_cam_snap, torch.ones(means_cam_snap.shape[0], 1, device=device)], dim=-1)
            means_world_snap = (c2w @ means_cam_snap_homo.T).T[:, :3]
            
            self.means.data[snap_mask] = means_world_snap
        
        # Step 2: 删除mask外的点
        if remove_outliers:
            outside_mask = valid_proj & (sampled_mask < 0.3)
            invalid_points = (~valid_proj) | outside_mask
            keep_mask = ~invalid_points
            
            if (~keep_mask).sum() > 0:
                n_removed = (~keep_mask).sum().item()
                self.means = self.means[keep_mask].detach().requires_grad_(True)
                self.radii = self.radii[keep_mask].detach().requires_grad_(True)
                self.rgbs = self.rgbs[keep_mask].detach().requires_grad_(True)
                self.quats = self.quats[keep_mask].detach().requires_grad_(True)
                self.opacities = self.opacities[keep_mask].detach().requires_grad_(True)
                
                self.grad_accum = torch.zeros(self.means.shape[0], device=device)
                self.denom = torch.zeros(self.means.shape[0], device=device)
                
                print(f"[Hard Snap] Removed {n_removed} outlier points, {self.means.shape[0]} remaining")
                return True
        
        return False
    
    def _enforce_normal_alignment(self, frame_idx, strength=0.5):
        """强制高斯点的方向与表面法线对齐（硬约束）"""
        N = self.means.shape[0]
        device = self.device
        
        gt_depth = self.depths[frame_idx].unsqueeze(-1)
        gt_mask = self.gt_masks[frame_idx]
        c2w = self.c2ws[frame_idx]
        viewmat = torch.inverse(c2w)
        
        # 计算法线图
        normal_map = depth_to_normal(gt_depth.squeeze(-1), self.K, gt_mask.squeeze(-1))
        
        # 投影高斯点到图像平面
        means_homo = torch.cat([self.means.data, torch.ones(N, 1, device=device)], dim=-1)
        means_cam = (viewmat @ means_homo.T).T[:, :3]
        z_cam = means_cam[:, 2]
        
        means_2d = means_cam @ self.K.T
        u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)
        v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)
        
        # 有效性检查
        valid_depth = z_cam > 0.01
        valid_u = (u >= 0) & (u < self.W - 1)
        valid_v = (v >= 0) & (v < self.H - 1)
        valid_proj = valid_depth & valid_u & valid_v
        
        if valid_proj.sum() < 10:
            return
        
        # 采样法线
        u_norm = (2.0 * u / (self.W - 1) - 1.0)
        v_norm = (2.0 * v / (self.H - 1) - 1.0)
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
        
        normal_4d = normal_map.permute(2, 0, 1).unsqueeze(0)
        mask_4d = gt_mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
        
        sampled_normal = F.grid_sample(normal_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_normal = sampled_normal.squeeze().T
        sampled_mask = F.grid_sample(mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        
        sampled_normal = F.normalize(sampled_normal, dim=-1)
        
        # 过滤有效点
        valid_mask = valid_proj & (sampled_mask > 0.5)
        normal_valid = torch.norm(sampled_normal, dim=-1) > 0.5
        valid_mask = valid_mask & normal_valid
        
        if valid_mask.sum() < 10:
            return
        
        # 转换到世界坐标系
        R_c2w = c2w[:3, :3]
        normals_world = (R_c2w @ sampled_normal[valid_mask].T).T
        normals_world = F.normalize(normals_world, dim=-1)
        
        # 计算目标四元数
        target_quats = normal_to_quaternion(normals_world)
        
        # SLERP插值
        current_quats = self.quats.data[valid_mask]
        dot = (current_quats * target_quats).sum(dim=-1, keepdim=True)
        target_quats = torch.where(dot < 0, -target_quats, target_quats)
        
        new_quats = (1 - strength) * current_quats + strength * target_quats
        new_quats = F.normalize(new_quats, dim=-1)
        
        self.quats.data[valid_mask] = new_quats

    def _enforce_depth_constraint(self, frame_idx):
        """强制所有点贴合GT深度（硬约束）"""
        N = self.means.shape[0]
        device = self.device
        
        gt_depth = self.depths[frame_idx].unsqueeze(-1)
        gt_mask = self.gt_masks[frame_idx]
        c2w = self.c2ws[frame_idx]
        viewmat = torch.inverse(c2w)
        
        # 变换到相机坐标系
        means_homo = torch.cat([self.means.data, torch.ones(N, 1, device=device)], dim=-1)
        means_cam = (viewmat @ means_homo.T).T[:, :3]
        z_cam = means_cam[:, 2]
        
        # 投影
        means_2d = means_cam @ self.K.T
        u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)
        v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)
        
        # 有效性检查
        valid_depth = z_cam > 0.01
        valid_u = (u >= 0) & (u < self.W - 1)
        valid_v = (v >= 0) & (v < self.H - 1)
        valid_proj = valid_depth & valid_u & valid_v
        
        # 采样GT深度
        u_norm = (2.0 * u / (self.W - 1) - 1.0)
        v_norm = (2.0 * v / (self.H - 1) - 1.0)
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
        
        gt_depth_4d = gt_depth.squeeze(-1).unsqueeze(0).unsqueeze(0)
        gt_mask_4d = gt_mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
        
        sampled_depth = F.grid_sample(gt_depth_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        sampled_mask = F.grid_sample(gt_mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        
        snap_mask = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01)
        
        if snap_mask.sum() < 10:
            return
        
        target_z = sampled_depth[snap_mask]
        current_z = z_cam[snap_mask]
        scale_factor = target_z / (current_z + 1e-8)
        
        means_cam_snap = means_cam[snap_mask] * scale_factor.unsqueeze(-1)
        means_cam_snap_homo = torch.cat([means_cam_snap, torch.ones(means_cam_snap.shape[0], 1, device=device)], dim=-1)
        means_world_snap = (c2w @ means_cam_snap_homo.T).T[:, :3]
        
        self.means.data[snap_mask] = means_world_snap

    def save_params(self, frame_idx):
        """保存高斯参数"""
        save_path = os.path.join(self.dir_params, f"gaussians_{frame_idx:03d}.pt")
        torch.save({
            'means': self.means.detach().cpu(),
            'quats': self.quats.detach().cpu(),
            'radii': self.radii.detach().cpu(),
            'rgbs': self.rgbs.detach().cpu(),
            'opacities': self.opacities.detach().cpu(),
            'num_points': self.means.shape[0]
        }, save_path)

    def render_freewheel_video(self, frame_idx, video_length=60, scale_factor=0.8):
        """渲染新视角视频"""
        save_path = os.path.join(self.dir_videos, f"frame_{frame_idx:03d}_novel.mp4")
        temp_path = os.path.join(self.dir_videos, f"frame_{frame_idx:03d}_novel_temp.mp4")
        writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.W, self.H))
        
        # 确定物体中心
        if self.means.shape[0] > 0:
            object_center = torch.median(self.means.detach(), dim=0)[0]
        else:
            object_center = torch.zeros(3, device=self.device, dtype=torch.float32)
            
        c2w_curr = self.c2ws[frame_idx]
        cam_pos_curr = c2w_curr[:3, 3]
        
        # 获取当前相机的基向量
        current_right = F.normalize(c2w_curr[:3, 0], dim=0)
        current_up = F.normalize(c2w_curr[:3, 1], dim=0)
        
        original_radius = torch.norm(cam_pos_curr - object_center)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
        bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device, dtype=torch.float32)
        
        scales = torch.exp(self.radii)
        opacities = torch.sigmoid(self.opacities)
        colors = torch.cat([torch.sigmoid(self.rgbs), torch.ones_like(self.rgbs[:, :1])], dim=1)
        
        print(f"[Novel View] Rendering Figure-8 video for frame {frame_idx}...")
        
        with torch.no_grad():
            for t_step in range(video_length):
                t = 2.0 * np.pi * float(t_step) / float(video_length)
                
                offset_x = scale_factor * np.sin(t)
                offset_y = (scale_factor * 0.5) * np.sin(2 * t)
                
                pos_temp = cam_pos_curr + current_right * offset_x + current_up * offset_y
                
                direction_to_center = F.normalize(object_center - pos_temp, dim=0)
                pos_new = object_center - direction_to_center * original_radius
                
                c2w_novel = compute_lookat_c2w(pos_new, object_center, world_up)
                viewmat_novel = torch.inverse(c2w_novel)
                
                meta = rasterization(
                    self.means, F.normalize(self.quats, dim=-1), 
                    scales, opacities, 
                    colors, 
                    viewmat_novel[None], self.K[None], self.W, self.H, packed=False
                )
                
                rgba = self._align_shape(meta[0][0])
                rgb = rgba[..., :3] + bg_color * (1.0 - rgba[..., 3:4])
                
                rgb_np = (rgb.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                writer.write(rgb_bgr)
                
        writer.release()
        
        # 用 ffmpeg 转码为 H.264
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', temp_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', save_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_path)
        
        print(f"[Novel View] Saved video to {save_path}")

    def _create_optimizer(self):
        """创建优化器"""
        return optim.Adam([
            {'params': [self.means], 'lr': 0.00016},
            {'params': [self.quats], 'lr': 0.001},
            {'params': [self.rgbs], 'lr': 0.0025},
            {'params': [self.radii], 'lr': 0.001},
            {'params': [self.opacities], 'lr': 0.05}
        ], lr=0.001)

    def train_frame(self, frame_idx, iterations):
        """训练单帧"""
        allow_densify = True if frame_idx == 0 else (frame_idx % 5 == 0)
        optimizer = self._create_optimizer()
        l1_loss = nn.L1Loss()
        
        # 数据准备
        gt_mask = self.gt_masks[frame_idx]
        core_mask = self.core_masks[frame_idx]
        gt_img_raw = self.images[frame_idx]
        gt_img = gt_img_raw * gt_mask + (1.0 - gt_mask) * 1.0 
        
        # Zero123 Condition 准备
        if self.use_sds:
            ref_img_chw = gt_img_raw.permute(2, 0, 1) 
            ref_mask_chw = gt_mask.permute(2, 0, 1)   
            ref_img_masked_raw = ref_img_chw * ref_mask_chw
            ref_img_crop = crop_image_by_mask(ref_img_masked_raw, ref_mask_chw)
            self.guidance.prepare_condition(ref_img_crop.unsqueeze(0), self.c2ws[frame_idx])
            save_image(ref_img_crop, f"{self.dir_debug}/ref_crop_frame_{frame_idx:03d}.png")

        gt_d = self.depths[frame_idx].unsqueeze(-1)
        c2w_curr = self.c2ws[frame_idx]
        viewmat_gt = torch.inverse(c2w_curr)
        bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        # 超参数
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

            # GT View 渲染
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

            loss = 0.8 * l1_loss(render_rgb, gt_img) + 0.2 * (1.0 - ssim(
                render_rgb.permute(2,0,1).unsqueeze(0), 
                gt_img.permute(2,0,1).unsqueeze(0)
            ))

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
                if i == iterations - 1:
                    render_d_final = render_d.detach()
                
                # 逐点深度约束
                per_point_depth_loss = compute_per_point_depth_loss(
                    self.means, gt_d, gt_mask, viewmat_gt, self.K, self.W, self.H
                )
                lambda_per_point = 1.0 if i < 500 else 0.5
                loss += lambda_per_point * per_point_depth_loss
                
                # 深度拉扯约束
                depth_pull_loss = compute_depth_pull_loss(
                    self.means, gt_d, gt_mask, viewmat_gt, self.K, self.W, self.H, c2w_curr, strength=0.5
                )
                loss += depth_pull_loss
                
                # 法线对齐约束
                if not hasattr(self, '_cached_normal_map') or self._cached_frame_idx != frame_idx:
                    self._cached_normal_map = depth_to_normal(gt_d.squeeze(-1), self.K, gt_mask.squeeze(-1))
                    self._cached_frame_idx = frame_idx
                
                normal_map = self._cached_normal_map
                normal_alignment_loss = compute_normal_alignment_loss(
                    self.means, self.quats, normal_map, gt_mask, 
                    viewmat_gt, self.K, self.W, self.H, c2w_curr
                )
                lambda_normal = 2.0 if i < 500 else 1.0
                loss += lambda_normal * normal_alignment_loss
                
                # 椭球形状约束
                radii_z = self.radii[:, 2]
                radii_xy_min = torch.min(self.radii[:, 0], self.radii[:, 1])
                margin = 1.0
                flatness_violation = F.relu(radii_z - (radii_xy_min - margin))
                flatness_loss = flatness_violation.mean()
                
                max_aspect_ratio = 3.0
                max_log_ratio = math.log(max_aspect_ratio)
                radii_max = torch.max(self.radii, dim=1)[0]
                radii_min = torch.min(self.radii, dim=1)[0]
                log_ratio = radii_max - radii_min
                aspect_violation = F.relu(log_ratio - max_log_ratio)
                aspect_loss = aspect_violation.mean()
                
                radii_min_bound = self.pixel_radii_min
                radii_max_bound = self.pixel_radii_max
                too_small = F.relu(radii_min_bound - self.radii)
                too_large = F.relu(self.radii - radii_max_bound)
                range_loss = too_small.mean() + (too_large ** 2).mean() * 10.0
                
                radii_std = torch.std(self.radii, dim=1)
                isotropy_loss = radii_std.mean()
                
                lambda_flatness = 1.0
                lambda_aspect = 3.0
                lambda_range = 5.0
                lambda_isotropy = 0.2
                
                shape_loss = (lambda_flatness * flatness_loss + 
                             lambda_aspect * aspect_loss + 
                             lambda_range * range_loss +
                             lambda_isotropy * isotropy_loss)
                loss += shape_loss

            # KNN Rigid Loss
            if frame_idx > 0 and self.knn_rigid_indices is not None:
                n_prev = self.means_prev_all.shape[0]
                n_curr = self.means.shape[0]
                if n_curr >= n_prev:
                    curr_rigid = self.means[:n_prev]
                    max_idx = self.knn_rigid_indices.max().item()
                    if max_idx < n_prev:
                        neighbors_curr = curr_rigid[self.knn_rigid_indices]
                        curr_exp = curr_rigid.unsqueeze(1).expand(-1, 20, -1)
                        dist_curr = torch.norm(neighbors_curr - curr_exp, dim=-1)
                        neighbors_prev = self.means_prev_all[self.knn_rigid_indices]
                        prev_exp = self.means_prev_all.unsqueeze(1).expand(-1, 20, -1)
                        dist_prev = torch.norm(neighbors_prev - prev_exp, dim=-1)
                        loss += 10.0 * (self.knn_rigid_weights * torch.abs(dist_curr - dist_prev)).mean()

            # SDS & Novel View
            if self.use_sds and i % sds_interval == 0 and i > sds_warmup:
                min_deg = 5.0
                delta_azimuth = (torch.rand(1).item() * 2 - 1) * max_angle_deg 
                delta_elevation = (torch.rand(1).item() * 2 - 1) * (max_angle_deg / 2.0)
                total_angle_diff = np.sqrt(delta_azimuth**2 + delta_elevation**2)
                
                if total_angle_diff > min_deg:
                    object_center = torch.median(self.means.detach(), dim=0)[0]
                    c2w_novel = get_orbit_camera(c2w_curr, delta_azimuth, delta_elevation, center=object_center, device=self.device)
                    
                    if torch.isnan(c2w_novel).any():
                        print("WARNING: NaN detected in novel camera pose! Skipping SDS.")
                        continue
                    
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
                    pred_img_centered = crop_and_resize_differentiable(pred_img, pred_alpha)
                    
                    dynamic_weight = lambda_sds_base * (0.5 + 1.5 * (total_angle_diff / max_angle_deg))
                    rel_pose = self.guidance.compute_relative_pose(c2w_novel)
                    loss_sds_val = self.guidance.sds_loss(pred_img_centered, rel_pose, guidance_scale=3.0)
                    loss += dynamic_weight * loss_sds_val
                    
                    opacity_novel = rgba_novel[..., 3]
                    loss_sparsity = opacity_novel.mean() * 0.5 + (opacity_novel * (1.0 - opacity_novel)).mean() * 0.5
                    loss += 0.01 * loss_sparsity * dynamic_weight

                    # Debug
                    if i % 500 == 0:
                        save_image(pred_img_centered, f"{self.dir_debug}/frame_{frame_idx:03d}_iter_{i:04d}_novel.png")
                        pcd = trimesh.points.PointCloud(self.means.detach().cpu().numpy())
                        pcd.export(os.path.join(self.dir_debug, f"debug_scene_{i}.ply"))
                        cam_pos = c2w_novel[:3, 3].detach().cpu().numpy()
                        cam_sphere = trimesh.creation.icosphere(radius=0.2)
                        cam_sphere.apply_translation(cam_pos)
                        cam_sphere.export(os.path.join(self.dir_debug, f"debug_cam_{i}.ply"))
                        print(f"[DEBUG] Saved PLY files to {self.dir_debug}. Please visualize them!")

            loss.backward()
            optimizer.step()
            
            # 硬约束
            with torch.no_grad():
                self._enforce_depth_constraint(frame_idx)
                
                if i % 10 == 0:
                    self._enforce_normal_alignment(frame_idx, strength=0.3)
                
                self.radii.data.clamp_(min=self.pixel_radii_min, max=self.pixel_radii_max)
                
                # 限制长宽比
                max_log_ratio = 1.1
                radii_max, max_indices = self.radii.data.max(dim=1)
                radii_min, min_indices = self.radii.data.min(dim=1)
                current_ratio = radii_max - radii_min
                needs_fix = current_ratio > max_log_ratio
                
                if needs_fix.any():
                    excess = current_ratio[needs_fix] - max_log_ratio
                    adjustment = excess / 2.0
                    fix_indices = torch.where(needs_fix)[0]
                    for j in range(len(fix_indices)):
                        idx = fix_indices[j].item()
                        ma = max_indices[idx].item()
                        mi = min_indices[idx].item()
                        adj = adjustment[j].item()
                        self.radii.data[idx, ma] -= adj
                        self.radii.data[idx, mi] += adj
            
            # Densify and Prune
            if allow_densify and i > 100 and i < iterations - 100 and i % 300 == 0:
                if self.means.grad is not None:
                    self.grad_accum += torch.norm(self.means.grad[:, :2], dim=-1)
                    self.denom += 1.0
                added = self._densify_and_prune(frame_idx=frame_idx, prune_depth_outliers=True)
                if added:
                    optimizer = self._create_optimizer()
            
            # 定期清理漂浮点
            if i > 0 and i % 500 == 0:
                with torch.no_grad():
                    need_rebuild = self._snap_points_to_depth_hard(frame_idx, remove_outliers=True)
                    if need_rebuild:
                        optimizer = self._create_optimizer()

        # 最终清理
        with torch.no_grad():
            self._snap_points_to_depth_hard(frame_idx, remove_outliers=True)
            print(f"[Final] Frame {frame_idx} complete. {self.means.shape[0]} Gaussians remaining.")
        
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
        """运行完整的重建流程"""
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
            if t > 0:
                self.means_prev_2 = self.means_prev_all.clone()
            else:
                self.means_prev_2 = self.means.detach().clone()

            img_pil.save(f"{self.dir_images}/frame_{t:03d}.png")
            depth_pil.save(f"{self.dir_depths}/depth_{t:03d}.png")

            self.render_freewheel_video(t, video_length=60, scale_factor=0.8)
            
        frames[0].save(f"{self.output_dir}/result_rgb.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
        depth_frames[0].save(f"{self.output_dir}/result_depth.gif", save_all=True, append_images=depth_frames[1:], duration=40, loop=0)
        
        print(f"Done! Results saved to {self.output_dir}")

