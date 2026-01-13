"""
Gaussian Splatting Solver
主要的训练逻辑，包含高斯点的初始化、训练、渲染等功能
"""
import os
import math
import subprocess
import shutil
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
from scipy import ndimage

from utils.camera import compute_lookat_c2w, get_orbit_camera
from utils.geometry import depth_to_normal, normal_to_quaternion, quaternion_to_axes
from utils.losses import (
    # compute_per_point_depth_loss,
    # compute_normal_alignment_loss,
    # compute_depth_pull_loss,
    # compute_behind_surface_loss,
    # compute_opacity_entropy_loss,
    # compute_depth_variance_loss,
    ssim
)
# from utils.image import crop_image_by_mask, crop_and_resize_differentiable
# from core.zero123_guide import Zero123Guide


# Flying pixel 移除后的无效深度标记值
# INVALID_DEPTH_VALUE = -1.0


# def propagate_depth_inpaint(depth, mask, invalid_value=INVALID_DEPTH_VALUE, max_iterations=100):
#     """
#     使用邻域传播方法填充无效深度区域
    
#     对于被标记为无效的深度像素（flying pixels），使用mask内的有效邻域深度进行传播填充。
#     这样可以保证高斯点在这些区域仍然能获得合理的深度监督。
    
#     Args:
#         depth: [H, W] 深度图 (torch.Tensor or np.ndarray)
#         mask: [H, W] 前景mask，只在mask内传播 (torch.Tensor or np.ndarray)
#         invalid_value: 无效深度的标记值
#         max_iterations: 最大迭代次数
    
#     Returns:
#         propagated_depth: [H, W] 传播填充后的深度图
#         invalid_mask: [H, W] 原始无效区域的mask（用于后续可选的软监督）
#     """
#     # 转换为numpy进行处理
#     is_torch = isinstance(depth, torch.Tensor)
#     device = depth.device if is_torch else None
    
#     if is_torch:
#         depth_np = depth.cpu().numpy().copy()
#         mask_np = mask.cpu().numpy().copy() if isinstance(mask, torch.Tensor) else mask.copy()
#     else:
#         depth_np = depth.copy()
#         mask_np = mask.copy()
    
#     # 确保mask是2D
#     if mask_np.ndim == 3:
#         mask_np = mask_np[..., 0]
    
#     # 识别无效深度区域
#     invalid_mask = (depth_np == invalid_value) | (depth_np <= 0)
    
#     # 只处理mask内的无效区域
#     foreground_mask = mask_np > 0.5
#     need_fill = invalid_mask & foreground_mask
    
#     if not need_fill.any():
#         # 没有需要填充的区域
#         if is_torch:
#             return depth.clone(), torch.zeros_like(depth, dtype=torch.bool)
#         return depth_np, invalid_mask
    
#     # 有效深度区域（mask内且深度有效）
#     valid_depth = foreground_mask & (~invalid_mask) & (depth_np > 0.01)
    
#     # 使用距离变换找到最近的有效深度
#     # 方法：迭代扩展有效区域
#     propagated = depth_np.copy()
#     filled = valid_depth.copy()
    
#     # 定义邻域（8邻域）
#     kernel = np.array([[1, 1, 1],
#                        [1, 0, 1],
#                        [1, 1, 1]], dtype=np.float32)
    
#     for iteration in range(max_iterations):
#         # 找到当前需要填充且有有效邻居的像素
#         unfilled = need_fill & (~filled)
#         if not unfilled.any():
#             break
        
#         # 对有效深度进行卷积，计算邻域平均
#         valid_depth_masked = np.where(filled, propagated, 0).astype(np.float32)
#         valid_count = ndimage.convolve(filled.astype(np.float32), kernel, mode='constant', cval=0)
#         depth_sum = ndimage.convolve(valid_depth_masked, kernel, mode='constant', cval=0)
        
#         # 计算邻域平均深度
#         with np.errstate(divide='ignore', invalid='ignore'):
#             avg_depth = np.where(valid_count > 0, depth_sum / valid_count, 0)
        
#         # 填充有有效邻居的无效像素
#         can_fill = unfilled & (valid_count > 0)
#         propagated[can_fill] = avg_depth[can_fill]
#         filled[can_fill] = True
    
#     # 对于仍然无法填充的区域（孤立区域），使用全局mask内深度均值
#     still_unfilled = need_fill & (~filled)
#     if still_unfilled.any():
#         global_mean = propagated[filled].mean() if filled.any() else 1.0
#         propagated[still_unfilled] = global_mean
    
#     # 转回torch
#     if is_torch:
#         propagated_tensor = torch.from_numpy(propagated).to(device)
#         invalid_mask_tensor = torch.from_numpy(need_fill).to(device)
#         return propagated_tensor, invalid_mask_tensor
    
#     return propagated, need_fill


# def propagate_depth_bilateral(depth, rgb, mask, invalid_value=INVALID_DEPTH_VALUE, 
#                                sigma_space=10, sigma_color=0.1):
#     """
#     使用双边滤波风格的深度传播
    
#     考虑颜色相似性进行深度传播，边缘处更准确
    
#     Args:
#         depth: [H, W] 深度图
#         rgb: [H, W, 3] RGB图像 (0-1范围)
#         mask: [H, W] 前景mask
#         invalid_value: 无效深度标记值
#         sigma_space: 空间高斯核标准差
#         sigma_color: 颜色相似性标准差
    
#     Returns:
#         propagated_depth: [H, W] 传播后的深度图
#         invalid_mask: [H, W] 原始无效区域mask
#     """
#     is_torch = isinstance(depth, torch.Tensor)
#     device = depth.device if is_torch else None
    
#     if is_torch:
#         depth_np = depth.cpu().numpy().copy()
#         rgb_np = rgb.cpu().numpy().copy() if isinstance(rgb, torch.Tensor) else rgb.copy()
#         mask_np = mask.cpu().numpy().copy() if isinstance(mask, torch.Tensor) else mask.copy()
#     else:
#         depth_np = depth.copy()
#         rgb_np = rgb.copy()
#         mask_np = mask.copy()
    
#     if mask_np.ndim == 3:
#         mask_np = mask_np[..., 0]
    
#     H, W = depth_np.shape
    
#     # 识别无效区域
#     invalid_mask = (depth_np == invalid_value) | (depth_np <= 0)
#     foreground_mask = mask_np > 0.5
#     need_fill = invalid_mask & foreground_mask
    
#     if not need_fill.any():
#         if is_torch:
#             return depth.clone(), torch.zeros_like(depth, dtype=torch.bool)
#         return depth_np, invalid_mask
    
#     propagated = depth_np.copy()
    
#     # 获取需要填充的像素坐标
#     fill_coords = np.where(need_fill)
#     valid_mask = foreground_mask & (~invalid_mask) & (depth_np > 0.01)
    
#     # 对每个需要填充的像素，找邻域内的有效深度并加权平均
#     window_size = int(sigma_space * 3)
    
#     for i in range(len(fill_coords[0])):
#         y, x = fill_coords[0][i], fill_coords[1][i]
        
#         # 定义搜索窗口
#         y_min = max(0, y - window_size)
#         y_max = min(H, y + window_size + 1)
#         x_min = max(0, x - window_size)
#         x_max = min(W, x + window_size + 1)
        
#         # 提取窗口
#         window_valid = valid_mask[y_min:y_max, x_min:x_max]
#         window_depth = depth_np[y_min:y_max, x_min:x_max]
#         window_rgb = rgb_np[y_min:y_max, x_min:x_max]
        
#         if not window_valid.any():
#             continue
        
#         # 计算空间权重
#         yy, xx = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
#         spatial_dist = np.sqrt((yy - y)**2 + (xx - x)**2)
#         spatial_weight = np.exp(-spatial_dist**2 / (2 * sigma_space**2))
        
#         # 计算颜色权重
#         center_color = rgb_np[y, x]
#         color_dist = np.sqrt(np.sum((window_rgb - center_color)**2, axis=-1))
#         color_weight = np.exp(-color_dist**2 / (2 * sigma_color**2))
        
#         # 组合权重
#         weight = spatial_weight * color_weight * window_valid.astype(np.float32)
        
#         if weight.sum() > 0:
#             propagated[y, x] = (window_depth * weight).sum() / weight.sum()
    
#     # 处理仍未填充的像素
#     still_unfilled = need_fill & (propagated == invalid_value)
#     if still_unfilled.any():
#         # 使用简单传播作为后备
#         propagated_simple, _ = propagate_depth_inpaint(
#             propagated, mask_np, invalid_value=invalid_value
#         )
#         propagated[still_unfilled] = propagated_simple[still_unfilled]
    
#     if is_torch:
#         propagated_tensor = torch.from_numpy(propagated).to(device)
#         invalid_mask_tensor = torch.from_numpy(need_fill).to(device)
#         return propagated_tensor, invalid_mask_tensor
    
#     return propagated, need_fill


class GaussianSplattingSolver:
    """
    Gaussian Splatting求解器
    
    实现基于深度图的高斯点云重建，支持：
    - 从深度图初始化高斯点
    - 多帧时序重建
    - Zero123 SDS新视角引导
    - 深度和法线约束
    """
    
    def __init__(self, dataset_path: str, output_dir: str, 
                 focal_ratio: float = 0.8, use_sds: bool = True, use_mask: bool = False):
        """
        初始化求解器
        
        Args:
            dataset_path: 数据集目录路径 (包含images/, aligned_depth_anything/, masks/子文件夹和droid_recon.npy)
            output_dir: 输出目录
            focal_ratio: 焦距比例（默认0.8）
            use_sds: 是否使用SDS引导
            use_mask: 是否使用mask进行训练（默认False，训练整张图像）
        """
        self.device = torch.device("cuda:0")
        self.output_dir = output_dir
        self.use_sds = use_sds
        self.use_mask = use_mask 
        
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
        print(f"Loading data from {dataset_path}...")
        
        # 加载数据
        self.images, self.depths_raw, self.c2ws, self.K = self._load_data_from_folders(dataset_path)
        
        self.num_frames, self.H, self.W, _ = self.images.shape
        self.focal = self.K[0, 0].item()
        print(f"Data Loaded: {self.num_frames} frames, {self.W}x{self.H}")
        print(f"Focal length: {self.focal:.2f}")
        
        # 不再使用flying pixel masks，直接使用原始深度
        self.has_flying_pixel_masks = False
        self.flying_pixel_masks = None
        print("[Depth] Using aligned depth maps directly")

        # 加载mask（如果启用）
        if self.use_mask:
            print("[Mask] Loading masks for training...")
            self.gt_masks = self._load_masks_from_images(dataset_path)
            if self.gt_masks is None:
                print("[Mask] Failed to load masks, exiting...")
                exit()
        else:
            print("[Mask] use_mask=False, training on whole image without mask")
            # 创建全1的mask（整张图像）
            self.gt_masks = torch.ones((self.num_frames, self.H, self.W, 1), device=self.device)
        
        # 处理无效深度：传播填充
        # self.depths, self.invalid_depth_masks = self._process_invalid_depths()
        self.depths = self.depths_raw.clone()
        self.invalid_depth_masks = torch.zeros_like(self.depths, dtype=torch.bool)
        
        # 时序追踪变量
        self.means_prev_all = None
        self.means_prev_2 = None
        self.knn_rigid_indices = None
        self.knn_rigid_weights = None

        # 初始化SDS引导
        # if self.use_sds:
        #     print("[SDS] Initializing Zero123 Guidance Model...")
        #     self.guidance = Zero123Guide(self.device)

        # 初始化高斯点
        self._init_spheres()

    # def _process_invalid_depths(self):
    #     """
    #     处理无效深度区域（flying pixels被移除的区域）
        
    #     使用邻域深度传播来填充无效区域，同时保存原始无效区域的mask
    #     以便后续可以对这些区域使用软监督
        
    #     Returns:
    #         depths: [N, H, W] 传播填充后的深度图
    #         invalid_masks: [N, H, W] 每帧的无效深度区域mask
    #     """
    #     if not self.has_flying_pixel_masks:
    #         # 没有 flying pixel 信息，直接返回原始深度
    #         invalid_masks = torch.zeros_like(self.depths_raw, dtype=torch.bool)
    #         return self.depths_raw.clone(), invalid_masks
        
    #     print("[Flying Pixel] Propagating depth for invalid regions...")
        
    #     depths_propagated = []
    #     invalid_masks = []
        
    #     for frame_idx in range(self.num_frames):
    #         depth = self.depths_raw[frame_idx]
    #         rgb = self.images[frame_idx]
    #         mask = self.gt_masks[frame_idx, ..., 0]  # [H, W]
            
    #         # 使用双边滤波风格的传播（考虑颜色边缘）
    #         propagated, invalid_mask = propagate_depth_bilateral(
    #             depth, rgb, mask,
    #             invalid_value=INVALID_DEPTH_VALUE,
    #             sigma_space=15,  # 空间范围
    #             sigma_color=0.15  # 颜色相似性阈值
    #         )
            
    #         depths_propagated.append(propagated)
    #         invalid_masks.append(invalid_mask)
            
    #         if frame_idx == 0 or (frame_idx + 1) % 10 == 0:
    #             n_invalid = invalid_mask.sum().item()
    #             print(f"  Frame {frame_idx}: {n_invalid} pixels propagated")
        
    #     depths = torch.stack(depths_propagated, dim=0)
    #     invalid_masks = torch.stack(invalid_masks, dim=0)
        
    #     # 保存可视化（第一帧）
    #     self._save_depth_propagation_debug(0, self.depths_raw[0], depths[0], invalid_masks[0])
        
    #     print(f"[Flying Pixel] Depth propagation complete.")
    #     return depths, invalid_masks
    
    # def _save_depth_propagation_debug(self, frame_idx, depth_raw, depth_propagated, invalid_mask):
    #     """保存深度传播的debug可视化"""
    #     import matplotlib
    #     matplotlib.use('Agg')
    #     import matplotlib.pyplot as plt
        
    #     debug_dir = os.path.join(self.output_dir, "debug_depth_propagation")
    #     os.makedirs(debug_dir, exist_ok=True)
        
    #     depth_raw_np = depth_raw.cpu().numpy()
    #     depth_prop_np = depth_propagated.cpu().numpy()
    #     invalid_np = invalid_mask.cpu().numpy()
        
    #     # 计算有效范围
    #     valid_raw = (depth_raw_np > 0.01) & (depth_raw_np != INVALID_DEPTH_VALUE)
    #     if valid_raw.any():
    #         vmin, vmax = np.percentile(depth_raw_np[valid_raw], [1, 99])
    #     else:
    #         vmin, vmax = 0, 5
        
    #     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
    #     # 原始深度
    #     ax = axes[0]
    #     depth_display = depth_raw_np.copy()
    #     depth_display[depth_raw_np == INVALID_DEPTH_VALUE] = np.nan
    #     im = ax.imshow(depth_display, cmap='turbo', vmin=vmin, vmax=vmax)
    #     ax.set_title(f'Raw Depth (Frame {frame_idx})')
    #     ax.axis('off')
    #     plt.colorbar(im, ax=ax, fraction=0.046)
        
    #     # 无效区域
    #     ax = axes[1]
    #     ax.imshow(invalid_np, cmap='Reds')
    #     ax.set_title(f'Invalid Regions ({invalid_np.sum()} pixels)')
    #     ax.axis('off')
        
    #     # 传播后深度
    #     ax = axes[2]
    #     im = ax.imshow(depth_prop_np, cmap='turbo', vmin=vmin, vmax=vmax)
    #     ax.set_title('Propagated Depth')
    #     ax.axis('off')
    #     plt.colorbar(im, ax=ax, fraction=0.046)
        
    #     # 差异图
    #     ax = axes[3]
    #     diff = np.abs(depth_prop_np - depth_raw_np)
    #     diff[~invalid_np] = 0  # 只显示传播区域的差异
    #     im = ax.imshow(diff, cmap='hot')
    #     ax.set_title('Propagated Regions')
    #     ax.axis('off')
    #     plt.colorbar(im, ax=ax, fraction=0.046)
        
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(debug_dir, f"propagation_frame_{frame_idx:03d}.png"), dpi=150)
    #     plt.close()
        
    #     print(f"  [Debug] Saved propagation visualization to {debug_dir}")

    def _load_data_from_folders(self, dataset_path):
        """
        从文件夹结构加载数据
        
        Args:
            dataset_path: 数据集根目录，包含:
                - images/: PNG图像文件
                - aligned_depth_anything/: NPY深度文件
                - droid_recon.npy: 相机位姿和内参
        
        Returns:
            images: [N, H, W, 3] RGB图像 (0-1范围)
            depths: [N, H, W] 深度图
            c2ws: [N, 4, 4] 相机到世界的变换矩阵
            K: [3, 3] 相机内参矩阵
        """
        images_dir = os.path.join(dataset_path, "images")
        depths_dir = os.path.join(dataset_path, "aligned_depth_anything")
        poses_file = os.path.join(dataset_path, "droid_recon.npy")
        
        # 加载相机位姿和内参
        print(f"Loading camera poses from {poses_file}...")
        droid_data = np.load(poses_file, allow_pickle=True).item()
        c2ws_np = droid_data['traj_c2w']  # [N, 4, 4]
        intrinsics = droid_data['intrinsics']  # [fx, fy, cx, cy]
        original_img_shape = droid_data['img_shape']  # (H, W) from droid reconstruction
        
        # 获取图像文件列表以确定实际图像尺寸
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        first_img = Image.open(os.path.join(images_dir, image_files[0]))
        W, H = first_img.size
        
        # 缩放内参以匹配实际图像尺寸
        # droid_recon 中的内参是针对较小的重建图像，需要缩放到实际图像尺寸
        orig_h, orig_w = original_img_shape
        scale_w = W / orig_w
        scale_h = H / orig_h
        
        fx, fy, cx, cy = intrinsics
        fx_scaled = fx * scale_w
        fy_scaled = fy * scale_h
        cx_scaled = cx * scale_w
        cy_scaled = cy * scale_h
        
        print(f"Scaling intrinsics from {original_img_shape} to ({H}, {W})")
        print(f"  Scale factors: w={scale_w:.2f}, h={scale_h:.2f}")
        print(f"  Original: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        print(f"  Scaled: fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, cx={cx_scaled:.2f}, cy={cy_scaled:.2f}")
        
        # 构建内参矩阵
        K = torch.tensor([
            [fx_scaled, 0, cx_scaled],
            [0, fy_scaled, cy_scaled],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        # 转换相机位姿 (OpenCV -> OpenGL坐标系)
        c2ws = torch.from_numpy(c2ws_np.astype(np.float32)).to(self.device)
        c2ws[..., 0:3, 1:3] *= -1  # 翻转Y和Z轴
        
        num_pose_frames = c2ws.shape[0]

        # 计算将要加载的帧数（images/depths/poses 取最小）
        num_frames_total = len(image_files)
        if num_pose_frames != num_frames_total:
            print(f"Warning: pose frames ({num_pose_frames}) != image frames ({num_frames_total}), using min")
        num_frames = min(num_pose_frames, num_frames_total)

        # 预分配数组
        images_list = []
        depths_list = []
        
        print("Loading images and depths...")
        for i, img_file in enumerate(image_files[:num_frames]):
            # 加载图像
            img_path = os.path.join(images_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            images_list.append(torch.from_numpy(img_np))
            
            # 加载深度 (aligned_depth_anything存储的是disparity，需要转换为depth)
            depth_file = img_file.replace('.png', '.npy')
            depth_path = os.path.join(depths_dir, depth_file)
            disp_np = np.load(depth_path).astype(np.float32)
            disp_np = np.clip(disp_np, a_min=1e-6, a_max=1e6)
            depth_np = 1.0 / disp_np
            depths_list.append(torch.from_numpy(depth_np))
            
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i + 1}/{num_frames} frames")
        
        images = torch.stack(images_list).to(self.device)
        depths = torch.stack(depths_list).to(self.device)
        
        print(f"Loaded {num_frames} frames: images {images.shape}, depths {depths.shape}")
        
        return images, depths, c2ws[:num_frames], K

    def _load_masks_from_images(self, dataset_path):
        """
        从图像文件夹加载分割mask
        
        Args:
            dataset_path: 数据集根目录，包含masks/子文件夹
        
        Returns:
            gt_masks: [N, H, W, 1] 完整轮廓mask
            core_masks: [N, H, W, 1] 腐蚀后的核心mask
        """
        masks_dir = os.path.join(dataset_path, "masks")
        print(f"Loading masks from {masks_dir}...")
        
        if not os.path.exists(masks_dir):
            print(f"Error: Masks directory not found: {masks_dir}")
            return None, None
        
        # 获取mask文件列表
        mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')])
        
        if len(mask_files) == 0:
            print(f"Error: No mask files found in {masks_dir}")
            return None, None
        
        full_masks = []
        core_masks = []
        kernel = np.ones((15, 15), np.uint8)
        
        print(f"Found {len(mask_files)} mask files")
        for i, mask_file in enumerate(mask_files):
            if i >= self.num_frames:
                break
            
            mask_path = os.path.join(masks_dir, mask_file)
            mask_img = Image.open(mask_path).convert('RGB')
            mask_np = np.array(mask_img)
            
            # 调整尺寸
            if mask_np.shape[1] != self.W or mask_np.shape[0] != self.H:
                mask_np = cv2.resize(mask_np, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            
            # 检测前景：假设mask中非黑色区域为前景
            # 如果mask是二值的，取任意通道；如果是彩色的，转换为灰度
            if len(mask_np.shape) == 3:
                # 使用绿色通道或者整体亮度判断前景
                mask_gray = mask_np.max(axis=2)  # 取最大通道值
            else:
                mask_gray = mask_np
            
            # 二值化：大于阈值的为前景
            mask_binary = (mask_gray > 30).astype(np.uint8)
            
            # 腐蚀得到核心mask
            core_np = cv2.erode(mask_binary, kernel, iterations=1)
            
            full_masks.append(torch.from_numpy(mask_binary.astype(np.float32)).unsqueeze(-1))
            core_masks.append(torch.from_numpy(core_np.astype(np.float32)).unsqueeze(-1))
            
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i + 1}/{len(mask_files)} masks")
        
        # 如果mask帧数不够，用最后一帧填充
        if len(full_masks) < self.num_frames:
            last_full = full_masks[-1]
            last_core = core_masks[-1]
            for _ in range(self.num_frames - len(full_masks)):
                full_masks.append(last_full)
                core_masks.append(last_core)
        
        print(f"Loaded {len(full_masks)} masks")
        return torch.stack(full_masks).to(self.device) # , torch.stack(core_masks).to(self.device)

    # def _load_masks_from_video(self, mask_path):
    #     """
    #     从视频加载分割mask
        
    #     绿色区域为前景（foreground），其余部分为背景
        
    #     gt_mask: 完整轮廓mask
    #     core_mask: 腐蚀后的核心mask（更稳定）
    #     """
    #     print(f"Loading masks from video: {mask_path}")
    #     if not os.path.exists(mask_path):
    #         print(f"Error: Video file not found: {mask_path}")
    #         return None, None
            
    #     cap = cv2.VideoCapture(mask_path)
    #     full_masks = []
    #     core_masks = []
    #     kernel = np.ones((15, 15), np.uint8) 
        
    #     count = 0
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         if count >= self.num_frames:
    #             break 
            
    #         # 调整尺寸
    #         if (frame.shape[1] != self.W) or (frame.shape[0] != self.H):
    #             frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            
    #         # 转换到HSV颜色空间检测绿色
    #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
    #         # 定义绿色的HSV范围（宽松一些以适应不同的绿色）
    #         # H: 35-85 (绿色范围), S: 50-255 (饱和度), V: 50-255 (亮度)
    #         lower_green = np.array([35, 50, 50])
    #         upper_green = np.array([85, 255, 255])
            
    #         # 创建绿色mask
    #         mask_np = cv2.inRange(hsv, lower_green, upper_green)
    #         mask_np = (mask_np > 0).astype(np.uint8)
            
    #         core_np = cv2.erode(mask_np, kernel, iterations=1)
    #         full_masks.append(torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(-1))
    #         core_masks.append(torch.from_numpy(core_np.astype(np.float32)).unsqueeze(-1))
    #         count += 1
            
    #     cap.release()
        
    #     # 如果mask帧数不够，用最后一帧填充
    #     if len(full_masks) < self.num_frames:
    #         last_full = full_masks[-1]
    #         last_core = core_masks[-1]
    #         for _ in range(self.num_frames - len(full_masks)):
    #             full_masks.append(last_full)
    #             core_masks.append(last_core)
                
    #     return torch.stack(full_masks).to(self.device), torch.stack(core_masks).to(self.device)

    def _align_shape(self, tensor):
        """对齐张量形状"""
        if tensor.shape[0] == self.W and tensor.shape[1] == self.H:
            return tensor.permute(1, 0, 2)
        return tensor

    def _copy_code_to_results(self):
        """将所有Python文件拷贝到结果目录的code子文件夹"""
        code_dir = os.path.join(self.output_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        
        # 获取项目根目录（假设是当前工作目录的上一级或当前目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        print(f"[Code Backup] Copying Python files from {project_root} to {code_dir}...")
        
        copied_count = 0
        # 遍历项目目录，复制所有.py文件
        for root, dirs, files in os.walk(project_root):
            # 跳过隐藏目录、结果目录、缓存目录等
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['results', '__pycache__', 'outputs']]
            
            for file in files:
                if file.endswith('.py'):
                    src_path = os.path.join(root, file)
                    # 保持相对目录结构
                    rel_path = os.path.relpath(src_path, project_root)
                    dst_path = os.path.join(code_dir, rel_path)
                    
                    # 创建目标目录
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
        
        print(f"[Code Backup] Copied {copied_count} Python files to {code_dir}")

    def _init_spheres(self):
        """
        初始化高斯椭球
        
        从第一帧深度图生成初始高斯点，并用表面法线初始化四元数
        """
        print("Initializing Gaussian Ellipsoids (Anisotropic) with Normal-aligned Orientation...")
        idx = 0
        depth = self.depths[idx]
        if self.use_mask:
            mask = self.gt_masks[idx, ..., 0] > 0.5
        else:
            # 使用整张图像
            mask = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)
        
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
            # if prune_depth_outliers and hasattr(self, 'depths') and hasattr(self, 'gt_masks'):
            #     depth_outlier_mask = self._detect_depth_outliers(frame_idx, threshold=0.5)
            #     if depth_outlier_mask.sum() > 0:
            #         keep_mask = ~depth_outlier_mask
            #         print(f"[Prune] Removing {depth_outlier_mask.sum().item()} depth outliers out of {self.means.shape[0]} points")
            #         self.means = self.means[keep_mask].detach().requires_grad_(True)
            #         self.radii = self.radii[keep_mask].detach().requires_grad_(True)
            #         self.rgbs = self.rgbs[keep_mask].detach().requires_grad_(True)
            #         self.quats = self.quats[keep_mask].detach().requires_grad_(True)
            #         self.opacities = self.opacities[keep_mask].detach().requires_grad_(True)
            
            self.grad_accum = torch.zeros(self.means.shape[0], device=self.device)
            self.denom = torch.zeros(self.means.shape[0], device=self.device)
            return to_clone.sum() > 0
    
    # def _detect_depth_outliers(self, frame_idx, threshold=0.5):
    #     """检测深度异常值"""
    #     N = self.means.shape[0]
    #     device = self.device
        
    #     gt_depth = self.depths[frame_idx].unsqueeze(-1)
    #     gt_mask = self.gt_masks[frame_idx]
    #     c2w = self.c2ws[frame_idx]
    #     viewmat = torch.inverse(c2w)
        
    #     # 变换到相机坐标系
    #     means_homo = torch.cat([self.means, torch.ones(N, 1, device=device)], dim=-1)
    #     means_cam = (viewmat @ means_homo.T).T[:, :3]
    #     z_cam = means_cam[:, 2]
        
    #     # 投影
    #     means_2d = means_cam @ self.K.T
    #     u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)
    #     v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)
        
    #     # 有效性检查
    #     valid_depth = z_cam > 0.01
    #     valid_u = (u >= 0) & (u < self.W - 1)
    #     valid_v = (v >= 0) & (v < self.H - 1)
    #     valid_proj = valid_depth & valid_u & valid_v
        
    #     # 采样GT深度
    #     u_norm = (2.0 * u / (self.W - 1) - 1.0)
    #     v_norm = (2.0 * v / (self.H - 1) - 1.0)
    #     grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
        
    #     gt_depth_4d = gt_depth.squeeze(-1).unsqueeze(0).unsqueeze(0)
    #     gt_mask_4d = gt_mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
        
    #     sampled_depth = F.grid_sample(gt_depth_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
    #     sampled_mask = F.grid_sample(gt_mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        
    #     # 计算相对深度误差
    #     rel_error = torch.abs(z_cam - sampled_depth) / (sampled_depth + 1e-8)
        
    #     # 异常值判定
    #     is_outlier = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01) & (rel_error > threshold)
        
    #     return is_outlier
    
    # def _prune_behind_surface(self, frame_idx, margin=0.05, opacity_threshold=None):
    #     """
    #     强制删除位于GT深度表面后面的点。
        
    #     Args:
    #         frame_idx: 当前帧索引
    #         margin: 容差，允许高斯点在表面后方的一小段距离内 (例如 5cm)
    #         opacity_threshold: 如果设置，只删除 opacity < threshold 的背面点；
    #                           如果为 None，则删除所有背面点
        
    #     Returns:
    #         bool: 是否有点被删除（需要重建优化器）
    #     """
    #     with torch.no_grad():
    #         N = self.means.shape[0]
    #         device = self.device
            
    #         gt_depth = self.depths[frame_idx]
    #         gt_mask = self.gt_masks[frame_idx]
    #         c2w = self.c2ws[frame_idx]
    #         viewmat = torch.inverse(c2w)
            
    #         H, W = gt_depth.shape[0], gt_depth.shape[1]
    #         gt_depth_2d = gt_depth.squeeze(-1) if gt_depth.dim() == 3 else gt_depth
    #         gt_mask_2d = gt_mask.squeeze(-1) if gt_mask.dim() == 3 else gt_mask
            
    #         # 1. 将高斯点转换到相机坐标系
    #         means_homo = torch.cat([self.means, torch.ones(N, 1, device=device)], dim=-1)
    #         means_cam = (viewmat @ means_homo.T).T[:, :3]
    #         z_cam = means_cam[:, 2]
            
    #         # 2. 投影到图像平面
    #         means_2d = means_cam @ self.K.T
    #         z_safe = z_cam.clamp(min=1e-5)
    #         u = means_2d[:, 0] / z_safe
    #         v = means_2d[:, 1] / z_safe
            
    #         # 3. 筛选在图像范围内的点
    #         valid_u = (u >= 0) & (u < W - 1)
    #         valid_v = (v >= 0) & (v < H - 1)
    #         valid_depth = z_cam > 0.1
    #         valid_proj = valid_u & valid_v & valid_depth
            
    #         if valid_proj.sum() == 0:
    #             return False
            
    #         # 4. 采样对应位置的 GT 深度和 mask
    #         u_norm = (2.0 * u / (W - 1) - 1.0)
    #         v_norm = (2.0 * v / (H - 1) - 1.0)
    #         grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
            
    #         gt_depth_sample = F.grid_sample(
    #             gt_depth_2d.unsqueeze(0).unsqueeze(0), 
    #             grid, 
    #             mode='nearest', 
    #             padding_mode='zeros', 
    #             align_corners=True
    #         ).squeeze()  # [N]
            
    #         gt_mask_sample = F.grid_sample(
    #             gt_mask_2d.float().unsqueeze(0).unsqueeze(0), 
    #             grid, 
    #             mode='nearest', 
    #             padding_mode='zeros', 
    #             align_corners=True
    #         ).squeeze()  # [N]
            
    #         # 5. 判定为背面的点
    #         # 条件：z > gt_depth + margin 且在 mask 内
    #         is_behind = (z_cam > (gt_depth_sample + margin)) & (gt_mask_sample > 0.5) & (gt_depth_sample > 0.01)
            
    #         # 6. 可选：只删除低 opacity 的背面点
    #         if opacity_threshold is not None:
    #             opacities_activated = torch.sigmoid(self.opacities)
    #             is_behind = is_behind & (opacities_activated < opacity_threshold)
            
    #         to_kill = is_behind
            
    #         if to_kill.sum() > 0:
    #             print(f"[Prune Behind Surface] Kill {to_kill.sum().item()} points behind surface (margin={margin}m)")
    #             keep_mask = ~to_kill
    #             self.means = self.means[keep_mask].detach().requires_grad_(True)
    #             self.radii = self.radii[keep_mask].detach().requires_grad_(True)
    #             self.rgbs = self.rgbs[keep_mask].detach().requires_grad_(True)
    #             self.quats = self.quats[keep_mask].detach().requires_grad_(True)
    #             self.opacities = self.opacities[keep_mask].detach().requires_grad_(True)
                
    #             # 更新梯度累积器
    #             if hasattr(self, 'grad_accum') and self.grad_accum.shape[0] > keep_mask.sum():
    #                 self.grad_accum = self.grad_accum[keep_mask]
    #                 self.denom = self.denom[keep_mask]
                
    #             return True
            
    #         return False
    
    # def _snap_points_to_depth_hard(self, frame_idx, remove_outliers=True, depth_tolerance=0.05):
    #     """硬约束深度对齐：强制将所有高斯点完全对齐到GT深度上"""
    #     N = self.means.shape[0]
    #     device = self.device
        
    #     gt_depth = self.depths[frame_idx].unsqueeze(-1)
    #     gt_mask = self.gt_masks[frame_idx]
    #     c2w = self.c2ws[frame_idx]
    #     viewmat = torch.inverse(c2w)
        
    #     # 变换到相机坐标系
    #     means_homo = torch.cat([self.means.data, torch.ones(N, 1, device=device)], dim=-1)
    #     means_cam = (viewmat @ means_homo.T).T[:, :3]
    #     z_cam = means_cam[:, 2]
        
    #     # 投影
    #     means_2d = means_cam @ self.K.T
    #     u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)
    #     v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)
        
    #     # 有效性检查
    #     valid_depth = z_cam > 0.01
    #     valid_u = (u >= 0) & (u < self.W - 1)
    #     valid_v = (v >= 0) & (v < self.H - 1)
    #     valid_proj = valid_depth & valid_u & valid_v
        
    #     # 采样GT深度
    #     u_norm = (2.0 * u / (self.W - 1) - 1.0)
    #     v_norm = (2.0 * v / (self.H - 1) - 1.0)
    #     grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
        
    #     gt_depth_4d = gt_depth.squeeze(-1).unsqueeze(0).unsqueeze(0)
    #     gt_mask_4d = gt_mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
        
    #     sampled_depth = F.grid_sample(gt_depth_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
    #     sampled_mask = F.grid_sample(gt_mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        
    #     # Step 1: 100%对齐mask内的点到GT深度
    #     snap_mask = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01)
        
    #     if snap_mask.sum() > 0:
    #         target_z = sampled_depth[snap_mask]
    #         current_z = z_cam[snap_mask]
    #         scale_factor = target_z / (current_z + 1e-8)
            
    #         means_cam_snap = means_cam[snap_mask] * scale_factor.unsqueeze(-1)
    #         means_cam_snap_homo = torch.cat([means_cam_snap, torch.ones(means_cam_snap.shape[0], 1, device=device)], dim=-1)
    #         means_world_snap = (c2w @ means_cam_snap_homo.T).T[:, :3]
            
    #         self.means.data[snap_mask] = means_world_snap
        
    #     # Step 2: 删除mask外的点
    #     # if remove_outliers:
    #     #     outside_mask = valid_proj & (sampled_mask < 0.3)
    #     #     invalid_points = (~valid_proj) | outside_mask
    #     #     keep_mask = ~invalid_points
            
    #     #     if (~keep_mask).sum() > 0:
    #     #         n_removed = (~keep_mask).sum().item()
    #     #         self.means = self.means[keep_mask].detach().requires_grad_(True)
    #     #         self.radii = self.radii[keep_mask].detach().requires_grad_(True)
    #     #         self.rgbs = self.rgbs[keep_mask].detach().requires_grad_(True)
    #     #         self.quats = self.quats[keep_mask].detach().requires_grad_(True)
    #     #         self.opacities = self.opacities[keep_mask].detach().requires_grad_(True)
                
    #     #         self.grad_accum = torch.zeros(self.means.shape[0], device=device)
    #     #         self.denom = torch.zeros(self.means.shape[0], device=device)
                
    #     #         print(f"[Hard Snap] Removed {n_removed} outlier points, {self.means.shape[0]} remaining")
    #     #         return True
        
    #     return False
    
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
        # core_mask = self.core_masks[frame_idx]
        gt_img_raw = self.images[frame_idx]
        if self.use_mask:
            # 使用mask：前景用原图，背景用白色
            gt_img = gt_img_raw * gt_mask + (1.0 - gt_mask) * 1.0
        else:
            # 不使用mask：直接使用原图
            gt_img = gt_img_raw 
        
        # Zero123 Condition 准备
        # if self.use_sds:
        #     ref_img_chw = gt_img_raw.permute(2, 0, 1) 
        #     ref_mask_chw = gt_mask.permute(2, 0, 1)   
        #     ref_img_masked_raw = ref_img_chw * ref_mask_chw
        #     ref_img_crop = crop_image_by_mask(ref_img_masked_raw, ref_mask_chw)
        #     self.guidance.prepare_condition(ref_img_crop.unsqueeze(0), self.c2ws[frame_idx])
        #     save_image(ref_img_crop, f"{self.dir_debug}/ref_crop_frame_{frame_idx:03d}.png")

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

            # if core_mask.sum() > 0:
            #     loss += 2.0 * l1_loss(render_alpha * core_mask, core_mask)
            # bg_mask = 1.0 - gt_mask
            # if bg_mask.sum() > 0:
            #     loss += 5.0 * l1_loss(render_alpha * bg_mask, torch.zeros_like(bg_mask))
            
            # 深度约束（如果使用mask则只在mask区域应用，否则应用到整张图）
            if self.use_mask:
                apply_depth_loss = gt_mask.sum() > 0
            else:
                apply_depth_loss = True
            
            if apply_depth_loss:
                means_cam = self.means @ viewmat_gt[:3, :3].T + viewmat_gt[:3, 3]
                d_vals = means_cam[:, 2:3]  # [N, 1] 每个点的深度
                d_cols = d_vals.expand(-1, 3)  # [N, 3] 用于渲染
                meta_d = rasterization(
                    self.means, F.normalize(self.quats, dim=-1), 
                    torch.exp(scales), torch.sigmoid(self.opacities), 
                    d_cols, viewmat_gt[None], self.K[None], self.W, self.H, packed=False
                )
                render_d = self._align_shape(meta_d[0][0][..., 0:1])
                
                # # 渲染深度平方，用于计算深度方差损失
                # d2_cols = (d_vals ** 2).expand(-1, 3)  # [N, 3] 深度平方
                # meta_d2 = rasterization(
                #     self.means, F.normalize(self.quats, dim=-1), 
                #     torch.exp(scales), torch.sigmoid(self.opacities), 
                #     d2_cols, viewmat_gt[None], self.K[None], self.W, self.H, packed=False
                # )
                # render_d2 = self._align_shape(meta_d2[0][0][..., 0:1])
                
                # # 对传播填充的区域使用软监督（降低权重）
                # if self.has_flying_pixel_masks:
                #     invalid_mask = self.invalid_depth_masks[frame_idx].unsqueeze(-1)
                #     # 有效区域权重=1.0，传播区域权重=0.3
                #     depth_weight = torch.where(invalid_mask, 
                #                                torch.tensor(0.3, device=self.device), 
                #                                torch.tensor(1.0, device=self.device))
                #     depth_loss = (torch.abs(render_d - gt_d) * gt_mask * depth_weight).mean()
                #     loss += lambda_depth * depth_loss
                # else:
                #     loss += lambda_depth * l1_loss(render_d * gt_mask, gt_d * gt_mask)
                if self.use_mask:
                    # 使用mask：只在mask区域计算深度loss
                    loss += lambda_depth * l1_loss(render_d * gt_mask, gt_d * gt_mask)
                else:
                    # 不使用mask：在整张图上计算深度loss
                    loss += lambda_depth * l1_loss(render_d, gt_d)
                
                if i == iterations - 1:
                    render_d_final = render_d.detach()
                
                # 逐点深度约束（传播区域使用软权重）
                # per_point_depth_loss = compute_per_point_depth_loss(
                # #     self.means, gt_d, gt_mask, viewmat_gt, self.K, self.W, self.H,
                # #     invalid_depth_mask=self.invalid_depth_masks[frame_idx] if self.has_flying_pixel_masks else None,
                # #     soft_weight=0.3
                # # )
                # per_point_depth_loss = compute_per_point_depth_loss(
                #     self.means, gt_d, gt_mask, viewmat_gt, self.K, self.W, self.H,
                #     invalid_depth_mask=None,
                #     soft_weight=0.3
                # )
                # lambda_per_point = 1.0 if i < 500 else 0.5
                # loss += lambda_per_point * per_point_depth_loss
                
                # 深度拉扯约束
                # depth_pull_loss = compute_depth_pull_loss(
                #     self.means, gt_d, gt_mask, viewmat_gt, self.K, self.W, self.H, c2w_curr, strength=0.5
                # )
                # loss += depth_pull_loss
                
                # Behind-Surface Opacity Loss: 惩罚位于GT深度表面后面的点的不透明度
                # 这个 loss 每一步都计算，强迫优化器不要在背面生成点
                # margin=0.05 表示允许 5cm 的误差，防止误伤表面本身
                # lambda_behind = 1.0
                # loss_behind = compute_behind_surface_loss(
                #     self.means, self.opacities, gt_d, gt_mask, 
                #     self.K, c2w_curr,
                #     margin=0.05, 
                #     lambda_scale=lambda_behind
                # )
                # loss += loss_behind
                
                # Entropy Loss: 强制不透明度二值化，清理半透明漂浮点
                # 原理：让所有高斯点的opacity要么接近0要么接近1，消除"幽灵点"
                # 权重从0.01开始，随着训练逐渐增加到0.05，让网络先学会颜色再清理
                # lambda_entropy = 0.01 + 0.04 * min(1.0, i / 1500.0)
                # entropy_loss = compute_opacity_entropy_loss(self.opacities)
                # loss += lambda_entropy * entropy_loss
                
                # Depth Variance Loss: 强制所有贡献颜色的高斯点紧贴GT深度表面
                # 原理：普通深度Loss只约束加权平均值，方差损失强制所有点都必须紧紧挨着D_gt
                # 如果一个点跑到了D_gt后面或前面，只要它有权重，(t_i - D_gt)^2就会变大
                # 优化器不得不把这个点推回表面，或者把它的权重降为0
                # lambda_depth_var = 0.5 if i < 500 else 1.0  # 后期增强
                # depth_var_loss = compute_depth_variance_loss(
                #     render_d, render_d2, gt_d, gt_mask, 
                #     render_alpha=render_alpha,
                #     alpha_threshold=0.1
                # )
                # loss += lambda_depth_var * depth_var_loss
                
                # 法线对齐约束
                # if not hasattr(self, '_cached_normal_map') or self._cached_frame_idx != frame_idx:
                #     self._cached_normal_map = depth_to_normal(gt_d.squeeze(-1), self.K, gt_mask.squeeze(-1))
                #     self._cached_frame_idx = frame_idx
                
                # normal_map = self._cached_normal_map
                # normal_alignment_loss = compute_normal_alignment_loss(
                #     self.means, self.quats, normal_map, gt_mask, 
                #     viewmat_gt, self.K, self.W, self.H, c2w_curr
                # )
                # lambda_normal = 2.0 if i < 500 else 1.0
                # loss += lambda_normal * normal_alignment_loss
                
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
                range_loss = (too_large ** 2).mean() * 10.0 # too_small.mean() + 
                
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
            # if self.use_sds and i % sds_interval == 0 and i > sds_warmup:
            #     min_deg = 5.0
            #     delta_azimuth = (torch.rand(1).item() * 2 - 1) * max_angle_deg 
            #     delta_elevation = (torch.rand(1).item() * 2 - 1) * (max_angle_deg / 2.0)
            #     total_angle_diff = np.sqrt(delta_azimuth**2 + delta_elevation**2)
                
            #     if total_angle_diff > min_deg:
            #         object_center = torch.median(self.means.detach(), dim=0)[0]
            #         c2w_novel = get_orbit_camera(c2w_curr, delta_azimuth, delta_elevation, center=object_center, device=self.device)
                    
            #         if torch.isnan(c2w_novel).any():
            #             print("WARNING: NaN detected in novel camera pose! Skipping SDS.")
            #             continue
                    
            #         viewmat_novel = torch.inverse(c2w_novel)
                    
            #         meta_novel = rasterization(
            #             self.means, F.normalize(self.quats, dim=-1), 
            #             torch.exp(scales), torch.sigmoid(self.opacities), 
            #             colors_precomp, 
            #             viewmat_novel[None], self.K[None], self.W, self.H, packed=False
            #         )
            #         rgba_novel = self._align_shape(meta_novel[0][0])
            #         rgb_novel = rgba_novel[..., :3] + bg_color * (1.0 - rgba_novel[..., 3:4])
                    
            #         pred_img = rgb_novel.permute(2, 0, 1).unsqueeze(0) 
            #         pred_alpha = rgba_novel[..., 3:4].permute(2, 0, 1).unsqueeze(0)
            #         pred_img_centered = crop_and_resize_differentiable(pred_img, pred_alpha)
                    
            #         dynamic_weight = lambda_sds_base * (0.5 + 1.5 * (total_angle_diff / max_angle_deg))
            #         rel_pose = self.guidance.compute_relative_pose(c2w_novel)
            #         loss_sds_val = self.guidance.sds_loss(pred_img_centered, rel_pose, guidance_scale=3.0)
            #         loss += dynamic_weight * loss_sds_val
                    
            #         opacity_novel = rgba_novel[..., 3]
            #         loss_sparsity = opacity_novel.mean() * 0.5 + (opacity_novel * (1.0 - opacity_novel)).mean() * 0.5
            #         loss += 0.01 * loss_sparsity * dynamic_weight

            #         # Debug
            #         if i % 500 == 0:
            #             save_image(pred_img_centered, f"{self.dir_debug}/frame_{frame_idx:03d}_iter_{i:04d}_novel.png")
            #             pcd = trimesh.points.PointCloud(self.means.detach().cpu().numpy())
            #             pcd.export(os.path.join(self.dir_debug, f"debug_scene_{i}.ply"))
            #             cam_pos = c2w_novel[:3, 3].detach().cpu().numpy()
            #             cam_sphere = trimesh.creation.icosphere(radius=0.2)
            #             cam_sphere.apply_translation(cam_pos)
            #             cam_sphere.export(os.path.join(self.dir_debug, f"debug_cam_{i}.ply"))
            #             print(f"[DEBUG] Saved PLY files to {self.dir_debug}. Please visualize them!")

            # Debug: 打印 loss
            if i % 100 == 0:
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}"
                    # "L_rgb": f"{(0.8 * l1_loss(render_rgb, gt_img)).item():.4f}"
                    # , 
                    # "L_behind": f"{loss_behind.item():.4f}"
                })

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
            
            # 每 100 步清理表面后的点
            # if i > 0 and i % 100 == 0:
            #     need_rebuild = self._prune_behind_surface(frame_idx, margin=0.05, opacity_threshold=None)
            #     if need_rebuild:
            #         optimizer = self._create_optimizer()
            
            # 定期清理漂浮点
            # if i > 0 and i % 500 == 0:
            #     with torch.no_grad():
            #         need_rebuild = self._snap_points_to_depth_hard(frame_idx, remove_outliers=True)
            #         if need_rebuild:
            #             optimizer = self._create_optimizer()

        # 最终清理
        # with torch.no_grad():
        #     self._snap_points_to_depth_hard(frame_idx, remove_outliers=True)
        #     print(f"[Final] Frame {frame_idx} complete. {self.means.shape[0]} Gaussians remaining.")
        
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
        
        # 在训练开始前备份所有Python代码
        self._copy_code_to_results()
        
        for t in range(self.num_frames):
            iters = 1000 if t == 0 else 1000
            
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
            
            # self.save_params(t)
            
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

            self.render_freewheel_video(t, video_length=120, scale_factor=0.5)
            
        frames[0].save(f"{self.output_dir}/result_rgb.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
        depth_frames[0].save(f"{self.output_dir}/result_depth.gif", save_all=True, append_images=depth_frames[1:], duration=40, loop=0)
        
        print(f"Done! Results saved to {self.output_dir}")

