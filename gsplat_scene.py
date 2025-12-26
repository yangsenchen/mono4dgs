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

def compute_lookat_c2w(camera_pos, target_pos, up_vector=None):
    """
    LookAt 函数 (适配您的 gsplat 设置：Z轴指向物体)
    """
    if up_vector is None:
        up_vector = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device, dtype=torch.float32)
    
    # 【修复】：按照您原始代码的逻辑，Z轴指向物体 (Forward)
    z_axis = F.normalize(target_pos - camera_pos, dim=0) 
    
    # 【修复】：计算 X 轴 (Right)
    # 原始代码逻辑是 -cross(up, z)，这里保持一致
    x_axis = -F.normalize(torch.cross(up_vector, z_axis, dim=0), dim=0)
    
    # 计算 Y 轴 (Down/Up)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=0), dim=0)
    
    c2w = torch.eye(4, device=camera_pos.device, dtype=torch.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = camera_pos
    
    return c2w

def compute_per_point_depth_loss(means, gt_depth, gt_mask, viewmat, K, W, H, margin=0.1):
    """
    逐点深度约束：直接约束每个高斯点的深度贴合GT深度
    
    核心思路：
    1. 对于每个高斯点，投影到图像平面得到 (u, v)
    2. 查询该像素位置的GT深度 d_gt
    3. 约束高斯点的相机坐标系深度 z 等于 d_gt
    
    Args:
        means: [N, 3] 高斯点3D位置（世界坐标系）
        gt_depth: [H, W, 1] GT深度图
        gt_mask: [H, W, 1] 物体mask
        viewmat: [4, 4] 相机view矩阵
        K: [3, 3] 相机内参
        W, H: 图像尺寸
        margin: 只对mask内部的点施加约束
    
    Returns:
        loss: 标量损失值
    """
    N = means.shape[0]
    device = means.device
    
    # Step 1: 将高斯点从世界坐标系变换到相机坐标系
    # viewmat = inverse(c2w), 所以 p_cam = viewmat @ p_world
    means_homo = torch.cat([means, torch.ones(N, 1, device=device)], dim=-1)  # [N, 4]
    means_cam = (viewmat @ means_homo.T).T[:, :3]  # [N, 3]
    
    z_cam = means_cam[:, 2]  # 高斯点在相机坐标系的深度
    
    # Step 2: 投影到图像平面
    # p_img = K @ p_cam (齐次除法)
    means_2d = means_cam @ K.T  # [N, 3]
    u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)  # x / z
    v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)  # y / z
    
    # Step 3: 过滤有效点（在图像内部、深度为正、在mask内）
    valid_depth = z_cam > 0.01
    valid_u = (u >= 0) & (u < W - 1)
    valid_v = (v >= 0) & (v < H - 1)
    valid_proj = valid_depth & valid_u & valid_v
    
    if valid_proj.sum() < 10:
        return torch.tensor(0.0, device=device)
    
    # Step 4: 采样GT深度（双线性插值）
    # 转换为归一化坐标 [-1, 1] for grid_sample
    u_norm = (2.0 * u / (W - 1) - 1.0)
    v_norm = (2.0 * v / (H - 1) - 1.0)
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
    
    # gt_depth: [H, W, 1] -> [1, 1, H, W]
    gt_depth_4d = gt_depth.squeeze(-1).unsqueeze(0).unsqueeze(0)
    gt_mask_4d = gt_mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
    
    # 采样得到每个点对应的GT深度和mask
    sampled_depth = F.grid_sample(gt_depth_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    sampled_mask = F.grid_sample(gt_mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    sampled_depth = sampled_depth.squeeze()  # [N]
    sampled_mask = sampled_mask.squeeze()    # [N]
    
    # Step 5: 只对mask内且投影有效的点计算loss
    valid_mask = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01)
    
    if valid_mask.sum() < 10:
        return torch.tensor(0.0, device=device)
    
    # Step 6: 计算深度差异loss（L1 或 L2）
    z_diff = torch.abs(z_cam[valid_mask] - sampled_depth[valid_mask])
    loss = z_diff.mean()
    
    return loss


def depth_to_normal(depth, K, mask=None):
    """
    从深度图计算表面法线（相机坐标系）
    
    使用有限差分计算深度梯度，然后推导法线方向
    
    Args:
        depth: [H, W] 或 [H, W, 1] 深度图
        K: [3, 3] 相机内参矩阵
        mask: [H, W] 或 [H, W, 1] 可选的mask，只计算mask内的法线
    
    Returns:
        normals: [H, W, 3] 法线图（相机坐标系，指向相机）
    """
    if depth.dim() == 3:
        depth = depth.squeeze(-1)
    if mask is not None and mask.dim() == 3:
        mask = mask.squeeze(-1)
    
    H, W = depth.shape
    device = depth.device
    
    # 提取相机内参
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # 创建像素坐标网格
    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u = u.float()
    v = v.float()
    
    # 反投影到3D点（相机坐标系）
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 使用Sobel算子计算梯度（更稳定）
    # 或者使用简单的有限差分
    # 这里使用中心差分，更准确
    
    # 计算 x, y, z 方向的偏导数
    # dz/du, dz/dv
    dz_du = torch.zeros_like(z)
    dz_dv = torch.zeros_like(z)
    dx_du = torch.zeros_like(z)
    dx_dv = torch.zeros_like(z)
    dy_du = torch.zeros_like(z)
    dy_dv = torch.zeros_like(z)
    
    # 中心差分 (避免边界问题)
    dz_du[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / 2.0
    dz_dv[1:-1, :] = (z[2:, :] - z[:-2, :]) / 2.0
    
    dx_du[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2.0
    dx_dv[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2.0
    
    dy_du[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / 2.0
    dy_dv[1:-1, :] = (y[2:, :] - y[:-2, :]) / 2.0
    
    # 边界使用前向/后向差分
    dz_du[:, 0] = z[:, 1] - z[:, 0]
    dz_du[:, -1] = z[:, -1] - z[:, -2]
    dz_dv[0, :] = z[1, :] - z[0, :]
    dz_dv[-1, :] = z[-1, :] - z[-2, :]
    
    dx_du[:, 0] = x[:, 1] - x[:, 0]
    dx_du[:, -1] = x[:, -1] - x[:, -2]
    dx_dv[0, :] = x[1, :] - x[0, :]
    dx_dv[-1, :] = x[-1, :] - x[-2, :]
    
    dy_du[:, 0] = y[:, 1] - y[:, 0]
    dy_du[:, -1] = y[:, -1] - y[:, -2]
    dy_dv[0, :] = y[1, :] - y[0, :]
    dy_dv[-1, :] = y[-1, :] - y[-2, :]
    
    # 构建切向量
    # 沿u方向的切向量: (dx/du, dy/du, dz/du)
    # 沿v方向的切向量: (dx/dv, dy/dv, dz/dv)
    tu = torch.stack([dx_du, dy_du, dz_du], dim=-1)  # [H, W, 3]
    tv = torch.stack([dx_dv, dy_dv, dz_dv], dim=-1)  # [H, W, 3]
    
    # 法线 = tu × tv (叉积)
    # 注意：叉积的顺序决定法线方向
    # tu 是沿 u 方向的切向量，tv 是沿 v 方向的切向量
    # tu × tv 给出指向观察者（相机）方向的法线
    normals = torch.cross(tu, tv, dim=-1)  # [H, W, 3]
    
    # 归一化
    norm_length = torch.norm(normals, dim=-1, keepdim=True) + 1e-8
    normals = normals / norm_length
    
    # 确保法线指向相机外（即从表面指向空间）
    # 在相机坐标系中，相机在原点看向+z，物体在z>0位置
    # 所以面向相机的表面法线应该z分量<0（指向-z，即指向相机）
    # 但我们希望法线指向"表面外部"，对于面向相机的表面就是指向相机
    # 如果z分量为正（法线指向物体内部），则翻转
    flip_mask = (normals[..., 2:3] > 0).float()
    normals = normals * (1 - 2 * flip_mask)
    
    # 对无效区域（深度为0或mask外）设置为零法线
    invalid = (depth < 0.01) | (depth > 100.0)
    if mask is not None:
        invalid = invalid | (mask < 0.5)
    normals[invalid] = 0
    
    return normals


def normal_to_quaternion(normal, up_hint=None):
    """
    将法线向量转换为四元数（让高斯椭球的最短轴对齐法线）
    
    高斯椭球应该"躺"在表面上，即：
    - 椭球的z轴（最短轴）应该与表面法线平行
    - 椭球的x,y轴（较长轴）应该在表面平面内
    
    Args:
        normal: [N, 3] 法线向量（应该已经归一化，指向表面外部）
        up_hint: [3] 可选的上方向参考，用于确定绕法线的旋转
    
    Returns:
        quats: [N, 4] 四元数 (w, x, y, z)
    """
    device = normal.device
    N = normal.shape[0]
    
    if up_hint is None:
        up_hint = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    # 归一化法线
    normal = F.normalize(normal, dim=-1)
    
    # 参考方向：(0, 0, 1) 是默认四元数 (1,0,0,0) 对应的z轴方向
    # 我们要找一个旋转，把 (0, 0, 1) 旋转到 normal
    # 这样高斯椭球的z轴就会与法线对齐
    ref = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, -1)
    
    # 使用 Rodrigues 旋转公式
    # 旋转轴 = ref × normal
    # 旋转角 = arccos(ref · normal)
    
    dot = (ref * normal).sum(dim=-1, keepdim=True)  # [N, 1]
    cross = torch.cross(ref, normal, dim=-1)  # [N, 3]
    
    # 处理平行情况（ref 和 normal 已经对齐）
    cross_norm = torch.norm(cross, dim=-1, keepdim=True) + 1e-8
    axis = cross / cross_norm  # 旋转轴
    
    # 处理反平行情况 (dot ≈ -1，需要旋转180度)
    anti_parallel = dot.squeeze(-1) < -0.999
    if anti_parallel.any():
        # 选择任意垂直轴作为旋转轴
        axis[anti_parallel] = torch.tensor([1.0, 0.0, 0.0], device=device)
    
    # 处理几乎平行情况 (dot ≈ 1，不需要旋转)
    parallel = dot.squeeze(-1) > 0.999
    
    # 计算旋转角
    angle = torch.acos(torch.clamp(dot, -1.0, 1.0))  # [N, 1]
    
    # 四元数: q = (cos(θ/2), sin(θ/2) * axis)
    half_angle = angle / 2.0
    w = torch.cos(half_angle)  # [N, 1]
    xyz = torch.sin(half_angle) * axis  # [N, 3]
    
    quats = torch.cat([w, xyz], dim=-1)  # [N, 4]
    
    # 对于几乎平行的情况，使用单位四元数
    if parallel.any():
        quats[parallel] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    
    # 归一化四元数
    quats = F.normalize(quats, dim=-1)
    
    return quats


def quaternion_to_axes(quats):
    """
    从四元数提取三个局部坐标轴
    
    Args:
        quats: [N, 4] 四元数 (w, x, y, z)
    
    Returns:
        axes: [N, 3, 3] 旋转矩阵，每行是一个轴 (x_axis, y_axis, z_axis)
    """
    quats = F.normalize(quats, dim=-1)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    
    # 旋转矩阵的列向量
    # 第一列 (x轴)
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)
    
    # 第二列 (y轴)
    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)
    
    # 第三列 (z轴) - 这是高斯椭球的最短轴方向
    r02 = 2*(x*z + w*y)
    r12 = 2*(y*z - w*x)
    r22 = 1 - 2*(x*x + y*y)
    
    # 组装旋转矩阵
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=1)  # [N, 3, 3]
    
    return R


def compute_normal_alignment_loss(means, quats, normal_map, mask, viewmat, K, W, H, c2w):
    """
    法线对齐约束：让高斯点的最短轴方向与表面法线对齐
    
    这是非常强力的约束，确保高斯椭球"贴"在物体表面上
    
    Args:
        means: [N, 3] 高斯点3D位置（世界坐标系）
        quats: [N, 4] 高斯点四元数
        normal_map: [H, W, 3] 法线图（相机坐标系）
        mask: [H, W, 1] 物体mask
        viewmat: [4, 4] 相机view矩阵
        K: [3, 3] 相机内参
        W, H: 图像尺寸
        c2w: [4, 4] 相机到世界变换矩阵
    
    Returns:
        loss: 标量损失值
    """
    N = means.shape[0]
    device = means.device
    
    # Step 1: 将高斯点投影到图像平面
    means_homo = torch.cat([means, torch.ones(N, 1, device=device)], dim=-1)
    means_cam = (viewmat @ means_homo.T).T[:, :3]
    z_cam = means_cam[:, 2]
    
    means_2d = means_cam @ K.T
    u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)
    v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)
    
    # 过滤有效点
    valid_depth = z_cam > 0.01
    valid_u = (u >= 0) & (u < W - 1)
    valid_v = (v >= 0) & (v < H - 1)
    valid_proj = valid_depth & valid_u & valid_v
    
    if valid_proj.sum() < 10:
        return torch.tensor(0.0, device=device)
    
    # Step 2: 采样法线
    u_norm = (2.0 * u / (W - 1) - 1.0)
    v_norm = (2.0 * v / (H - 1) - 1.0)
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
    
    # normal_map: [H, W, 3] -> [1, 3, H, W]
    normal_4d = normal_map.permute(2, 0, 1).unsqueeze(0)
    mask_4d = mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
    
    # 采样每个点对应的法线和mask
    sampled_normal = F.grid_sample(normal_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    sampled_normal = sampled_normal.squeeze().T  # [N, 3]
    sampled_mask = F.grid_sample(mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
    
    # 归一化采样的法线
    sampled_normal = F.normalize(sampled_normal, dim=-1)
    
    # Step 3: 过滤有效点
    valid_mask = valid_proj & (sampled_mask > 0.5)
    normal_valid = torch.norm(sampled_normal, dim=-1) > 0.5
    valid_mask = valid_mask & normal_valid
    
    if valid_mask.sum() < 10:
        return torch.tensor(0.0, device=device)
    
    # Step 4: 从四元数提取高斯椭球的最短轴（z轴）
    # 这个轴应该与表面法线对齐
    R = quaternion_to_axes(quats)  # [N, 3, 3]
    z_axis_local = R[:, :, 2]  # [N, 3] 高斯椭球的z轴（最短轴方向）
    
    # Step 5: 将法线从相机坐标系转换到世界坐标系
    # 法线变换需要使用 (R^T)^(-1) = R^T 的转置，但对于正交矩阵，R^(-T) = R
    # 所以法线变换用 c2w 的旋转部分
    R_c2w = c2w[:3, :3]
    sampled_normal_world = (R_c2w @ sampled_normal[valid_mask].T).T  # [M, 3]
    sampled_normal_world = F.normalize(sampled_normal_world, dim=-1)
    
    # Step 6: 计算对齐loss
    # 我们希望 z_axis_local 与 sampled_normal_world 平行（可以同向或反向）
    # 使用 1 - |dot| 作为loss，这样同向和反向都是0 loss
    z_axis_valid = z_axis_local[valid_mask]
    dot_product = (z_axis_valid * sampled_normal_world).sum(dim=-1)  # [M]
    
    # 使用 1 - dot^2 可以让同向和反向都接近0
    # 或者使用 1 - |dot| 但要小心梯度
    alignment_loss = (1.0 - dot_product.abs()).mean()
    
    # 额外的惩罚：如果点积接近0（垂直），惩罚更大
    perpendicular_penalty = torch.exp(-4.0 * dot_product.abs()).mean() * 0.1
    
    return alignment_loss + perpendicular_penalty


def compute_depth_pull_loss(means, gt_depth, gt_mask, viewmat, K, W, H, c2w, strength=1.0):
    """
    深度拉扯约束：直接计算每个高斯点"应该在的正确位置"，然后约束点向那个位置移动
    
    这是更强力的版本：不仅惩罚深度差异，还计算正确的3D目标位置
    
    Args:
        means: [N, 3] 高斯点3D位置（世界坐标系）
        gt_depth: [H, W, 1] GT深度图
        gt_mask: [H, W, 1] 物体mask
        viewmat: [4, 4] 相机view矩阵
        K: [3, 3] 相机内参
        W, H: 图像尺寸
        c2w: [4, 4] 相机到世界变换矩阵
        strength: loss强度
    
    Returns:
        loss: 标量损失值
    """
    N = means.shape[0]
    device = means.device
    
    # Step 1: 将高斯点从世界坐标系变换到相机坐标系
    means_homo = torch.cat([means, torch.ones(N, 1, device=device)], dim=-1)  # [N, 4]
    means_cam = (viewmat @ means_homo.T).T[:, :3]  # [N, 3]
    
    z_cam = means_cam[:, 2]  # 高斯点在相机坐标系的深度
    
    # Step 2: 投影到图像平面
    means_2d = means_cam @ K.T  # [N, 3]
    u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)  # x / z
    v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)  # y / z
    
    # Step 3: 过滤有效点
    valid_depth = z_cam > 0.01
    valid_u = (u >= 0) & (u < W - 1)
    valid_v = (v >= 0) & (v < H - 1)
    valid_proj = valid_depth & valid_u & valid_v
    
    if valid_proj.sum() < 10:
        return torch.tensor(0.0, device=device)
    
    # Step 4: 采样GT深度
    u_norm = (2.0 * u / (W - 1) - 1.0)
    v_norm = (2.0 * v / (H - 1) - 1.0)
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
    
    gt_depth_4d = gt_depth.squeeze(-1).unsqueeze(0).unsqueeze(0)
    gt_mask_4d = gt_mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
    
    sampled_depth = F.grid_sample(gt_depth_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
    sampled_mask = F.grid_sample(gt_mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
    
    # Step 5: 只对mask内且投影有效的点计算loss
    valid_mask = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01)
    
    if valid_mask.sum() < 10:
        return torch.tensor(0.0, device=device)
    
    # Step 6: 计算目标位置（在相机坐标系中）
    # 目标深度 = GT深度，目标xy = 当前点的xy方向 * (目标深度 / 当前深度)
    # 这样保持射线方向，只改变沿射线的距离
    
    target_z = sampled_depth[valid_mask]
    current_z = z_cam[valid_mask]
    
    # 方法1：直接约束深度差（简单有效）
    depth_loss = torch.abs(current_z - target_z).mean()
    
    # 方法2：计算完整的3D目标位置并约束（更强力）
    # 目标点在相机坐标系中的位置
    scale_factor = target_z / (current_z + 1e-8)
    scale_factor = torch.clamp(scale_factor, 0.1, 10.0)  # 防止极端缩放
    
    means_cam_valid = means_cam[valid_mask]
    target_cam = means_cam_valid * scale_factor.unsqueeze(-1)
    
    # 转换回世界坐标系
    target_cam_homo = torch.cat([target_cam, torch.ones(target_cam.shape[0], 1, device=device)], dim=-1)
    target_world = (c2w @ target_cam_homo.T).T[:, :3]
    
    # 计算当前位置与目标位置的L2距离
    position_loss = torch.norm(means[valid_mask] - target_world, dim=-1).mean()
    
    return strength * (depth_loss + 0.5 * position_loss)


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
        print("Initializing Gaussian Ellipsoids (Anisotropic) with Normal-aligned Orientation...")
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
        
        # 计算像素级大小的 radii
        # 一个像素在3D空间的大小 ≈ depth / focal
        # 我们希望高斯点大约是 1-2 个像素大小
        avg_depth = d.mean().item()
        pixel_size_3d = avg_depth / self.focal  # 一个像素对应的3D尺寸
        target_radius = pixel_size_3d * 1.5  # 目标：约1.5个像素
        target_radii = math.log(target_radius)
        
        print(f"[*] Avg depth: {avg_depth:.3f}, Pixel size 3D: {pixel_size_3d:.6f}")
        print(f"[*] Target radius: {target_radius:.6f}, Target radii (log): {target_radii:.3f}")
        
        # 初始化 radii：像素级大小，扁平形状
        self.radii = torch.ones((N, 3), device=self.device) * target_radii  # x, y 轴
        self.radii[:, 2] = target_radii - 1.5  # z 轴更薄
        
        # 保存像素级 radii 上限供训练时使用
        self.pixel_radii_max = target_radii + 0.5  # 允许略大于1.5像素
        self.pixel_radii_min = target_radii - 3.0  # 最小约0.05像素 
        
        # ==============================
        # [核心改进] 用深度图法线初始化四元数
        # 这确保高斯椭球从一开始就"贴"在物体表面
        # ==============================
        print("[*] Computing surface normals from depth map...")
        
        # 计算完整的法线图（相机坐标系）
        normal_map_cam = depth_to_normal(depth, self.K, mask.float())  # [H, W, 3]
        
        # 提取每个有效点的法线
        # valid 是一个 [H, W] 的布尔mask
        normals_cam = normal_map_cam[valid]  # [N, 3] 相机坐标系法线
        
        # 将法线从相机坐标系转换到世界坐标系
        R_c2w = c2w[:3, :3]
        normals_world = (R_c2w @ normals_cam.T).T  # [N, 3] 世界坐标系法线
        
        # 归一化（处理可能的零法线）
        normal_norms = torch.norm(normals_world, dim=-1, keepdim=True)
        valid_normal = normal_norms.squeeze() > 0.1
        normals_world = F.normalize(normals_world, dim=-1)
        
        # 将法线转换为四元数
        self.quats = normal_to_quaternion(normals_world)
        
        # 对于法线无效的点，使用默认四元数（单位四元数）
        invalid_normal = ~valid_normal
        if invalid_normal.sum() > 0:
            self.quats[invalid_normal, 0] = 1.0
            self.quats[invalid_normal, 1:] = 0.0
        
        print(f"[*] Initialized {N} quaternions from surface normals ({valid_normal.sum().item()} valid normals)")
        
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
        print(f"[*] Initialized {N} Gaussians (Ellipsoids) with Normal-aligned Orientation.")

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

    def _densify_and_prune(self, frame_idx=0, prune_depth_outliers=True):
        """
        Densify高梯度点，并剔除深度异常值
        """
        with torch.no_grad():
            grads = self.grad_accum / self.denom
            grads[self.denom == 0] = 0.0
            to_clone = grads >= 0.0002  # 降低阈值，增加点密度以保留细节
            
            if to_clone.sum() > 0:
                self.means = torch.cat([self.means, self.means[to_clone]]).detach().requires_grad_(True)
                self.radii = torch.cat([self.radii, self.radii[to_clone]]).detach().requires_grad_(True)
                self.rgbs = torch.cat([self.rgbs, self.rgbs[to_clone]]).detach().requires_grad_(True)
                self.quats = torch.cat([self.quats, self.quats[to_clone]]).detach().requires_grad_(True)
                self.opacities = torch.cat([self.opacities, self.opacities[to_clone]]).detach().requires_grad_(True)
            
            # ==============================
            # [新增] 深度异常值剔除
            # 检测并删除严重偏离GT深度的高斯点
            # ==============================
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
        """
        检测深度异常值：投影到图像平面后，深度与GT深度差异超过阈值的点
        
        Args:
            frame_idx: 当前帧索引
            threshold: 深度差异阈值（相对误差）
        
        Returns:
            outlier_mask: [N] bool tensor，True表示是异常值
        """
        N = self.means.shape[0]
        device = self.device
        
        gt_depth = self.depths[frame_idx].unsqueeze(-1)  # [H, W, 1]
        gt_mask = self.gt_masks[frame_idx]  # [H, W, 1]
        c2w = self.c2ws[frame_idx]
        viewmat = torch.inverse(c2w)
        
        # 将高斯点变换到相机坐标系
        means_homo = torch.cat([self.means, torch.ones(N, 1, device=device)], dim=-1)
        means_cam = (viewmat @ means_homo.T).T[:, :3]
        z_cam = means_cam[:, 2]
        
        # 投影到图像平面
        means_2d = means_cam @ self.K.T
        u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)
        v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)
        
        # 过滤有效投影
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
        
        # 异常值判定：在mask内且投影有效，但深度误差超过阈值
        is_outlier = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01) & (rel_error > threshold)
        
        return is_outlier
    
    def _snap_points_to_depth_hard(self, frame_idx, remove_outliers=True, depth_tolerance=0.05):
        """
        [硬约束深度对齐] 强制将所有高斯点完全对齐到GT深度上
        
        这个函数会：
        1. 将所有有效点100%对齐到GT深度（不是部分移动）
        2. 删除投影到mask外的点（这些点在新视角会漂浮）
        3. 删除深度严重偏离的点
        
        Args:
            frame_idx: 当前帧索引
            remove_outliers: 是否删除mask外的点
            depth_tolerance: 允许的相对深度误差（超过这个会被删除）
        """
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
        
        # ==============================
        # Step 1: 100%对齐mask内的点到GT深度
        # ==============================
        snap_mask = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01)
        
        if snap_mask.sum() > 0:
            target_z = sampled_depth[snap_mask]
            current_z = z_cam[snap_mask]
            
            # 完全对齐，不做clamp（允许任意缩放）
            scale_factor = target_z / (current_z + 1e-8)
            
            # 计算新的相机坐标系位置（100%对齐）
            means_cam_snap = means_cam[snap_mask] * scale_factor.unsqueeze(-1)
            
            # 转换回世界坐标系
            means_cam_snap_homo = torch.cat([means_cam_snap, torch.ones(means_cam_snap.shape[0], 1, device=device)], dim=-1)
            means_world_snap = (c2w @ means_cam_snap_homo.T).T[:, :3]
            
            # 更新高斯点位置
            self.means.data[snap_mask] = means_world_snap
        
        # ==============================
        # Step 2: 删除mask外的点（这些点会在新视角漂浮）
        # ==============================
        if remove_outliers:
            # 识别需要删除的点：
            # 1. 投影到mask外的点
            # 2. 投影到图像外的点
            # 3. 深度为负的点
            outside_mask = valid_proj & (sampled_mask < 0.3)  # 在mask外
            invalid_points = (~valid_proj) | outside_mask
            
            # 保留有效点
            keep_mask = ~invalid_points
            
            if (~keep_mask).sum() > 0:
                n_removed = (~keep_mask).sum().item()
                self.means = self.means[keep_mask].detach().requires_grad_(True)
                self.radii = self.radii[keep_mask].detach().requires_grad_(True)
                self.rgbs = self.rgbs[keep_mask].detach().requires_grad_(True)
                self.quats = self.quats[keep_mask].detach().requires_grad_(True)
                self.opacities = self.opacities[keep_mask].detach().requires_grad_(True)
                
                # 重置梯度累积
                self.grad_accum = torch.zeros(self.means.shape[0], device=device)
                self.denom = torch.zeros(self.means.shape[0], device=device)
                
                print(f"[Hard Snap] Removed {n_removed} outlier points, {self.means.shape[0]} remaining")
                return True  # 需要重建optimizer
        
        return False  # 不需要重建optimizer
    
    def _enforce_normal_alignment(self, frame_idx, strength=0.5):
        """
        [每次迭代后调用] 强制高斯点的方向与表面法线对齐
        这是硬约束：直接修改四元数，不通过loss
        
        Args:
            frame_idx: 当前帧索引
            strength: 对齐强度 (0=不对齐, 1=完全对齐)
        """
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
        sampled_normal = sampled_normal.squeeze().T  # [N, 3]
        sampled_mask = F.grid_sample(mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        
        # 归一化法线
        sampled_normal = F.normalize(sampled_normal, dim=-1)
        
        # 过滤有效点
        valid_mask = valid_proj & (sampled_mask > 0.5)
        normal_valid = torch.norm(sampled_normal, dim=-1) > 0.5
        valid_mask = valid_mask & normal_valid
        
        if valid_mask.sum() < 10:
            return
        
        # 将法线从相机坐标系转换到世界坐标系
        R_c2w = c2w[:3, :3]
        normals_world = (R_c2w @ sampled_normal[valid_mask].T).T
        normals_world = F.normalize(normals_world, dim=-1)
        
        # 计算目标四元数
        target_quats = normal_to_quaternion(normals_world)
        
        # 使用SLERP进行平滑过渡，避免突变
        current_quats = self.quats.data[valid_mask]
        
        # 简化版SLERP：线性插值然后归一化
        # 确保四元数符号一致（选择最短路径）
        dot = (current_quats * target_quats).sum(dim=-1, keepdim=True)
        target_quats = torch.where(dot < 0, -target_quats, target_quats)
        
        # 线性插值
        new_quats = (1 - strength) * current_quats + strength * target_quats
        new_quats = F.normalize(new_quats, dim=-1)
        
        # 更新四元数
        self.quats.data[valid_mask] = new_quats

    def _enforce_depth_constraint(self, frame_idx):
        """
        [每次迭代后调用] 强制所有点贴合GT深度
        这是最严格的约束：直接修改点位置，不通过loss
        """
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
        
        # 只对mask内的有效点进行对齐
        snap_mask = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01)
        
        if snap_mask.sum() < 10:
            return
        
        # 计算目标位置：沿射线方向缩放到目标深度
        target_z = sampled_depth[snap_mask]
        current_z = z_cam[snap_mask]
        
        # 100%对齐
        scale_factor = target_z / (current_z + 1e-8)
        
        # 计算新的相机坐标系位置
        means_cam_snap = means_cam[snap_mask] * scale_factor.unsqueeze(-1)
        
        # 转换回世界坐标系
        means_cam_snap_homo = torch.cat([means_cam_snap, torch.ones(means_cam_snap.shape[0], 1, device=device)], dim=-1)
        means_world_snap = (c2w @ means_cam_snap_homo.T).T[:, :3]
        
        # 更新高斯点位置（直接赋值，不通过梯度）
        self.means.data[snap_mask] = means_world_snap

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



    def render_freewheel_video(self, frame_idx, video_length=60, scale_factor=0.8):
        save_path = os.path.join(self.dir_videos, f"frame_{frame_idx:03d}_novel.mp4")
        temp_path = os.path.join(self.dir_videos, f"frame_{frame_idx:03d}_novel_temp.mp4")
        # 先用 mp4v 生成临时文件，后面用 ffmpeg 转成 H.264
        writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.W, self.H))
        
        # 1. 确定物体中心
        if self.means.shape[0] > 0:
            object_center = torch.median(self.means.detach(), dim=0)[0]
        else:
            object_center = torch.zeros(3, device=self.device, dtype=torch.float32)
            
        c2w_curr = self.c2ws[frame_idx]
        cam_pos_curr = c2w_curr[:3, 3]
        
        # 2. 获取当前相机的基向量（直接从矩阵取，保证方向和当前帧一致）
        # c2w 列向量: 0:Right, 1:Up/Down, 2:Forward/Back
        current_right = F.normalize(c2w_curr[:3, 0], dim=0)
        current_up    = F.normalize(c2w_curr[:3, 1], dim=0)
        
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
                
                # 在相机平面内晃动
                # 使用当前相机的 Right 和 Up 向量进行偏移，确保相对运动自然
                offset_x = scale_factor * np.sin(t)
                offset_y = (scale_factor * 0.5) * np.sin(2 * t)
                
                pos_temp = cam_pos_curr + current_right * offset_x + current_up * offset_y
                
                # 投射回球面上，保持距离不变
                direction_to_center = F.normalize(object_center - pos_temp, dim=0) # 指向物体
                pos_new = object_center - direction_to_center * original_radius
                
                # 计算 LookAt (使用修复后的函数)
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
        
        # 用 ffmpeg 转码为 H.264，使浏览器可以预览
        import subprocess
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', temp_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', save_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_path)  # 删除临时文件
        
        print(f"[Novel View] Saved video to {save_path}")


    def train_frame(self, frame_idx, iterations):
        allow_densify = True if frame_idx == 0 else (frame_idx % 5 == 0)

        optimizer = optim.Adam([
            {'params': [self.means], 'lr': 0.00016},
            {'params': [self.quats], 'lr': 0.001},
            {'params': [self.rgbs], 'lr': 0.0025},
            {'params': [self.radii], 'lr': 0.001},  # 极低学习率，保持像素级大小
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
                
                # ==============================
                # [新增] 逐点深度约束 - 直接把每个高斯点拉到GT深度上
                # ==============================
                # 这比渲染深度loss更强力：直接约束每个点的3D位置
                per_point_depth_loss = compute_per_point_depth_loss(
                    self.means, gt_d, gt_mask, viewmat_gt, self.K, self.W, self.H
                )
                # 使用较大的权重确保高斯点贴合GT深度
                lambda_per_point = 1.0 if i < 500 else 0.5  # 前期强约束，后期适当放松
                loss += lambda_per_point * per_point_depth_loss
                
                # [更强力版本] 深度拉扯约束 - 计算目标位置并约束
                depth_pull_loss = compute_depth_pull_loss(
                    self.means, gt_d, gt_mask, viewmat_gt, self.K, self.W, self.H, c2w_curr, strength=0.5
                )
                loss += depth_pull_loss
                
                # ==============================
                # [新增] 法线对齐约束 - 让高斯椭球方向贴合表面法线
                # 这是确保高斯点方向正确的核心约束
                # ==============================
                # 从深度图计算法线（每帧只计算一次，缓存起来）
                if not hasattr(self, '_cached_normal_map') or self._cached_frame_idx != frame_idx:
                    self._cached_normal_map = depth_to_normal(gt_d.squeeze(-1), self.K, gt_mask.squeeze(-1))
                    self._cached_frame_idx = frame_idx
                
                normal_map = self._cached_normal_map
                
                # 计算法线对齐loss
                normal_alignment_loss = compute_normal_alignment_loss(
                    self.means, self.quats, normal_map, gt_mask, 
                    viewmat_gt, self.K, self.W, self.H, c2w_curr
                )
                
                # 法线约束权重：前期强约束，后期适当放松
                # 但始终保持较高权重确保方向正确
                lambda_normal = 2.0 if i < 500 else 1.0
                loss += lambda_normal * normal_alignment_loss
                
                # ==============================
                # [新增] 椭球形状约束 - 消除尖刺，保持合理形状
                # ==============================
                
                # 约束1: 保持扁平形状（z轴最薄）
                radii_z = self.radii[:, 2]  # z轴 radii
                radii_xy_min = torch.min(self.radii[:, 0], self.radii[:, 1])  # xy轴较小者
                margin = 1.0  # radii 空间，exp(-1) ≈ 0.37 的缩放差异
                flatness_violation = F.relu(radii_z - (radii_xy_min - margin))
                flatness_loss = flatness_violation.mean()
                
                # 约束2: 限制长宽比（防止尖刺）
                # 任意两个轴的比例不能超过 max_aspect_ratio
                max_aspect_ratio = 3.0  # 最大允许3:1的长宽比（在scale空间是exp(log_ratio)）
                max_log_ratio = math.log(max_aspect_ratio)  # 约 1.1
                
                radii_max = torch.max(self.radii, dim=1)[0]  # 每个点的最大radii
                radii_min = torch.min(self.radii, dim=1)[0]  # 每个点的最小radii
                log_ratio = radii_max - radii_min  # 对数空间的比例
                
                # 惩罚超过最大比例的情况
                aspect_violation = F.relu(log_ratio - max_log_ratio)
                aspect_loss = aspect_violation.mean()
                
                # 约束3: 限制radii的绝对范围（像素级大小）
                # 使用动态计算的像素级边界
                radii_min_bound = self.pixel_radii_min
                radii_max_bound = self.pixel_radii_max
                
                # 惩罚超出范围的情况（强力惩罚过大的高斯点）
                too_small = F.relu(radii_min_bound - self.radii)
                too_large = F.relu(self.radii - radii_max_bound)
                # 对过大的高斯点使用平方惩罚，更强力
                range_loss = too_small.mean() + (too_large ** 2).mean() * 10.0
                
                # 约束4: 各向同性正则化（轻微惩罚极端各向异性）
                # 鼓励椭球更接近圆盘形状，而不是细长条
                radii_std = torch.std(self.radii, dim=1)  # 每个点三个轴的标准差
                isotropy_loss = radii_std.mean()
                
                # 总形状约束（强力约束保持像素级大小）
                lambda_flatness = 1.0
                lambda_aspect = 3.0   # 强力限制长宽比
                lambda_range = 5.0    # 强力限制尺寸范围
                lambda_isotropy = 0.2
                
                shape_loss = (lambda_flatness * flatness_loss + 
                             lambda_aspect * aspect_loss + 
                             lambda_range * range_loss +
                             lambda_isotropy * isotropy_loss)
                loss += shape_loss

            # KNN Rigid Loss：保持相邻点之间的距离不变
            # 注意：当点被删除后，需要检查索引有效性
            if frame_idx > 0 and self.knn_rigid_indices is not None:
                n_prev = self.means_prev_all.shape[0]
                n_curr = self.means.shape[0]
                # 只有当当前点数>=之前点数时才计算（点可能被删除）
                if n_curr >= n_prev:
                    curr_rigid = self.means[:n_prev]
                    # 检查knn索引是否有效
                    max_idx = self.knn_rigid_indices.max().item()
                    if max_idx < n_prev:
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
            
            # ==============================
            # [核心硬约束] 每次迭代后强制约束
            # 1. 深度对齐：高斯点贴合GT深度
            # 2. 法线对齐：高斯椭球方向正确
            # 3. 形状约束：消除尖刺，保持合理形状
            # ==============================
            with torch.no_grad():
                self._enforce_depth_constraint(frame_idx)
                
                # 法线硬约束：每10次迭代强制对齐一次
                # strength=0.3 表示每次对齐30%，避免突变但确保收敛
                if i % 10 == 0:
                    self._enforce_normal_alignment(frame_idx, strength=0.3)
                
                # [硬约束] 强制限制radii范围（像素级大小）
                self.radii.data.clamp_(min=self.pixel_radii_min, max=self.pixel_radii_max)
                
                # [硬约束] 强制限制长宽比
                # 确保最大轴和最小轴的差距不超过 log(3) ≈ 1.1
                max_log_ratio = 1.1
                radii_max, max_indices = self.radii.data.max(dim=1)
                radii_min, min_indices = self.radii.data.min(dim=1)
                current_ratio = radii_max - radii_min
                
                # 如果比例超标，压缩到允许范围
                needs_fix = current_ratio > max_log_ratio
                
                if needs_fix.any():
                    # 计算需要调整的量（平分到两端）
                    excess = current_ratio[needs_fix] - max_log_ratio
                    adjustment = excess / 2.0
                    
                    # 获取需要修复的点的索引
                    fix_indices = torch.where(needs_fix)[0]
                    
                    # 批量调整：使用scatter_add
                    for j in range(len(fix_indices)):
                        idx = fix_indices[j].item()
                        ma = max_indices[idx].item()
                        mi = min_indices[idx].item()
                        adj = adjustment[j].item()
                        self.radii.data[idx, ma] -= adj
                        self.radii.data[idx, mi] += adj
            
            if allow_densify and i > 100 and i < iterations - 100 and i % 300 == 0:
                if self.means.grad is not None:
                    self.grad_accum += torch.norm(self.means.grad[:, :2], dim=-1)
                    self.denom += 1.0
                added = self._densify_and_prune(frame_idx=frame_idx, prune_depth_outliers=True)
                if added:
                    optimizer = optim.Adam([
                        {'params': [self.means], 'lr': 0.00016},
                        {'params': [self.quats], 'lr': 0.001},
                        {'params': [self.rgbs], 'lr': 0.0025},
                        {'params': [self.radii], 'lr': 0.001},  # 极低学习率，保持像素级大小
                        {'params': [self.opacities], 'lr': 0.05}
                    ], lr=0.001)
            
            # ==============================
            # [定期清理] 删除mask外的漂浮点
            # ==============================
            if i > 0 and i % 500 == 0:
                with torch.no_grad():
                    need_rebuild = self._snap_points_to_depth_hard(frame_idx, remove_outliers=True)
                    if need_rebuild:
                        optimizer = optim.Adam([
                            {'params': [self.means], 'lr': 0.00016},
                            {'params': [self.quats], 'lr': 0.001},
                            {'params': [self.rgbs], 'lr': 0.0025},
                            {'params': [self.radii], 'lr': 0.001},  # 极低学习率，保持像素级大小
                            {'params': [self.opacities], 'lr': 0.05}
                        ], lr=0.001)

        # ==============================
        # [最终清理] 训练结束前彻底清除所有漂浮点
        # ==============================
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
            
            self.save_params(t) # 这行该放哪？暂时未定
            
            img_pil = Image.fromarray(rgb_np)
            depth_pil = Image.fromarray(depth_np)
            frames.append(img_pil)
            depth_frames.append(depth_pil)
            
            self.means_prev_all = self.means.detach().clone()
            if t > 0: self.means_prev_2 = self.means_prev_all.clone()
            else: self.means_prev_2 = self.means.detach().clone()

            img_pil.save(f"{self.dir_images}/frame_{t:03d}.png")
            depth_pil.save(f"{self.dir_depths}/depth_{t:03d}.png")

            self.render_freewheel_video(t, video_length=60, scale_factor=0.8)
            
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