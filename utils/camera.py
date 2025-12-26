"""
相机工具函数
- compute_lookat_c2w: LookAt 相机矩阵
- get_orbit_camera: 轨道相机
"""
import torch
import torch.nn.functional as F


def compute_lookat_c2w(camera_pos, target_pos, up_vector=None):
    """
    LookAt 函数 (适配 gsplat 设置：Z轴指向物体)
    
    Args:
        camera_pos: [3] 相机位置
        target_pos: [3] 目标位置
        up_vector: [3] 上方向向量，默认为 [0, 1, 0]
    
    Returns:
        c2w: [4, 4] 相机到世界变换矩阵
    """
    if up_vector is None:
        up_vector = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device, dtype=torch.float32)
    
    # Z轴指向物体 (Forward)
    z_axis = F.normalize(target_pos - camera_pos, dim=0) 
    
    # X轴 (Right)
    x_axis = -F.normalize(torch.cross(up_vector, z_axis, dim=0), dim=0)
    
    # Y轴 (Down/Up)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=0), dim=0)
    
    c2w = torch.eye(4, device=camera_pos.device, dtype=torch.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = camera_pos
    
    return c2w


def get_orbit_camera(c2w, angle_x, angle_y, center=None, device='cuda'):
    """
    轨道相机：围绕物体中心旋转相机
    
    Args:
        c2w: [4, 4] 当前相机到世界变换矩阵
        angle_x: 水平旋转角度（度）
        angle_y: 垂直旋转角度（度）
        center: [3] 旋转中心，默认为原点
        device: 设备
    
    Returns:
        new_c2w: [4, 4] 新的相机到世界变换矩阵
    """
    if center is None:
        center = torch.zeros(3, device=device)
    
    # 1. 计算当前位置和半径
    pos_curr = c2w[:3, 3]
    vec = pos_curr - center
    radius = torch.norm(vec) + 1e-7
    
    x, y, z = vec[0], vec[1], vec[2]
    
    # 2. 转球坐标
    cos_theta = torch.clamp(y / radius, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    phi = torch.atan2(z, x)
    
    # 3. 应用旋转
    theta_new = theta - torch.deg2rad(torch.tensor(angle_y, device=device))
    theta_new = torch.clamp(theta_new, 0.1, 3.0)
    phi_new = phi - torch.deg2rad(torch.tensor(angle_x, device=device))
    
    # 4. 转回笛卡尔坐标
    y_new = radius * torch.cos(theta_new)
    h = radius * torch.sin(theta_new)
    x_new = h * torch.cos(phi_new)
    z_new = h * torch.sin(phi_new)
    
    pos_new = torch.stack([x_new, y_new, z_new]) + center
    
    # 5. LookAt 矩阵构建
    # Forward (Z-axis): Points TO Object (OpenCV convention)
    z_axis = F.normalize(center - pos_new, dim=0) 
    
    # Global Up
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    # X轴强制取反 (解决镜像问题)
    x_axis = -F.normalize(torch.cross(world_up, z_axis), dim=0)
    
    # Y轴跟随变化
    y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=0)
    
    new_c2w = torch.eye(4, device=device)
    new_c2w[:3, 0] = x_axis
    new_c2w[:3, 1] = y_axis
    new_c2w[:3, 2] = z_axis
    new_c2w[:3, 3] = pos_new
    
    return new_c2w

