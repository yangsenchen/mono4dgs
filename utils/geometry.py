"""
几何计算工具函数
- depth_to_normal: 从深度图计算表面法线
- normal_to_quaternion: 将法线向量转换为四元数
- quaternion_to_axes: 从四元数提取三个局部坐标轴
"""
import torch
import torch.nn.functional as F


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
    
    # 计算 x, y, z 方向的偏导数
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
    tu = torch.stack([dx_du, dy_du, dz_du], dim=-1)  # [H, W, 3]
    tv = torch.stack([dx_dv, dy_dv, dz_dv], dim=-1)  # [H, W, 3]
    
    # 法线 = tu × tv (叉积)
    normals = torch.cross(tu, tv, dim=-1)  # [H, W, 3]
    
    # 归一化
    norm_length = torch.norm(normals, dim=-1, keepdim=True) + 1e-8
    normals = normals / norm_length
    
    # 确保法线指向相机外
    flip_mask = (normals[..., 2:3] > 0).float()
    normals = normals * (1 - 2 * flip_mask)
    
    # 对无效区域设置为零法线
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
    ref = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, -1)
    
    # 使用 Rodrigues 旋转公式
    dot = (ref * normal).sum(dim=-1, keepdim=True)  # [N, 1]
    cross = torch.cross(ref, normal, dim=-1)  # [N, 3]
    
    # 处理平行情况
    cross_norm = torch.norm(cross, dim=-1, keepdim=True) + 1e-8
    axis = cross / cross_norm  # 旋转轴
    
    # 处理反平行情况 (dot ≈ -1，需要旋转180度)
    anti_parallel = dot.squeeze(-1) < -0.999
    if anti_parallel.any():
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

