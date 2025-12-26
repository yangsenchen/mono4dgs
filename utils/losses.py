"""
Loss函数集合
- compute_per_point_depth_loss: 逐点深度约束
- compute_normal_alignment_loss: 法线对齐约束
- compute_depth_pull_loss: 深度拉扯约束
- ssim: SSIM相似度损失
"""
import math
import torch
import torch.nn.functional as F

from .geometry import quaternion_to_axes


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
    
    # Step 4: 采样GT深度（双线性插值）
    u_norm = (2.0 * u / (W - 1) - 1.0)
    v_norm = (2.0 * v / (H - 1) - 1.0)
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
    
    gt_depth_4d = gt_depth.squeeze(-1).unsqueeze(0).unsqueeze(0)
    gt_mask_4d = gt_mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
    
    sampled_depth = F.grid_sample(gt_depth_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    sampled_mask = F.grid_sample(gt_mask_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    sampled_depth = sampled_depth.squeeze()  # [N]
    sampled_mask = sampled_mask.squeeze()    # [N]
    
    # Step 5: 只对mask内且投影有效的点计算loss
    valid_mask = valid_proj & (sampled_mask > 0.5) & (sampled_depth > 0.01)
    
    if valid_mask.sum() < 10:
        return torch.tensor(0.0, device=device)
    
    # Step 6: 计算深度差异loss（L1）
    z_diff = torch.abs(z_cam[valid_mask] - sampled_depth[valid_mask])
    loss = z_diff.mean()
    
    return loss


def compute_normal_alignment_loss(means, quats, normal_map, mask, viewmat, K, W, H, c2w):
    """
    法线对齐约束：让高斯点的最短轴方向与表面法线对齐
    
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
    
    normal_4d = normal_map.permute(2, 0, 1).unsqueeze(0)
    mask_4d = mask.squeeze(-1).unsqueeze(0).unsqueeze(0)
    
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
    R = quaternion_to_axes(quats)  # [N, 3, 3]
    z_axis_local = R[:, :, 2]  # [N, 3] 高斯椭球的z轴（最短轴方向）
    
    # Step 5: 将法线从相机坐标系转换到世界坐标系
    R_c2w = c2w[:3, :3]
    sampled_normal_world = (R_c2w @ sampled_normal[valid_mask].T).T  # [M, 3]
    sampled_normal_world = F.normalize(sampled_normal_world, dim=-1)
    
    # Step 6: 计算对齐loss
    z_axis_valid = z_axis_local[valid_mask]
    dot_product = (z_axis_valid * sampled_normal_world).sum(dim=-1)  # [M]
    
    # 使用 1 - |dot| 作为loss
    alignment_loss = (1.0 - dot_product.abs()).mean()
    
    # 额外的惩罚：如果点积接近0（垂直），惩罚更大
    perpendicular_penalty = torch.exp(-4.0 * dot_product.abs()).mean() * 0.1
    
    return alignment_loss + perpendicular_penalty


def compute_depth_pull_loss(means, gt_depth, gt_mask, viewmat, K, W, H, c2w, strength=1.0):
    """
    深度拉扯约束：直接计算每个高斯点"应该在的正确位置"，然后约束点向那个位置移动
    
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
    means_homo = torch.cat([means, torch.ones(N, 1, device=device)], dim=-1)
    means_cam = (viewmat @ means_homo.T).T[:, :3]
    
    z_cam = means_cam[:, 2]
    
    # Step 2: 投影到图像平面
    means_2d = means_cam @ K.T
    u = means_2d[:, 0] / (means_2d[:, 2] + 1e-8)
    v = means_2d[:, 1] / (means_2d[:, 2] + 1e-8)
    
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
    
    # Step 6: 计算目标位置
    target_z = sampled_depth[valid_mask]
    current_z = z_cam[valid_mask]
    
    # 方法1：直接约束深度差
    depth_loss = torch.abs(current_z - target_z).mean()
    
    # 方法2：计算完整的3D目标位置并约束
    scale_factor = target_z / (current_z + 1e-8)
    scale_factor = torch.clamp(scale_factor, 0.1, 10.0)
    
    means_cam_valid = means_cam[valid_mask]
    target_cam = means_cam_valid * scale_factor.unsqueeze(-1)
    
    # 转换回世界坐标系
    target_cam_homo = torch.cat([target_cam, torch.ones(target_cam.shape[0], 1, device=device)], dim=-1)
    target_world = (c2w @ target_cam_homo.T).T[:, :3]
    
    # 计算当前位置与目标位置的L2距离
    position_loss = torch.norm(means[valid_mask] - target_world, dim=-1).mean()
    
    return strength * (depth_loss + 0.5 * position_loss)


def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算两张图像的SSIM相似度
    
    Args:
        img1: [B, C, H, W] 图像1
        img2: [B, C, H, W] 图像2
        window_size: 窗口大小
        size_average: 是否对结果取平均
    
    Returns:
        ssim_value: SSIM值
    """
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
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
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

