"""
Loss函数集合
- compute_per_point_depth_loss: 逐点深度约束
- compute_normal_alignment_loss: 法线对齐约束
- compute_depth_pull_loss: 深度拉扯约束
- compute_behind_surface_loss: 背面惩罚约束
- compute_opacity_entropy_loss: 不透明度熵损失（清理半透明漂浮点）
- compute_depth_variance_loss: 深度方差损失（强制高斯点紧贴GT深度表面）
- ssim: SSIM相似度损失
"""
import math
import torch
import torch.nn.functional as F

from .geometry import quaternion_to_axes


def compute_per_point_depth_loss(means, gt_depth, gt_mask, viewmat, K, W, H, margin=0.1,
                                  invalid_depth_mask=None, soft_weight=0.3):
    """
    逐点深度约束：直接约束每个高斯点的深度贴合GT深度
    
    核心思路：
    1. 对于每个高斯点，投影到图像平面得到 (u, v)
    2. 查询该像素位置的GT深度 d_gt
    3. 约束高斯点的相机坐标系深度 z 等于 d_gt
    
    对于 flying pixel 被移除后传播填充的区域，使用软权重监督
    
    Args:
        means: [N, 3] 高斯点3D位置（世界坐标系）
        gt_depth: [H, W, 1] GT深度图
        gt_mask: [H, W, 1] 物体mask
        viewmat: [4, 4] 相机view矩阵
        K: [3, 3] 相机内参
        W, H: 图像尺寸
        margin: 只对mask内部的点施加约束
        invalid_depth_mask: [H, W] 无效深度区域mask（传播填充的区域）
        soft_weight: 传播区域的软权重（0-1）
    
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
    
    # 如果有无效深度mask，对传播区域使用软权重
    if invalid_depth_mask is not None:
        # 采样无效深度mask
        invalid_mask_4d = invalid_depth_mask.float().unsqueeze(0).unsqueeze(0)
        sampled_invalid = F.grid_sample(invalid_mask_4d, grid, mode='nearest', 
                                         padding_mode='zeros', align_corners=True)
        sampled_invalid = sampled_invalid.squeeze()  # [N]
        
        # 计算每个点的权重：有效区域=1.0，传播区域=soft_weight
        point_weights = torch.where(sampled_invalid[valid_mask] > 0.5,
                                    torch.tensor(soft_weight, device=device),
                                    torch.tensor(1.0, device=device))
        
        # 加权平均
        loss = (z_diff * point_weights).sum() / (point_weights.sum() + 1e-8)
    else:
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


def compute_behind_surface_loss(means, opacities, gt_depth, gt_mask, K, c2w, 
                                 margin=0.05, lambda_scale=1.0):
    """
    惩罚位于GT深度表面之后的高斯点的不透明度。
    
    核心思想：与其每隔500步去"移动"或"删除"背面的点，不如每一步都惩罚
    那些跑到了深度图后面的点的不透明度（Opacity）。
    
    Args:
        means: [N, 3] 世界坐标系下的高斯点中心
        opacities: [N] 原始不透明度参数 (未经过sigmoid)
        gt_depth: [H, W, 1] 真实深度图
        gt_mask: [H, W, 1] 前景 Mask
        K: [3, 3] 相机内参
        c2w: [4, 4] 相机到世界变换矩阵
        margin: 容差，允许高斯点在表面后方的一小段距离内 (例如 5cm)
        lambda_scale: loss 权重
    
    Returns:
        loss: 标量损失值
    """
    device = means.device
    H, W = gt_depth.shape[0], gt_depth.shape[1]
    gt_depth_2d = gt_depth.squeeze(-1) if gt_depth.dim() == 3 else gt_depth
    gt_mask_2d = gt_mask.squeeze(-1) if gt_mask.dim() == 3 else gt_mask
    
    # 1. 将高斯点转换到相机坐标系
    viewmat = torch.inverse(c2w)
    means_homo = torch.cat([means, torch.ones(means.shape[0], 1, device=device)], dim=-1)
    means_cam = (viewmat @ means_homo.T).T[:, :3]
    z_cam = means_cam[:, 2]
    
    # 2. 投影到图像平面
    means_2d = means_cam @ K.T
    z_safe = z_cam.clamp(min=1e-5)
    u = means_2d[:, 0] / z_safe
    v = means_2d[:, 1] / z_safe
    
    # 3. 筛选在图像范围内的点
    valid_u = (u >= 0) & (u < W - 1)
    valid_v = (v >= 0) & (v < H - 1)
    valid_depth = z_cam > 0.1
    valid_proj = valid_u & valid_v & valid_depth
    
    if valid_proj.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 4. 采样对应位置的 GT 深度
    # 归一化到 [-1, 1] 用于 grid_sample
    u_norm = (2.0 * u[valid_proj] / (W - 1) - 1.0)
    v_norm = (2.0 * v[valid_proj] / (H - 1) - 1.0)
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N_valid, 2]
    
    gt_depth_sample = F.grid_sample(
        gt_depth_2d.unsqueeze(0).unsqueeze(0), 
        grid, 
        mode='nearest', 
        padding_mode='zeros', 
        align_corners=True
    ).squeeze()
    
    gt_mask_sample = F.grid_sample(
        gt_mask_2d.float().unsqueeze(0).unsqueeze(0), 
        grid, 
        mode='nearest', 
        padding_mode='zeros', 
        align_corners=True
    ).squeeze()

    # 5. 找到位于表面"后面"的点
    # 条件：不仅要被Mask覆盖，且深度大于 GT深度 + margin
    z_points = z_cam[valid_proj]
    
    # 这里非常关键：
    # 我们只惩罚那些 z > gt + margin 的点。
    # margin 建议设为 0.05 (5cm) 或者根据场景尺度调整
    is_behind = (z_points > (gt_depth_sample + margin)) & (gt_mask_sample > 0.5) & (gt_depth_sample > 0.01)
    
    if is_behind.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    # 6. 计算 Loss
    # 我们希望这些点的 opacity 趋近于 0
    # opacities 是未激活的 logits，所以使用 sigmoid
    behind_opacities = torch.sigmoid(opacities[valid_proj][is_behind])
    
    # 惩罚项：直接让 opacity 变小
    # 可以加一个权重的 scaling，比如越远惩罚越重
    diff = z_points[is_behind] - gt_depth_sample[is_behind]
    # clamp diff 到最小 1.0，这样即使 diff 很小也有惩罚
    weighted_loss = behind_opacities * torch.clamp(diff, min=1.0)
    
    return lambda_scale * weighted_loss.mean()


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


def compute_opacity_entropy_loss(opacities, eps=1e-6):
    """
    熵损失（Entropy Loss / Binary Opacity）：强制不透明度二值化
    
    原理：强制所有高斯点的不透明度（Opacity, α）要么接近 0（完全消失），
    要么接近 1（完全不透明）。不允许存在半透明的"幽灵点"。
    
    公式：L_entropy = - Σ (α_i * log(α_i) + (1-α_i) * log(1-α_i))
    
    当 α 接近 0 或 1 时，熵趋近于 0（最小化）
    当 α = 0.5 时，熵最大（被惩罚）
    
    Args:
        opacities: [N] 原始不透明度参数 (未经过sigmoid)
        eps: 数值稳定性的小常数
    
    Returns:
        loss: 标量损失值（越小越好，表示opacity越二值化）
    """
    # 将 logits 转换为概率
    alpha = torch.sigmoid(opacities)
    
    # 裁剪以避免 log(0)
    alpha = torch.clamp(alpha, eps, 1.0 - eps)
    
    # 计算二值熵: H = - (α * log(α) + (1-α) * log(1-α))
    # 我们希望最小化熵，即让 α 趋向于 0 或 1
    entropy = - (alpha * torch.log(alpha) + (1.0 - alpha) * torch.log(1.0 - alpha))
    
    return entropy.mean()


def compute_depth_variance_loss(render_d, render_d2, gt_d, mask, render_alpha=None, alpha_threshold=0.1):
    """
    深度方差损失（Depth Variance Loss）：强制所有贡献颜色的高斯点紧贴GT深度表面
    
    原理：普通深度Loss只约束加权平均值，就像全班平均分60分可以是所有人都考60分，
    也可以是一半人考0分一半人考120分。方差损失强制所有点都必须紧紧挨着D_gt。
    
    公式：L_var = Σ ω_i * (t_i - D_gt)^2
    
    通过渲染技巧计算：
    - render_d = Σ ω_i * t_i  (深度的加权平均)
    - render_d2 = Σ ω_i * t_i^2  (深度平方的加权平均)
    - 方差 = E[X^2] - E[X]^2 = render_d2 - render_d^2
    
    但我们想要的是相对于GT的方差，而不是相对于均值的方差：
    - L_var = Σ ω_i * (t_i - D_gt)^2
           = Σ ω_i * t_i^2 - 2*D_gt * Σ ω_i * t_i + D_gt^2 * Σ ω_i
           = render_d2 - 2*D_gt*render_d + D_gt^2 * render_alpha
    
    Args:
        render_d: [H, W, 1] 渲染的深度图 (Σ ω_i * t_i)
        render_d2: [H, W, 1] 渲染的深度平方图 (Σ ω_i * t_i^2)
        gt_d: [H, W, 1] GT深度图
        mask: [H, W, 1] 前景mask
        render_alpha: [H, W, 1] 渲染的alpha图 (Σ ω_i)，如果为None则假设为1
        alpha_threshold: 只在alpha大于此阈值的像素上计算loss
    
    Returns:
        loss: 标量损失值
    """
    # 确保形状一致
    if render_d.dim() == 2:
        render_d = render_d.unsqueeze(-1)
    if render_d2.dim() == 2:
        render_d2 = render_d2.unsqueeze(-1)
    if gt_d.dim() == 2:
        gt_d = gt_d.unsqueeze(-1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    
    # 计算有效mask
    valid_mask = (mask > 0.5) & (gt_d > 0.01)
    if render_alpha is not None:
        if render_alpha.dim() == 2:
            render_alpha = render_alpha.unsqueeze(-1)
        valid_mask = valid_mask & (render_alpha > alpha_threshold)
        # 计算相对于GT的方差: E[(t - D_gt)^2] = E[t^2] - 2*D_gt*E[t] + D_gt^2
        # 其中 E[t] = render_d / alpha, E[t^2] = render_d2 / alpha
        # 但由于我们用的是加权和而不是加权平均，直接用：
        # variance = render_d2 - 2*gt_d*render_d + gt_d^2 * render_alpha
        variance = render_d2 - 2.0 * gt_d * render_d + gt_d * gt_d * render_alpha
    else:
        # 如果没有alpha，假设alpha=1，简化为：
        # variance = render_d2 - 2*gt_d*render_d + gt_d^2
        variance = render_d2 - 2.0 * gt_d * render_d + gt_d * gt_d
    
    # 方差应该是非负的，但由于数值误差可能出现小的负值
    variance = F.relu(variance)
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=render_d.device, requires_grad=True)
    
    # 在有效区域内取平均
    loss = (variance * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
    
    return loss

