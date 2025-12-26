"""
图像处理工具函数
- crop_image_by_mask: 非微分裁剪（用于参考图）
- crop_and_resize_differentiable: 微分裁剪（用于SDS）
"""
import torch
import torch.nn.functional as F


def crop_image_by_mask(image_tensor, mask_tensor, target_size=256, padding=0.1):
    """
    非微分裁剪，仅用于处理参考图 Condition。
    确保 Zero123 看到的物体是居中且放大的。
    
    Args:
        image_tensor: [C, H, W] 图像张量
        mask_tensor: [1, H, W] mask张量
        target_size: 输出尺寸
        padding: 边界padding比例
    
    Returns:
        crop_resized: [C, target_size, target_size] 裁剪后的图像
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
    
    # Resize 到 target_size x target_size
    crop_resized = F.interpolate(
        crop.unsqueeze(0), 
        (target_size, target_size), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    return crop_resized


def crop_and_resize_differentiable(img_tensor, mask_tensor, target_size=256, padding=0.1):
    """
    微分裁剪，用于 SDS 过程中对渲染出的预测图进行裁剪
    
    Args:
        img_tensor: [1, 3, H, W] 图像张量
        mask_tensor: [1, 1, H, W] mask张量
        target_size: 输出尺寸
        padding: 边界padding比例
    
    Returns:
        cropped_img: [1, 3, target_size, target_size] 裁剪后的图像
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

