#!/usr/bin/env python3
"""
Flying Pixel 3D可视化检测脚本

该脚本将depth和rgb反投影到3D点云，用于检测是否存在flying pixel问题。
Flying pixels通常发生在物体边缘，由于深度估计不准确导致点云飘散。

Usage:
    python scripts/visualize_flying_pixels.py --data_path <path_to_npz> [--frame_idx 0]
"""
import os
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path

# 检测是否有显示器
HAS_DISPLAY = os.environ.get('DISPLAY') is not None

if HAS_DISPLAY:
    import open3d as o3d
else:
    print("Warning: No display detected, will save point cloud to file instead of interactive visualization")


def depth_to_pointcloud(depth, rgb, K, mask=None):
    """
    将深度图反投影到3D点云
    
    Args:
        depth: [H, W] 深度图
        rgb: [H, W, 3] RGB图像 (0-1范围)
        K: [3, 3] 相机内参
        mask: [H, W] 可选mask
    
    Returns:
        points: [N, 3] 点云坐标
        colors: [N, 3] 点云颜色
    """
    H, W = depth.shape
    
    # 提取相机内参
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # 创建像素坐标网格
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    
    # 反投影到3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 组合点云
    points = np.stack([x, y, z], axis=-1)  # [H, W, 3]
    colors = rgb  # [H, W, 3]
    
    # 展平
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    
    # 有效性过滤
    valid = (depth.flatten() > 0.01) & (depth.flatten() < 100.0)
    if mask is not None:
        valid = valid & (mask.flatten() > 0.5)
    
    return points[valid], colors[valid]


def compute_edge_mask(depth, threshold=0.1):
    """
    检测深度边缘（可能存在flying pixel的区域）
    
    Args:
        depth: [H, W] 深度图
        threshold: 深度变化阈值（相对于深度值的比例）
    
    Returns:
        edge_mask: [H, W] 边缘mask
    """
    # 计算深度梯度
    grad_x = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    grad_y = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
    
    # 相对梯度（相对于当前深度）
    rel_grad_x = grad_x / (np.abs(depth) + 1e-6)
    rel_grad_y = grad_y / (np.abs(depth) + 1e-6)
    
    # 边缘检测
    edge_mask = (rel_grad_x > threshold) | (rel_grad_y > threshold)
    
    return edge_mask


def detect_flying_pixels(depth, K, depth_threshold=0.1, neighbor_radius=2):
    """
    检测flying pixels
    
    Flying pixels特征：
    1. 位于深度边缘附近
    2. 与邻域深度有较大跳变
    
    Args:
        depth: [H, W] 深度图
        K: [3, 3] 相机内参
        depth_threshold: 深度跳变阈值
        neighbor_radius: 邻域半径
    
    Returns:
        flying_mask: [H, W] flying pixel mask
    """
    H, W = depth.shape
    flying_mask = np.zeros((H, W), dtype=bool)
    
    # 使用Laplacian检测深度不连续
    depth_blur = cv2.GaussianBlur(depth.astype(np.float32), (5, 5), 0)
    laplacian = cv2.Laplacian(depth_blur, cv2.CV_32F)
    
    # 边缘检测
    edge_mask = compute_edge_mask(depth, threshold=depth_threshold)
    
    # 膨胀边缘区域
    kernel = np.ones((neighbor_radius*2+1, neighbor_radius*2+1), np.uint8)
    edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1)
    
    # flying pixels = 边缘区域 + 高Laplacian响应
    laplacian_threshold = np.std(laplacian) * 2
    flying_mask = edge_dilated.astype(bool) & (np.abs(laplacian) > laplacian_threshold)
    
    return flying_mask, edge_mask


def visualize_pointcloud_interactive(points, colors, title="Point Cloud"):
    """使用Open3D进行交互式3D可视化"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(pcd)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)
    
    vis.run()
    vis.destroy_window()


def save_pointcloud(points, colors, output_path):
    """保存点云到PLY文件"""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to: {output_path}")


def render_pointcloud_matplotlib(points, colors, output_dir, num_views=6, point_size=0.5, max_points=50000):
    """
    使用matplotlib渲染点云的多视角图像（无需显示器）
    
    Args:
        points: [N, 3] 点云坐标
        colors: [N, 3] 点云颜色
        output_dir: 输出目录
        num_views: 视角数量
        point_size: 点大小
        max_points: 最大显示点数（防止太慢）
    """
    import matplotlib
    matplotlib.use('Agg')  # 无显示器模式
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 下采样以加速渲染
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    views_dir = os.path.join(output_dir, "multiview")
    os.makedirs(views_dir, exist_ok=True)
    
    # 定义视角 (elevation, azimuth)
    view_configs = [
        (0, 0, "front"),
        (0, 90, "right"),
        (0, -90, "left"),
        (90, 0, "top"),
        (30, 45, "oblique_1"),
        (30, -45, "oblique_2"),
    ]
    
    fig = plt.figure(figsize=(18, 12))
    
    for idx, (elev, azim, name) in enumerate(view_configs[:num_views]):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=point_size, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{name.upper()} (elev={elev}, azim={azim})')
        ax.set_facecolor((0.1, 0.1, 0.1))
    
    plt.tight_layout()
    mosaic_path = os.path.join(output_dir, "pointcloud_multiview.png")
    plt.savefig(mosaic_path, dpi=150, facecolor='black')
    plt.close()
    print(f"Multi-view mosaic saved to: {mosaic_path}")
    
    return views_dir


def render_orthographic_projections(points, colors, flying_mask_flat, output_dir, image_size=800):
    """
    渲染正交投影视图（更容易看到flying pixels的分布）
    
    这种方法不需要3D渲染，直接投影到2D
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    valid_indices = np.where((points[:, 2] > 0) & (points[:, 2] < 100))[0]
    pts = points[valid_indices]
    cols = colors[valid_indices]
    flying = flying_mask_flat[valid_indices] if flying_mask_flat is not None else None
    
    if len(pts) == 0:
        return
    
    output_path = os.path.join(output_dir, "orthographic_projections.png")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='black')
    
    # 定义投影视图 (axis1, axis2, title)
    projections = [
        (0, 1, "Front View (X-Y)"),   # 正面
        (2, 1, "Side View (Z-Y)"),     # 侧面 - 最容易看到flying pixels
        (0, 2, "Top View (X-Z)"),      # 俯视
    ]
    
    for idx, (ax1, ax2, title) in enumerate(projections):
        # 普通点云
        ax = axes[0, idx]
        ax.scatter(pts[:, ax1], pts[:, ax2], c=cols, s=0.5, alpha=0.6)
        ax.set_title(title, color='white', fontsize=12)
        ax.set_facecolor((0.05, 0.05, 0.1))
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        # Flying pixels高亮
        ax = axes[1, idx]
        ax.scatter(pts[:, ax1], pts[:, ax2], c=cols, s=0.3, alpha=0.3)
        if flying is not None:
            flying_pts = pts[flying]
            if len(flying_pts) > 0:
                ax.scatter(flying_pts[:, ax1], flying_pts[:, ax2], 
                          c='red', s=2, alpha=0.9, label='Flying Pixels')
        ax.set_title(f"{title} - Flying Pixels (RED)", color='white', fontsize=12)
        ax.set_facecolor((0.05, 0.05, 0.1))
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='black')
    plt.close()
    print(f"Orthographic projections saved to: {output_path}")


def create_depth_slice_visualization(points, colors, flying_mask_flat, output_dir, num_slices=5):
    """
    创建深度切片可视化，显示不同深度层的点云分布
    这有助于识别flying pixels在深度方向的分布
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    z = points[:, 2]
    valid = (z > 0.01) & (z < 100)
    pts = points[valid]
    cols = colors[valid]
    flying = flying_mask_flat[valid] if flying_mask_flat is not None else None
    
    if len(pts) == 0:
        return
    
    z = pts[:, 2]
    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min
    
    fig, axes = plt.subplots(1, num_slices, figsize=(4*num_slices, 4), facecolor='black')
    
    for i in range(num_slices):
        z_lo = z_min + (i / num_slices) * z_range
        z_hi = z_min + ((i + 1) / num_slices) * z_range
        
        mask = (z >= z_lo) & (z < z_hi)
        slice_pts = pts[mask]
        slice_cols = cols[mask]
        slice_flying = flying[mask] if flying is not None else None
        
        ax = axes[i]
        ax.scatter(slice_pts[:, 0], slice_pts[:, 1], c=slice_cols, s=0.5, alpha=0.6)
        
        if slice_flying is not None and slice_flying.sum() > 0:
            flying_pts = slice_pts[slice_flying]
            ax.scatter(flying_pts[:, 0], flying_pts[:, 1], c='red', s=2, alpha=0.9)
        
        ax.set_title(f"Depth: {z_lo:.2f} - {z_hi:.2f}\n({mask.sum()} pts, {slice_flying.sum() if slice_flying is not None else 0} flying)", 
                    color='white', fontsize=10)
        ax.set_facecolor((0.05, 0.05, 0.1))
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "depth_slices.png")
    plt.savefig(output_path, dpi=150, facecolor='black')
    plt.close()
    print(f"Depth slices visualization saved to: {output_path}")


def create_visualization_images(rgb, depth, edge_mask, flying_mask, output_dir):
    """创建可视化图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 原始RGB
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "rgb.png"), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    
    # 2. 深度图可视化
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(output_dir, "depth.png"), depth_colored)
    
    # 3. 边缘mask
    edge_vis = (edge_mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(output_dir, "edge_mask.png"), edge_vis)
    
    # 4. Flying pixel mask
    flying_vis = (flying_mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(output_dir, "flying_mask.png"), flying_vis)
    
    # 5. RGB叠加flying pixels（红色高亮）
    overlay = rgb_uint8.copy()
    overlay[flying_mask] = [255, 0, 0]  # 红色标记flying pixels
    overlay_blended = cv2.addWeighted(rgb_uint8, 0.6, overlay, 0.4, 0)
    cv2.imwrite(os.path.join(output_dir, "rgb_flying_overlay.png"), 
                cv2.cvtColor(overlay_blended, cv2.COLOR_RGB2BGR))
    
    # 6. 深度叠加边缘
    depth_with_edge = depth_colored.copy()
    depth_with_edge[edge_mask] = [0, 0, 255]  # 红色边缘
    cv2.imwrite(os.path.join(output_dir, "depth_edge_overlay.png"), depth_with_edge)
    
    print(f"Visualization images saved to: {output_dir}")


def analyze_flying_pixels(depth, flying_mask, K):
    """分析flying pixels的分布和特征"""
    H, W = depth.shape
    total_pixels = H * W
    flying_count = flying_mask.sum()
    flying_ratio = flying_count / total_pixels * 100
    
    print("\n" + "="*60)
    print("Flying Pixel Analysis")
    print("="*60)
    print(f"Image size: {W} x {H}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Flying pixels detected: {flying_count:,} ({flying_ratio:.2f}%)")
    
    if flying_count > 0:
        flying_depths = depth[flying_mask]
        print(f"\nFlying pixel depth statistics:")
        print(f"  - Min depth: {flying_depths.min():.3f}")
        print(f"  - Max depth: {flying_depths.max():.3f}")
        print(f"  - Mean depth: {flying_depths.mean():.3f}")
        print(f"  - Std depth: {flying_depths.std():.3f}")
    
    # 按区域分析（将图像分成3x3网格）
    print(f"\nFlying pixel distribution (3x3 grid):")
    grid_h, grid_w = H // 3, W // 3
    for i in range(3):
        row_str = ""
        for j in range(3):
            region = flying_mask[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            region_ratio = region.sum() / region.size * 100
            row_str += f"{region_ratio:5.1f}% | "
        print(f"  {row_str}")
    
    print("="*60)
    return flying_ratio


def main():
    parser = argparse.ArgumentParser(description="Flying Pixel 3D Visualization")
    parser.add_argument("--data_path", type=str, 
                        default="/root/autodl-tmp/mega-sam/outputs_cvd/breakdance-flare_sgd_cvd_hr.npz",
                        help="Path to npz data file")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Frame index to visualize")
    parser.add_argument("--output_dir", type=str, default="flying_pixel_analysis",
                        help="Output directory for visualization")
    parser.add_argument("--depth_threshold", type=float, default=0.15,
                        help="Depth discontinuity threshold for edge detection")
    parser.add_argument("--focal_ratio", type=float, default=0.8,
                        help="Focal length ratio if not in data")
    parser.add_argument("--show_all", action="store_true",
                        help="Show all points (not just flying pixels)")
    parser.add_argument("--highlight_flying", action="store_true",
                        help="Highlight flying pixels in red in the point cloud")
    args = parser.parse_args()
    
    print(f"Loading data from: {args.data_path}")
    data = np.load(args.data_path)
    
    # 提取数据
    images = data['images'].astype(np.float32) / 255.0  # [N, H, W, 3]
    depths = data['depths'].astype(np.float32)  # [N, H, W]
    
    num_frames, H, W, _ = images.shape
    print(f"Loaded {num_frames} frames, size {W}x{H}")
    
    # 相机内参
    if 'intrinsic' in data:
        K = data['intrinsic']
    else:
        focal = float(W) * args.focal_ratio
        K = np.array([
            [focal, 0, W / 2.0],
            [0, focal, H / 2.0],
            [0, 0, 1]
        ])
        print(f"Using estimated focal length: {focal:.2f}")
    
    # 选择帧
    frame_idx = min(args.frame_idx, num_frames - 1)
    print(f"\nAnalyzing frame {frame_idx}...")
    
    rgb = images[frame_idx]
    depth = depths[frame_idx]
    
    # 检测flying pixels
    flying_mask, edge_mask = detect_flying_pixels(depth, K, depth_threshold=args.depth_threshold)
    
    # 分析flying pixels
    flying_ratio = analyze_flying_pixels(depth, flying_mask, K)
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, f"frame_{frame_idx:03d}")
    
    # 保存可视化图像
    create_visualization_images(rgb, depth, edge_mask, flying_mask, output_dir)
    
    # 生成点云
    print("\nGenerating point cloud...")
    points, colors = depth_to_pointcloud(depth, rgb, K)
    print(f"Total points: {len(points):,}")
    
    if args.highlight_flying:
        # 重新生成点云，标记flying pixels
        flying_mask_flat = flying_mask.flatten()
        valid = (depth.flatten() > 0.01) & (depth.flatten() < 100.0)
        flying_in_valid = flying_mask_flat[valid]
        
        # 将flying pixels标记为红色
        colors_highlighted = colors.copy()
        colors_highlighted[flying_in_valid] = [1.0, 0.0, 0.0]
        colors = colors_highlighted
    
    # 保存点云
    ply_path = os.path.join(output_dir, "pointcloud.ply")
    try:
        import open3d as o3d
        save_pointcloud(points, colors, ply_path)
        
        # 只保存flying pixels的点云
        if flying_ratio > 0:
            flying_valid = flying_mask.flatten() & (depth.flatten() > 0.01) & (depth.flatten() < 100.0)
            flying_points = depth_to_pointcloud_raw(depth, K)[flying_valid]
            flying_colors = rgb.reshape(-1, 3)[flying_valid]
            
            ply_flying_path = os.path.join(output_dir, "flying_pixels.ply")
            save_pointcloud(flying_points, flying_colors, ply_flying_path)
        
        # 获取flying mask的flat版本（与点云对应）
        flying_flat = flying_mask.flatten()
        valid_mask = (depth.flatten() > 0.01) & (depth.flatten() < 100.0)
        flying_in_valid = flying_flat[valid_mask]
        
        # 生成多视角渲染（使用matplotlib，无需显示器）
        print("\nRendering multi-view images with matplotlib...")
        render_pointcloud_matplotlib(points, colors, output_dir, num_views=6)
        
        # 生成正交投影视图（侧视图最容易看到flying pixels）
        print("\nRendering orthographic projections...")
        render_orthographic_projections(points, colors, flying_in_valid, output_dir)
        
        # 生成深度切片可视化
        print("\nRendering depth slices...")
        create_depth_slice_visualization(points, colors, flying_in_valid, output_dir)
        
    except ImportError:
        print("Open3D not installed, skipping PLY export")
    
    # 交互式可视化
    if HAS_DISPLAY:
        print("\nLaunching interactive 3D visualization...")
        print("Controls: Left-click + drag to rotate, scroll to zoom")
        visualize_pointcloud_interactive(points, colors, f"Frame {frame_idx} - Flying Pixels Analysis")
    else:
        print("\nNo display available. Please view the saved files:")
        print(f"  - Point cloud: {ply_path}")
        print(f"  - Images: {output_dir}")
    
    # 多帧分析
    if num_frames > 1:
        print(f"\n\nAnalyzing all {num_frames} frames...")
        all_flying_ratios = []
        for i in range(num_frames):
            fm, _ = detect_flying_pixels(depths[i], K, depth_threshold=args.depth_threshold)
            ratio = fm.sum() / fm.size * 100
            all_flying_ratios.append(ratio)
        
        print("\nFlying pixel ratio per frame:")
        for i, ratio in enumerate(all_flying_ratios):
            bar = "█" * int(ratio * 2)
            print(f"  Frame {i:3d}: {ratio:5.2f}% {bar}")
        
        print(f"\nOverall statistics:")
        print(f"  - Mean: {np.mean(all_flying_ratios):.2f}%")
        print(f"  - Max:  {np.max(all_flying_ratios):.2f}% (frame {np.argmax(all_flying_ratios)})")
        print(f"  - Min:  {np.min(all_flying_ratios):.2f}% (frame {np.argmin(all_flying_ratios)})")


def depth_to_pointcloud_raw(depth, K):
    """仅返回点云坐标（用于flying pixel提取）"""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    z = depth
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


if __name__ == "__main__":
    main()

