#!/usr/bin/env python3
"""
Gaussian Splatting Scene Reconstruction

从单目视频重建动态场景的高斯点云表示

Usage:
    python main.py --dataset_path <path_to_dataset> --exp_name <name>

Example:
    python main.py \
        --dataset_path /root/autodl-tmp/shape-of-motion/datasets/breakdance-flare \
        --exp_name my_experiment \
        --use_sds
"""
import os
import argparse
import datetime

from core.gaussian_solver import GaussianSplattingSolver


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian Splatting Scene Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/root/autodl-tmp/shape-of-motion/datasets/breakdance-flare",
        help="Path to the dataset directory (contains images/, aligned_depth_anything/, masks/ subdirectories and droid_recon.npy)"
    )
    
    # 实验配置
    parser.add_argument(
        "--exp_name", 
        type=str, 
        default="breakdance-flare",
        help="Experiment name (used in output folder)"
    )
    parser.add_argument(
        "--output_root", 
        type=str, 
        default="results",
        help="Root directory for output files"
    )
    
    # 相机参数
    parser.add_argument(
        "--focal_ratio", 
        type=float, 
        default=0.8,
        help="Focal length ratio (focal = width * ratio) if not provided in data"
    )
    
    # SDS参数
    parser.add_argument(
        "--use_sds", 
        action="store_true",
        help="Enable Zero123 SDS guidance for novel view supervision"
    )
    
    # Mask参数
    parser.add_argument(
        "--use_mask",
        action="store_true",
        help="Enable mask-based training (default: False, train on whole image)"
    )

    args = parser.parse_args()

    # 生成输出目录名
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    folder_name = f"{current_time}_{args.exp_name}"
    full_output_dir = os.path.join(args.output_root, folder_name)

    # 创建求解器并运行
    solver = GaussianSplattingSolver(
        dataset_path=args.dataset_path, 
        output_dir=full_output_dir,
        focal_ratio=args.focal_ratio,
        use_sds=args.use_sds,
        use_mask=args.use_mask
    )
    solver.run()


if __name__ == "__main__":
    main()

