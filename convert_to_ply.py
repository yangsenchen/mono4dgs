import torch
import numpy as np
import os
import argparse
from plyfile import PlyData, PlyElement

def save_ply_strict_padding(pt_path, output_path):
    print(f"Exporting (Strict Format): {pt_path} -> {output_path}")
    
    # 1. 加载数据
    data = torch.load(pt_path, map_location="cpu")
    
    xyz = data['means'].numpy()
    opacities = data['opacities'].numpy()
    scale = data['radii'].numpy()
    rotation = data['quats'].numpy() # (N, 4)
    
    # 2. 颜色转换 (RGB -> SH DC)
    # 你的模型只有 RGB (0阶 SH)，我们计算 f_dc
    rgb = torch.sigmoid(data['rgbs']).numpy()
    SH_C0 = 0.28209479177387814
    f_dc = (rgb - 0.5) / SH_C0

    # 3. 构建严格的 Header 字段列表 (完全匹配文档顺序)
    # Header Order: x,y,z -> scale -> rot -> opacity -> f_dc -> f_rest
    dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ('opacity', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    
    # [关键修复] 添加 f_rest_0 到 f_rest_44 (共 45 个)
    # 虽然你没训练这些，但必须占位，否则 Viewer 会报错或显示蓝色
    for i in range(45):
        dtype_list.append((f'f_rest_{i}', 'f4'))

    # 4. 创建结构化数组
    elements = np.empty(xyz.shape[0], dtype=dtype_list)
    
    # 5. 填充数据
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    
    elements['scale_0'] = scale[:, 0]
    elements['scale_1'] = scale[:, 1]
    elements['scale_2'] = scale[:, 2]
    
    elements['rot_0'] = rotation[:, 0]
    elements['rot_1'] = rotation[:, 1]
    elements['rot_2'] = rotation[:, 2]
    elements['rot_3'] = rotation[:, 3]
    
    elements['opacity'] = opacities
    
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    
    # [关键修复] 填充 f_rest 为 0
    # 这样 Viewer 认为这是一个标准的 3 阶 SH 文件，但高阶颜色为黑色（无视视角变化）
    for i in range(45):
        elements[f'f_rest_{i}'] = np.zeros(xyz.shape[0], dtype=np.float32)
    
    # 6. 写入 PLY
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Saved strict PLY to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="params 文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="ply 输出路径")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".pt")])
    
    for f in files:
        pt_path = os.path.join(args.input_dir, f)
        ply_path = os.path.join(args.output_dir, f.replace(".pt", ".ply"))
        save_ply_strict_padding(pt_path, ply_path)