import numpy as np

file_name = '/root/autodl-tmp/learn-genmojo/data/car-turn/MegaSAM_Outputs/car-turn_sgd_cvd_hr.npz'

# print("文件中的数组名称 (keys):", data.files)

with np.load(file_name) as data:
    # 查看 cam_c2w
    cam_c2w_array = data['cam_c2w']
    print(f"\n数组 'cam_c2w' 形状: {cam_c2w_array.shape}")
    print(f"第一个 4x4 矩阵:\n{cam_c2w_array[0]}")

    # 查看 images
    images_array = data['images']
    print(f"\n数组 'images' 形状: {images_array.shape}")
    print(f"第一张图像的前 2x2 像素值:\n{images_array[0, :2, :2, :]}")


    d_array = data['depths']
    print(f"\n数组 'd' 形状: {d_array.shape}")
    # print(f"第一张图像的前 2x2 像素值:\n{d_array[0, :2, :2, :]}")