import numpy as np
import cv2
import os

def create_video_for_sam2():
    # 1. 配置路径
    # 请确认这是你的 npz 文件路径
    npz_path = "/root/autodl-tmp/learn-genmojo/data/car-turn/MegaSAM_Outputs/car-turn_sgd_cvd_hr.npz"
    output_video_path = "images_for_sam2.mp4"
    fps = 24 # 帧率，SAM2 对帧率不敏感，24或30都可以
    
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    # 2. 获取图像数据
    if 'images' in data:
        images_np = data['images']
    elif 'image' in data:
        images_np = data['image']
    else:
        print("Error: No image data found in npz.")
        return

    num_frames = images_np.shape[0]
    H, W = images_np.shape[1:3]
    print(f"Found {num_frames} frames. Resolution: {W}x{H}")
    
    # 3. 初始化视频写入器
    # 使用 mp4v 编码生成 MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    
    print(f"Exporting video to {output_video_path} ...")

    for i in range(num_frames):
        img_frame = images_np[i]
        
        # 归一化处理 (Float 0-1 -> Int 0-255)
        if img_frame.max() <= 1.05:
            img_frame = (img_frame * 255).astype(np.uint8)
        else:
            img_frame = img_frame.astype(np.uint8)
            
        # 颜色空间转换: RGB (NPZ) -> BGR (OpenCV)
        # 这一点很重要，不然生成的视频颜色会反（人脸变蓝）
        img_bgr = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)
        
        video_writer.write(img_bgr)
        
        if i % 10 == 0:
            print(f"Processed frame {i}/{num_frames}", end='\r')

    video_writer.release()
    print(f"\nDone! Video saved to: {os.path.abspath(output_video_path)}")

if __name__ == "__main__":
    create_video_for_sam2()