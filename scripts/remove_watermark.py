from moviepy.editor import VideoFileClip
import numpy as np

def mask_bottom_right(video_path, output_path, mask_w=150, mask_h=10):
    """
    将视频右下角指定区域变黑
    :param video_path: 原视频路径
    :param output_path: 输出视频路径
    :param mask_w: 遮罩宽度 (像素)
    :param mask_h: 遮罩高度 (像素)
    """
    
    # 1. 加载视频
    clip = VideoFileClip(video_path)
    
    # 2. 定义处理每一帧的函数
    def process_frame(frame):
        # 【关键修改】这里必须先复制一份，否则会报 read-only 错误
        frame = frame.copy()
        
        # frame 是一个 numpy 数组，形状为 (高度, 宽度, 颜色通道)
        height, width, _ = frame.shape
        
        # 计算右下角的坐标范围
        y_start = height - mask_h
        x_start = width - mask_w
        
        # 将该区域的像素设为全黑
        frame[y_start:height, x_start:width] = [0, 0, 0]
        
        return frame

    # 3. 将处理函数应用到视频流
    new_clip = clip.fl_image(process_frame)
    
    # 4. 导出视频 (保留音频)
    print(f"正在处理视频，将在右下角生成 {mask_w}x{mask_h} 的黑色遮罩...")
    new_clip.write_videofile(output_path, audio_codec='aac')
    
    # 释放资源
    clip.close()
    new_clip.close()

if __name__ == "__main__":
    # --- 请确认文件名是否正确 ---
    input_file = "sam2.mp4"       # 输入视频
    output_file = "sam2_.mp4"  # 输出视频
    
    # 执行处理
    mask_bottom_right(input_file, output_file, mask_w=300, mask_h=50)