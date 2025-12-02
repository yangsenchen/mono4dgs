import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_all_curves(history, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.switch_backend('Agg')
    
    iters = history['iter']
    if len(iters) == 0: return

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f'Training Progress (Iter {iters[-1]})', fontsize=16)

    # 1. Total Loss
    axs[0, 0].plot(iters, history['total_loss'], 'k-', label='Total')
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].set_yscale('log')
    axs[0, 0].grid(True)

    # 2. RGB Loss
    axs[0, 1].plot(iters, history['loss_rgb'], 'r-', label='RGB L1')
    axs[0, 1].set_title('RGB Loss')
    axs[0, 1].grid(True)

    # 3. SSIM Loss (New!)
    if 'loss_ssim' in history and len(history['loss_ssim']) == len(iters):
        axs[0, 2].plot(iters, history['loss_ssim'], 'b-', label='SSIM (1-score)')
        axs[0, 2].set_title('Structure Loss')
        axs[0, 2].grid(True)

    # 4. Depth Loss
    if 'loss_depth' in history and len(history['loss_depth']) == len(iters):
        axs[1, 0].plot(iters, history['loss_depth'], 'g-', label='Depth L1')
        axs[1, 0].set_title('Depth Loss')
        axs[1, 0].set_yscale('log')
        axs[1, 0].grid(True)

    # 5. PSNR
    axs[1, 1].plot(iters, history['psnr'], 'c-', linewidth=2)
    axs[1, 1].set_title('PSNR')
    axs[1, 1].grid(True)
    
    # 6. Points
    axs[1, 2].plot(iters, history['num_points'], 'y-', linewidth=2)
    axs[1, 2].set_title('Num Points')
    axs[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_dashboard.png"), dpi=100)
    plt.close()

def save_gif(frames_list, path, fps=30):
    if len(frames_list) == 0: return
    frames = [Image.fromarray(f) if not isinstance(f, Image.Image) else f for f in frames_list]
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True, duration=1000/fps, loop=0)