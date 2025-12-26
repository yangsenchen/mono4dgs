"""
Zero123 SDS引导模块
- CLIPCameraProjection: CLIP相机投影
- Zero123Guide: Zero123 SDS损失计算
"""
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class CLIPCameraProjection(nn.Module):
    """CLIP相机投影模块"""
    
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(772, 768) 
        self.norm = nn.Identity()

    def forward(self, x):
        return self.proj(x)


class Zero123Guide(nn.Module):
    """
    Zero123 SDS引导模块
    
    用于计算Score Distillation Sampling损失，引导3D高斯生成新视角纹理
    """
    
    def __init__(self, device, model_key="ashawkey/zero123-xl-diffusers"):
        super().__init__()
        self.device = device
        self.dtype = torch.float16
        
        print(f"[SDS] Initializing Zero123 Guidance...")
        print(f"[SDS] Loading components from: {model_key}...")

        try:
            self.vae = AutoencoderKL.from_pretrained(
                model_key, subfolder="vae", torch_dtype=self.dtype
            ).to(device)
            self.unet = UNet2DConditionModel.from_pretrained(
                model_key, subfolder="unet", torch_dtype=self.dtype
            ).to(device)
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_key, subfolder="image_encoder", torch_dtype=self.dtype
            ).to(device)
            self.scheduler = DDPMScheduler.from_pretrained(
                model_key, subfolder="scheduler"
            )
            self.cc_projection = CLIPCameraProjection().to(device, dtype=self.dtype)
            
            print("[SDS] Downloading camera projection weights (safetensors)...")
            cc_path = hf_hub_download(
                repo_id=model_key, 
                filename="diffusion_pytorch_model.safetensors", 
                subfolder="clip_camera_projection"
            )
            state_dict = load_file(cc_path)
            new_state_dict = {}
            for k, v in state_dict.items():
                if "proj" in k or "linear" in k:
                    new_state_dict["proj.weight"] = v if "weight" in k else new_state_dict.get("proj.weight")
                    new_state_dict["proj.bias"] = v if "bias" in k else new_state_dict.get("proj.bias")
                else:
                    new_state_dict[k] = v
            
            if new_state_dict.get("proj.weight") is not None:
                self.cc_projection.load_state_dict(new_state_dict, strict=True)
                print("[SDS] Camera projection loaded successfully.")
            else:
                raise RuntimeError("Weight mismatch in camera projection.")

        except Exception as e:
            print(f"[Error] Failed to load components: {e}")
            raise RuntimeError("Could not load Zero123 components.")

        gc.collect()
        torch.cuda.empty_cache()
        
        self.min_step = 0.02
        self.max_step = 0.98
        self.ref_embeddings = None
        self.null_embeddings = None
        self.ref_latents = None
        self.c2w_ref = None

    def get_cam_embeddings(self, elevation, azimuth, radius):
        """获取相机位姿嵌入"""
        zero_tensor = torch.zeros_like(radius)
        camera_pose = torch.stack([elevation, azimuth, radius, zero_tensor], dim=-1).to(self.device, dtype=self.dtype)
        if self.ref_embeddings is None:
             raise RuntimeError("Reference embeddings not initialized.")
        batch_size = camera_pose.shape[0]
        ref_emb_expanded = self.ref_embeddings.repeat(batch_size, 1)
        mlp_input = torch.cat([ref_emb_expanded, camera_pose], dim=-1)
        return self.cc_projection(mlp_input)

    @torch.no_grad()
    def prepare_condition(self, ref_image_tensor, c2w_ref):
        """
        准备条件输入
        
        Args:
            ref_image_tensor: [1, 3, 256, 256], Range [0, 1]
            c2w_ref: [4, 4] 参考视角的相机矩阵
        """
        # CLIP Branch
        ref_img_224 = F.interpolate(ref_image_tensor, (224, 224), mode='bilinear', align_corners=False)
        ref_img_norm_clip = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        )(ref_img_224)
        self.ref_embeddings = self.image_encoder(ref_img_norm_clip).image_embeds.to(dtype=self.dtype)
        
        # 生成 Unconditional (Null) Embeddings 用于 CFG
        self.null_embeddings = torch.zeros_like(self.ref_embeddings)

        # VAE Branch
        ref_img_256 = F.interpolate(ref_image_tensor, (256, 256), mode='bilinear', align_corners=False)
        ref_img_norm_vae = (ref_img_256 - 0.5) * 2
        ref_img_norm_vae = ref_img_norm_vae.to(dtype=self.dtype)
        self.ref_latents = self.vae.encode(ref_img_norm_vae).latent_dist.mode() * 0.18215
        
        self.c2w_ref = c2w_ref

    def compute_relative_pose(self, c2w_curr):
        """
        计算当前视角相对于参考视角的相对位姿
        
        Args:
            c2w_curr: [4, 4] 当前视角的相机矩阵
        
        Returns:
            d_elevation, d_azimuth, d_radius: 相对位姿参数
        """
        T_ref = self.c2w_ref[:3, 3]
        T_curr = c2w_curr[:3, 3]

        def cartesian_to_spherical(xyz):
            xy = torch.sqrt(xyz[0]**2 + xyz[2]**2)
            elevation = torch.atan2(xyz[1], xy)
            azimuth = torch.atan2(xyz[0], xyz[2])
            radius = torch.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
            return elevation, azimuth, radius

        el_ref, az_ref, r_ref = cartesian_to_spherical(T_ref)
        el_curr, az_curr, r_curr = cartesian_to_spherical(T_curr)
        
        d_azimuth = torch.rad2deg(az_curr - az_ref)
        d_elevation = torch.rad2deg(el_curr - el_ref)
        d_radius = torch.tensor(0.0, device=self.device)
        
        d_azimuth = (d_azimuth + 180) % 360 - 180
        return d_elevation.unsqueeze(0), d_azimuth.unsqueeze(0), d_radius.unsqueeze(0)

    def sds_loss(self, pred_rgb, relative_pose, guidance_scale=3.0):
        """
        计算SDS损失
        
        Args:
            pred_rgb: [1, 3, H, W] 预测的RGB图像
            relative_pose: (d_elevation, d_azimuth, d_radius) 相对位姿
            guidance_scale: CFG引导强度，通常设为 3.0 到 5.0
        
        Returns:
            loss: SDS损失值
        """
        pred_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
        vae_input = (pred_256 - 0.5) * 2
        vae_input = vae_input.to(dtype=self.dtype) 
        
        # 1. Encode Latents
        latents = self.vae.encode(vae_input).latent_dist.sample()
        latents = latents * 0.18215
        
        # 2. Add Noise
        noise = torch.randn_like(latents)
        t = torch.randint(
            int(self.min_step * self.scheduler.config.num_train_timesteps), 
            int(self.max_step * self.scheduler.config.num_train_timesteps), 
            [1], device=self.device
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        
        # 3. Prepare Batch for CFG (Cond + Uncond)
        latent_model_input = torch.cat([noisy_latents] * 2)
        latent_model_input = torch.cat([latent_model_input, torch.cat([self.ref_latents] * 2)], dim=1)
        
        d_el, d_az, d_r = relative_pose
        camera_embeddings = self.get_cam_embeddings(d_el, d_az, d_r)
        
        # 构建 Cond 和 Uncond 的 Embedding
        encoder_hidden_states = torch.cat([self.ref_embeddings, self.null_embeddings], dim=0).unsqueeze(1)
        camera_embeddings = torch.cat([camera_embeddings] * 2, dim=0)

        # 4. Predict Noise
        noise_pred = self.unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=encoder_hidden_states,
            class_labels=camera_embeddings
        ).sample

        # 5. Apply CFG
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # 6. Compute SDS Gradients
        w = 1 - (t / self.scheduler.config.num_train_timesteps)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        latents_float = latents.float()
        grad_float = grad.float()
        target = (latents_float - grad_float).detach()
        loss = 0.5 * F.mse_loss(latents_float, target, reduction="sum")
        
        return loss

