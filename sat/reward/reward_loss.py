from typing import List, Optional, Union
import os
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
import math

from sgm.modules.diffusionmodules.loss import VideoDiffusionLoss
from .reward import DiffRewardModel

# import rearrange
from einops import rearrange
import random
from sat import mpu

def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
    os.makedirs(save_path, exist_ok=True)

    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

class VidDiffwithRMLoss(VideoDiffusionLoss):
    def __init__(self, reward_type=None, segments=None, step_thresold=None, 
                 modulating_reward=None, tracking_strategy=None, reward_normalization=None, 
                 positive_reward=None, reward_cfg=None, partial=1.0, **kwargs):
        super().__init__(**kwargs)
        self.rewarder = DiffRewardModel(reward_type = reward_type,
                 reward_cfg = reward_cfg,
                 segments = segments,
                 step_thresold = step_thresold,
                 modulating_reward = modulating_reward,
                 tracking_strategy = tracking_strategy,
                 reward_normalization = reward_normalization,
                 positive_reward = positive_reward)
        
        self.partial = 1.0

    def __call__(self, network, denoiser, sampler, conditioner, autoencoder, input, batch, scale_factor):
        # prepare    
        device = input.device
        network.to("cpu")
        conditioner.to("cpu")
        autoencoder.to("cpu")
        torch.cuda.empty_cache()

        # process input
        noise = torch.randn_like(input)
        print(noise.device)
        print(input.dtype)
        print(noise.dtype)
        mp_size = mpu.get_model_parallel_world_size()
        if mp_size > 1:
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())

        # process condition
        print(f"Pre conditioner Allocated memory: {torch.cuda.memory_allocated() / (1024**2)} MB")
        for name, param in conditioner.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        conditioner.to(device)
        with torch.no_grad():
            cond, ucond = conditioner.get_unconditional_conditioning(batch, batch, force_uc_zero_embeddings = ["txt"])
        conditioner.to("cpu")
        torch.cuda.empty_cache()
        print(f"Pre diffuson Allocated memory: {torch.cuda.memory_allocated() / (1024**2)} MB")
        
        # denoising and sampling
        for name, param in network.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        network.to(device)
        scale = None
        scale_emb = None
        denoiser_fn = lambda input, sigma, c, **addtional_model_inputs: denoiser(
            network, input, sigma, c, concat_images=None, **addtional_model_inputs
        )
        with torch.no_grad():#should moidify when training
            samples_z = sampler(denoiser_fn, noise, cond, uc=ucond, scale=scale, scale_emb=scale_emb).to(input.dtype)
        network.to("cpu")
        torch.cuda.empty_cache()
        # print(samples_z.shape)
        # assert False
        print(samples_z.dtype)
        print(f"Pre decoding Allocated memory: {torch.cuda.memory_allocated() / (1024**2)} MB")

        # decoding
        for name, param in autoencoder.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        T = samples_z.shape[1]
        samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
        autoencoder.to(device)
        latent = 1.0 / scale_factor * samples_z
        recons = []
        loop_num = (T - 1) // 2
        for i in range(loop_num):
            if i == 0:
                start_frame, end_frame = 0, 3
            else:
                start_frame, end_frame = i * 2 + 1, i * 2 + 3
            if i == loop_num - 1:
                clear_fake_cp_cache = True
            else:
                clear_fake_cp_cache = False
            recon = autoencoder.decode(
                latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
            )
            recons.append(recon)
        autoencoder.to("cpu")
        samples_x = torch.cat(recons, dim=2)
        # samples_x = samples_x.permute(0, 2, 1, 3, 4).contiguous()
        samples_x = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        # samples_x = torch.cat(recons, dim=2).to(torch.float32)
        # samples_x = samples_x.permute(0, 2, 1, 3, 4).contiguous()
        # samples_x = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        # print(samples_x.dtype)
        # print(samples_x.shape)
        # if mpu.get_model_parallel_rank() == 0:
        #     save_video_as_grid_and_mp4(samples_x, "reward_model_test.mp4", fps=8)
        print(f"Pre reward score Allocated memory: {torch.cuda.memory_allocated() / (1024**2)} MB")

        # reward score
        loss = self.rewarder.reward_scorer(batch["txt"],samples_x)
        print("loss is " + str(loss))
        print(f"After reward score Allocated memory: {torch.cuda.memory_allocated() / (1024**2)} MB")
        # assert False

        # print(alphas_cumprod_sqrt)
        # print(idx)
        print("denoiser")
        print(vars(denoiser))
        print("sampler")
        print(vars(sampler))
        sigmas, timesteps = sampler.prepare_discretization()
        print("timesteps")
        print(sigmas)
        print(timesteps)
        print("sigma_sampler")
        print(vars(self.sigma_sampler))
        print("Loss haven't implemented!!!")
        assert False

        pass