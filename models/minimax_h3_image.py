# MiniMax-H3-Image (T2I) training — the H3 DiT without its audio stream, one latent frame, PDD-distilled to 6 steps.
#
# Built on models/minimax_h3.py: importing it applies the fork's comfy patches (out-of-place AdaLN/gate, batch-enabled
# AdalnProj/DiTBlock/Attention with attention_mask, eager split-half rope, PDD-aware batched FinalLayer), so the
# comfy loader builds comfy's MiniMaxH3ImageModel out of the patched blocks. LoRA targeting (DiTBlock + RefinerBlock,
# AdaLN excluded), block swap and the quantized-comfy load path are inherited from MinimaxH3Pipeline.
#
# Deviations from comfy's MiniMaxH3ImageModel._forward (comfy/ldm/minimax/model.py), all needed for batched training:
#   - batch > 1: text is right-padded to the longest caption in the batch; pad rows are masked out as attention keys
#   - the token refiner runs on each caption's unpadded tokens (comfy's refiner has no mask, pad rows would leak in)
#   - per-caption RoPE: image rows start their t-axis at the caption's real length (comfy: = text_len, unpadded),
#     not at the padded length -> rope tables are per sample, [B, S, 1, rot/2, 2, 2]
#   - PDD head: comfy picks the head bank interval from the sampler's sigma schedule; training passes the grid
#     index of each sample's sigma explicitly (FinalLayer head_index)
# T2I only: no reference images (<Picture n> edit refs), no text_token_tags (all text rows are tag 1 without refs).
#
# Training timesteps: ONLY the checkpoint's PDD grid sigmas (safetensors metadata config.transformer.pdd_sigmas,
# drop the final 0), each through its own head -- as ai-toolkit does. Continuous sigma sampling would train heads
# off the interval they were distilled for. Optional [model] pdd_step_weights = [w0..w5] reweights the 6 sigmas.
#
# The model is step+guidance distilled: plain flow-matching LoRA training de-distills it. Use ostris'
# minimax_h3_image_training_adapter_v1 as a frozen [[adapter.runtime_adapters]] during training and leave it OFF
# at inference (same exception to "train regime = inference regime" as adapter v3 for H3 video).
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F

from models.base import ComfyPipeline, make_contiguous, PreprocessMediaFile
from models import minimax_h3  # applies the comfy patches
from models.minimax_h3 import MinimaxH3Pipeline, FRAMERATE
from utils.common import AUTOCAST_DTYPE, VIDEO_EXTENSIONS
from utils.offloading import ModelOffloader
import comfy.latent_formats
import comfy.model_management
from comfy.ldm.minimax.model import PackedLayout, SEG_TAG, patchify_video, unpatchify_video, rope_rotation_table


class PreprocessMediaFileH3Image(PreprocessMediaFile):
    # support_video=True only for the (c, 1, h, w) layout, so latents stay 5D like the H3 video pipeline's stills
    def __init__(self, config):
        super().__init__(config, support_video=True, framerate=FRAMERATE, round_height=32, round_width=32)

    def __call__(self, spec, mask_filepath, size_bucket=None):
        if Path(spec[1]).suffix in VIDEO_EXTENSIONS:
            raise ValueError(f'MiniMax-H3-Image trains on images only, got a video: {spec[1]}')
        return super().__call__(spec, mask_filepath, size_bucket=size_bucket)


class MinimaxH3ImagePipeline(MinimaxH3Pipeline):
    name = 'minimax_h3_image'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['DiTBlock', 'RefinerBlock']
    keep_in_high_precision = ['time_embedder', 'condition_proj', 'final_layer', 'rope.inv_freq', 'token_refiner', 'video_patch_proj', 'adaln_t_table']
    spatial_compression = 16
    channels = 24
    is_video_vae = True

    def __init__(self, config):
        # skip MinimaxH3Pipeline.__init__ (video + audio VAE pair): one VAE, either the H3-Image VAE or the H3 video
        # VAE -- the encoders are bit-identical, so both give the same latents. ComfyPipeline's VAE patch handles 5D.
        ComfyPipeline.__init__(self, config)
        self.is_video_vae = True
        self.latent_format = comfy.latent_formats.MiniMaxH3Video()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.framerate = FRAMERATE
        self.cfg = self.model_config.get('cfg', 1.0)
        if self.cfg != 1.0:
            raise NotImplementedError('minimax_h3_image: CFG-augmented training is not implemented; use the ostris '
                                      'image training adapter as a frozen runtime_adapter instead')

    def load_diffusion_model(self):
        super().load_diffusion_model()
        dm = self.diffusion_model
        if not hasattr(dm, 'pdd_sigmas'):
            raise RuntimeError('minimax_h3_image: the checkpoint did not load as MiniMaxH3ImageModel '
                               f'(got {dm.__class__.__name__}) -- wrong diffusion_model file?')
        n = dm.final_layer.pdd_steps
        grid = [float(s) for s in dm.pdd_sigmas]
        if len(grid) != n + 1:
            raise RuntimeError(f'minimax_h3_image: pdd_sigmas has {len(grid)} points for {n} PDD heads')
        self.pdd_sigmas = torch.tensor(grid[:-1], dtype=torch.float32)  # one training sigma per head
        weights = self.model_config.get('pdd_step_weights', None)
        if weights is None:
            weights = [1.0] * n
        if len(weights) != n:
            raise ValueError(f'pdd_step_weights needs {n} entries, got {len(weights)}')
        self.pdd_step_weights = torch.tensor(weights, dtype=torch.float32)
        print(f'MiniMax-H3-Image: {n} PDD heads, training sigmas {[round(s, 4) for s in grid[:-1]]}, '
              f'weights {weights}')

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFileH3Image(self.config)

    # single VAE, images only: ComfyPipeline's vae_encode (process_in with MiniMaxH3Video latent format)
    def vae_encode(self, img):
        return ComfyPipeline.vae_encode(self, img)

    def get_call_vae_fn(self, vae):
        return ComfyPipeline.get_call_vae_fn(self, vae)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']
        bs, c, f, h, w = latents.shape
        assert f == 1, f'MiniMax-H3-Image latents must be single-frame, got {f} frames'
        device = latents.device

        text_embeds, attention_mask = self.get_conds(inputs)
        text_lens = attention_mask.sum(dim=1).to(torch.int32)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')
            mask = mask.unsqueeze(2)  # (bs, 1, 1, h, w)
        else:
            mask = torch.tensor([])

        n = self.pdd_sigmas.shape[0]
        if timestep_quantile is not None:
            idx = torch.full((bs,), min(int(timestep_quantile * n), n - 1), dtype=torch.long)
        else:
            idx = torch.multinomial(self.pdd_step_weights, bs, replacement=True)
        t = self.pdd_sigmas[idx].to(device)
        head_index = idx.to(device=device, dtype=torch.int32)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        return (noisy_latents, t, head_index, text_embeds, attention_mask, text_lens), (target, mask)

    def get_loss_fn(self):
        return ComfyPipeline.get_loss_fn(self)


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.use_adaln_curves:
            self.adaln_t_table = model.adaln_t_table
        else:
            self.time_embedder = model.time_embedder
        self.video_patch_proj = model.video_patch_proj
        self.condition_proj = model.condition_proj
        self.rope = model.rope
        self.token_refiner = model.token_refiner
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def make_packed_sequence(self, x, t, context, context_mask, text_len_real):
        # one sample with batch dim 1; context is right-padded to the batch's longest caption
        assert x.shape[0] == 1
        device = x.device
        dtype = context.dtype
        latent_t, lat_h, lat_w = x.shape[2], x.shape[3], x.shape[4]
        assert lat_h % self.patch_size[1] == 0 and lat_w % self.patch_size[2] == 0, 'latent H/W must be even'
        text_len = context.shape[1]
        n_txt = int(text_len_real)

        # sequence structure uses the padded text length (identical across the batch) ...
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, 0)
        # ... rope positions the caption's real length, as comfy inference sees it (pad rows are masked keys,
        # their positions are irrelevant -- they continue the text axis)
        pos = layout.position_ids.clone()
        if n_txt != text_len:
            real = PackedLayout(n_txt, latent_t, lat_h, lat_w, 0).position_ids
            pos[text_len:] = real[n_txt:]

        sigma = t.flatten()[0].float()
        tv = float(1.0 - sigma)
        mod_segments = [(a, b, SEG_TAG[kind]) for a, b, kind in layout.segments]  # one t row: row base 0

        video_rows = patchify_video(x.to(torch.float32), self.patch_size)
        video_embed = self.video_patch_proj(video_rows).to(dtype)

        text_states = context[:, :n_txt]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))

        h = torch.zeros(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        attention_mask = torch.ones((layout.seq_len,), dtype=context_mask.dtype, device=device)
        for a, b, kind in layout.segments:
            if kind == 'text':
                h[a:a + n_txt] = text_states[0]
                attention_mask[a:b] = context_mask[0]
            else:
                assert kind == 'video'
                h[a:b] = video_embed

        t_vals = torch.tensor([tv], dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            p = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = p.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (p - i0).unsqueeze(1))
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)

        rope_freqs = rope_rotation_table(self.rope_freqs(pos, device), dtype)  # [1, S, 1, rot/2, 2, 2]
        va, vb, _ = layout.segments[-1]
        mod_segments = torch.tensor(mod_segments, dtype=torch.int32, device=device)
        extra_ints = torch.tensor([va, vb, 0, latent_t, lat_h, lat_w], dtype=torch.int32, device=device)
        return h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        x, t, head_index, context, context_mask, text_lens = inputs
        first = None
        h_list, mask_list, temb_list, rope_list = [], [], [], []
        for items in zip(x, t, context, context_mask):
            i = len(h_list)
            h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints = self.make_packed_sequence(
                *(v.unsqueeze(0) for v in items), text_lens[i])
            if first is None:
                first = (mod_segments, extra_ints)
            else:
                # batched AdaLN needs one segment table for the whole batch (same padded text length + bucket)
                assert (mod_segments == first[0]).all() and (extra_ints == first[1]).all()
            h_list.append(h)
            mask_list.append(attention_mask)
            temb_list.append(t_emb)
            rope_list.append(rope_freqs)
        h = torch.stack(h_list)
        attention_mask = torch.stack(mask_list)[:, None, None, :]  # 4D bool for SDPA
        t_emb = torch.stack(temb_list)
        rope_freqs = torch.cat(rope_list)  # [B, S, 1, rot/2, 2, 2]
        outputs = make_contiguous(h, attention_mask, t_emb, first[0], rope_freqs, first[1], head_index)
        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, head_index = inputs
        self.offloader.wait_for_block(self.block_idx)
        h = self.layer(h, t_emb, mod_segments.tolist(), rope_freqs, attention_mask=attention_mask)
        self.offloader.submit_move_blocks_forward(self.block_idx)
        return make_contiguous(h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, head_index)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, head_index = inputs
        video_seg = extra_ints[:3]
        latent_t, lat_h, lat_w = extra_ints[3:]
        v, = self.final_layer(h, t_emb, video_seg, None, head_index=head_index)
        out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        return -out
