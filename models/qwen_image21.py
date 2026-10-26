import os
import sys
import math
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

import peft

from models.base import ComfyPipeline, load_runtime_adapters, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, is_main_process
from utils.offloading import ModelOffloader
import comfy.ldm.qwen_image21.model as qi21_model
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention


# Qwen-Image 2.1 is NOT an update of Qwen-Image: it's a new 7B single-stream DiT
# (comfy/ldm/qwen_image21). Text tokens (Qwen3-VL-8B last hidden state, pre-norm, projected by
# `txt_in`) and image tokens (64-ch RGBA VAE latents at 16x, patch size 1, embedded by `img_in`)
# form one sequence [text, image] that runs through 32 shared `transformer_blocks`. Attention is
# block-causal: text tokens attend causally among themselves, image tokens attend to everything.
# One `modulation` MLP (from the timestep embedding) is shared by all blocks; the text prefix is
# modulated with t = 0, the image rows with the sampled t. Flow matching (comfy ModelType.FLUX):
# target = noise - latents, t in [0, 1], the model applies the x1000 timestep factor itself.
#
# Training-specific deviations from comfy's `_forward` (which we split into layers below):
#   - comfy's `_gated_residual` adds the block output IN PLACE into the residual stream. In fp32
#     autograd rejects that outright (LayerNorm saved the pre-add tensor); under bf16 autocast it
#     slips through (the norm saves an fp32 copy) but still mutates the tensor that activation
#     checkpointing / pipeline parallelism hold as the layer INPUT, so recomputation would see the
#     post-add value. We swap in an out-of-place version at import time (test/test_qwen_image21.py). `in_training` (set in models/base.py) already routes the norms/rope
#     off the fused comfy-kitchen kernels that have no backward.
#   - comfy's DiT ignores the text attention mask entirely (ComfyUI feeds one prompt at a time).
#     Training batches are right-padded, so we build the block-causal mask ourselves with pad
#     tokens excluded as keys, and give the image block its RoPE start position from each
#     caption's TRUE length (not the padded one) to keep train == inference positions.
#   - The inference-only prefix K/V cache is never touched (we drive the blocks directly).
#
# Edit training (dataset `control_path` = source images, stem-matched to the targets): mirrors the
# TextEncodeQwenImage21 node with one reference image. The text encoder sees the source (resized to
# ~edit_resolution^2 at multiples of 32, like the node) and drops its vision tokens, recording the
# slot where the image sat; the source's VAE latents are spliced in at that slot:
# [text[:slot], ref, text[slot:], target]. Text and reference rows form the t = 0 prefix, only the
# target is noised and supervised. The reference is VAE-encoded at the target's bucket size, so
# ref size == target size, as at inference (the node sizes the target latent to the first ref).


def _gated_residual_out_of_place(x, y, gate, prefix_len):
    g_prefix, g_target = gate
    if prefix_len:
        return torch.cat([x[:, :prefix_len] + y[:, :prefix_len] * g_prefix,
                          x[:, prefix_len:] + y[:, prefix_len:] * g_target], dim=1)
    return x + y * g_target


# QwenImage21TransformerBlock.forward looks this helper up as a module global on every call.
qi21_model._gated_residual = _gated_residual_out_of_place


def _masked_attention(attn_mask):
    # Replaces comfy's `block_causal_attention` closure: one SDPA over the whole sequence with an
    # explicit (bs, 1, N, N) bool mask (True = may attend) that encodes block-causality AND padding.
    def attn(q, k, v, heads, preferred_attention=None):
        return optimized_attention(q.flatten(2), k.flatten(2), v.flatten(2), heads, mask=attn_mask, preferred_attention=preferred_attention)
    return attn


class QwenImage21Pipeline(ComfyPipeline):
    name = 'qwen_image21'
    checkpointable_layers = ['InitialLayer', 'TransformerLayer']
    # LoRA-target every linear in the 32 blocks (attention to_q/k/v/to_out + SwiGLU MLP) plus the
    # text-conditioning entry (`txt_in`, name-matched in configure_adapter) and the image embedder
    # (`img_in`). The krea2 lesson: freezing the text pathway can leave the blocks overfitting a
    # frozen text representation instead of binding a trigger concept, so it stays trainable here.
    adapter_target_modules = ['QwenImage21TransformerBlock']
    adapter_target_names = ['txt_in', 'img_in']
    # Kept in bf16 when diffusion_model_dtype='float8' (the block linears are what gets quantized).
    keep_in_high_precision = ['txt_in', 'img_in', 'time_text_embed', 'modulation', 'norm_out', 'proj_out']
    # TextEncodeQwenImage21 sizes reference images (and so the edit target) at multiples of 32.
    pixels_round_to_multiple = 32

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def configure_adapter(self, adapter_config):
        target_model = self.diffusion_model
        target_linear_modules = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules and not any(name == n or name.startswith(n + '.') for n in self.adapter_target_names):
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.add(full_submodule_name)
        target_linear_modules = list(target_linear_modules)

        if adapter_config['type'] != 'lora':
            raise NotImplementedError(f"Adapter type {adapter_config['type']} is not implemented")
        peft_config = peft.LoraConfig(
            r=adapter_config['rank'],
            lora_alpha=adapter_config['alpha'],
            lora_dropout=adapter_config['dropout'],
            bias='none',
            target_modules=target_linear_modules,
        )
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(target_model, peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in target_model.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

        # Frozen runtime_adapters (see load_runtime_adapters in base.py); single-file adapters get
        # their synthetic LoraConfig from this method's target list, so txt_in/img_in keys match too.
        load_runtime_adapters(self.lora_model, target_model, target_linear_modules, adapter_config)

    # VAE: the 2.1 VAE is a 2D-latent (no frame axis) 16x RGBA autoencoder. comfy's VAE.encode takes
    # (b, h, w, c) RGB in [0, 1] and pads alpha = 1 itself, returning (b, 64, h/16, w/16) — the
    # ComfyPipeline default VAE fn does exactly that; edit datasets additionally encode the source.
    def get_call_vae_fn(self, vae):
        parent_vae_fn = super().get_call_vae_fn(vae)

        def fn(tensor, control_tensor=None):
            result = parent_vae_fn(tensor)
            if control_tensor is not None:
                result['control_latents'] = parent_vae_fn(control_tensor)['latents']
            return result
        return fn

    def _load_reference_image(self, path):
        # TextEncodeQwenImage21's resize: ~resolution^2 pixels, aspect kept, multiples of 32, lanczos;
        # the vision tower sees alpha composited over white.
        resolution = self.model_config.get('edit_resolution', 1024)
        img = Image.open(path)
        img = img.convert('RGBA') if img.mode in ('RGBA', 'LA', 'P') else img.convert('RGB')
        image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None]  # (1, H, W, C)
        samples = image.movedim(-1, 1)
        ratio = samples.shape[3] / samples.shape[2]
        width = max(32, round(math.sqrt(resolution * resolution * ratio) / 32) * 32)
        height = max(32, round(math.sqrt(resolution * resolution / ratio) / 32) * 32)
        if (width, height) != (samples.shape[3], samples.shape[2]):
            image = comfy.utils.common_upscale(samples, width, height, 'lanczos', 'disabled').movedim(1, -1)
        rgb = image[..., :3]
        if image.shape[-1] > 3:
            rgb = rgb * image[..., 3:] + (1.0 - image[..., 3:])
        return rgb

    def get_call_text_encoder_fn(self, text_encoder):
        # Differs from ComfyPipeline.get_call_text_encoder_fn in two ways:
        # 1. It tokenizes through the model's own tokenizer method (QwenImage21Tokenizer ->
        #    Qwen3VLTokenizer), which wraps the caption in the Qwen-Image 2.1 chat template and
        #    forces disable_weights itself. The shared `tokenize` helper calls
        #    SD1Tokenizer.tokenize_with_weights directly, which would skip the template and encode
        #    the bare caption — not what ComfyUI feeds the DiT at inference.
        # 2. The 2.1 text encoder drops the system turn and pops `attention_mask` from `extra`
        #    when nothing is padded (single caption per caching batch, or equal lengths), so fall
        #    back to an all-ones mask.
        te_idx = None
        for i, te in enumerate(self.text_encoders):
            if text_encoder == te:
                te_idx = i
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool], control_file: list[str] | None):
            if control_file is not None:
                # Edit pairs: one caption per encode call, because the TE reports the image slot for
                # the whole call. No padding needed; the cache stores variable-length rows anyway.
                embeds, masks, slots = [], [], []
                for text, path in zip(captions, control_file):
                    tokens = text_encoder.tokenizer.tokenize_with_weights(text, images=[self._load_reference_image(path)], prevent_empty_text=True)
                    o = text_encoder.encode_from_tokens_scheduled(tokens)
                    text_embeds, extra = o[0][0][0], o[0][1]
                    image_slots = extra.get('image_slots', [])
                    assert len(image_slots) == 1, f'expected one image slot, got {image_slots}'
                    embeds.append(text_embeds)
                    masks.append(torch.ones(text_embeds.shape[0], dtype=torch.bool, device=text_embeds.device))
                    slots.append(torch.tensor([image_slots[0]], dtype=torch.int64))
                return {
                    f'text_embeds_{te_idx}': embeds,
                    f'attention_mask_{te_idx}': masks,
                    f'image_slot_{te_idx}': slots,
                }

            tokenizer = getattr(text_encoder.tokenizer, text_encoder.tokenizer.clip)

            tokenizer.min_length = 1
            max_length = 0
            for text in captions:
                tokens = text_encoder.tokenizer.tokenize_with_weights(text)
                for v in tokens.values():
                    max_length = max(max_length, len(v[0]))

            # Pad to max length in the batch ourselves, or the ComfyUI backend concats
            # variable-length tensors and fails.
            tokenizer.min_length = max_length
            tokens_dict = defaultdict(list)
            for text in captions:
                tokens = text_encoder.tokenizer.tokenize_with_weights(text)
                for k, v in tokens.items():
                    tokens_dict[k].extend(v)

            o = text_encoder.encode_from_tokens_scheduled(tokens_dict)

            text_embeds = o[0][0]
            extra = o[0][1]
            attention_mask = extra.get('attention_mask')
            if attention_mask is None:
                attention_mask = torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device)
            return {
                f'text_embeds_{te_idx}': text_embeds,
                f'attention_mask_{te_idx}': attention_mask,
                # -1 = no reference image (T2I captions, and the uncond caption)
                f'image_slot_{te_idx}': torch.full((len(captions), 1), -1, dtype=torch.int64),
            }

        return fn

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.transformer_blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()  # (bs, 64, h, w); the 2.1 latent format is 2D, no frame axis
        latents = self.model_patcher.model.process_latent_in(latents)
        text_embeds = inputs['text_embeds_0']
        attention_mask = inputs['attention_mask_0']
        mask = inputs['mask']

        # text embeds are variable length
        max_seq_len = max([e.size(0) for e in text_embeds])
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in text_embeds]
        )
        attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attention_mask]
        )
        assert text_embeds.shape[:2] == attention_mask.shape[:2]
        text_attention_mask = attention_mask.to(torch.bool)

        bs, c, h, w = latents.shape
        device = latents.device

        if 'control_latents' in inputs:
            ref_latents = self.model_patcher.model.process_latent_in(inputs['control_latents'].float())
            assert ref_latents.shape == latents.shape, f'reference latents {tuple(ref_latents.shape)} != target latents {tuple(latents.shape)}'
            image_slot = inputs.get('image_slot_0')
            if image_slot is None:
                raise RuntimeError('Edit dataset but the text embedding cache has no image slots; re-cache with --regenerate_cache')
            image_slot = (torch.stack(image_slot) if isinstance(image_slot, list) else image_slot).view(bs).to(device)
            if (image_slot < 0).any():
                # the uncond caption is cached once, without a source image, so it has no slot
                raise NotImplementedError('uncond_fraction is not supported for Qwen-Image 2.1 edit datasets')
        else:
            ref_latents = latents.new_zeros((bs, c, 0, 0))
            image_slot = torch.full((bs,), -1, dtype=torch.int64, device=device)

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        # `shift` is the multiplicative (SD3-style) form; comfy's flow "shift" is its log (mu), so
        # ComfyUI's 0.69 at 1024x1024 == shift ~2.0 here. `flux_shift` reproduces the model's
        # resolution-dependent schedule: mu 0.5 at 256 image tokens -> 0.9 at 8192 (patch size 1,
        # so the token count is the latent h*w).
        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)(h * w)
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        return (noisy_latents, t, text_embeds, text_attention_mask, ref_latents, image_slot), (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.transformer_blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.transformer_blocks = None
        diffusion_model.to('cuda')
        diffusion_model.transformer_blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.txt_in = model.txt_in
        self.img_in = model.img_in
        self.time_text_embed = model.time_text_embed
        self.modulation = model.modulation
        self.pe_embedder = model.pe_embedder
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timesteps, context, text_mask, ref, image_slot = inputs

        bs, c, H, W = x.shape
        device = x.device
        dtype = AUTOCAST_DTYPE
        has_ref = ref.shape[-1] > 0

        # Mirrors QwenImage21Transformer2DModel.build_sequence with at most one reference:
        # sequence = [text[:slot], ref, text[slot:], target], images in raster order (patch size 1).
        # Without a reference, slot = the (padded) text length, i.e. [text, target]. Right-padding
        # sits at the end of text[slot:], so one slot works for the whole batch.
        txt = self.txt_in(context)
        txt_len = txt.shape[1]
        if has_ref:
            assert (image_slot == image_slot[0]).all(), f'image slots differ across the batch: {image_slot.tolist()}'
            slot = int(image_slot[0])
            h, w = ref.shape[-2:]
            ref_len = h * w
            hidden_states = torch.cat([txt[:, :slot], self.img_in(ref.flatten(2).transpose(1, 2)), txt[:, slot:], self.img_in(x.flatten(2).transpose(1, 2))], dim=1)
        else:
            slot, h, w, ref_len = txt_len, 0, 0, 0
            hidden_states = torch.cat([txt, self.img_in(x.flatten(2).transpose(1, 2))], dim=1)
        img_len = H * W
        prefix_len = txt_len + ref_len
        N = prefix_len + img_len
        ref_extent = max(h, w)  # RoPE axis-0 positions a reference block occupies

        def image_hw(h, w):
            # rows/cols centred like comfy, with half a token where the grid has the other parity than the target
            hh = torch.arange(h, device=device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - H % 2)
            ww = torch.arange(w, device=device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - W % 2)
            return torch.stack([hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1)

        # 3-axis RoPE ids. Text counts on all axes; a reference block sits at one axis-0 position and
        # advances the count by max(h, w). The target block starts (axis 0) after the caption's TRUE
        # token count, which is where inference puts it; pad tokens keep counting but are masked out.
        ids_a = torch.arange(slot, device=device, dtype=torch.float32).unsqueeze(1).expand(slot, 3)
        ids_ref = torch.cat([torch.full((ref_len, 1), float(slot), device=device), image_hw(h, w)], dim=-1)
        ids_b = (slot + ref_extent + torch.arange(txt_len - slot, device=device, dtype=torch.float32)).unsqueeze(1).expand(txt_len - slot, 3)
        img_hw = image_hw(H, W)
        txt_true_len = text_mask.sum(dim=1)
        ids = []
        for i in range(bs):
            img_pos = torch.full((img_len, 1), float(txt_true_len[i] + ref_extent), device=device, dtype=torch.float32)
            ids.append(torch.cat([ids_a, ids_ref, ids_b, torch.cat([img_pos, img_hw], dim=-1)], dim=0))
        ids = torch.stack(ids, dim=0)  # (bs, N, 3)
        # EmbedND -> (bs, 1, N, ...); comfy transposes to (bs, N, 1, ...) for its (B, N, H, D) q/k layout.
        pe = self.pe_embedder(ids).transpose(1, 2).contiguous()

        # Timestep -> shared modulation. Row order matches comfy: [t_0 .. t_{bs-1}, 0]; the last
        # (t = 0) row modulates the prefix (text + reference), the others the target rows.
        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)  # (bs + 1, D)
        mod = self.modulation(temb)  # (bs + 1, 4D)

        # Block-causal attention mask with padding: (bs, 1, N, N) bool, True = may attend.
        # Query q sees key k iff k < limit[q] and k is not a pad token, where limit is q + 1 for text
        # (causal over everything before it, reference included), the block end for the reference,
        # and N for the target — comfy's segments.
        idx = torch.arange(N, device=device)
        limit = idx + 1
        limit[slot:slot + ref_len] = slot + ref_len
        limit[prefix_len:] = N
        key_valid = torch.cat([
            text_mask[:, :slot],
            torch.ones((bs, ref_len), dtype=torch.bool, device=device),
            text_mask[:, slot:],
            torch.ones((bs, img_len), dtype=torch.bool, device=device),
        ], dim=1)  # (bs, N)
        attn_mask = (idx[None, :] < limit[:, None])[None] & key_valid[:, None, :]
        attn_mask = attn_mask.unsqueeze(1)

        sizes = torch.tensor([prefix_len, H, W], device=device)

        outputs = make_contiguous(hidden_states, pe, mod, temb, attn_mask, sizes)
        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, pe, mod, temb, attn_mask, sizes = inputs
        prefix_len = int(sizes[0])

        # Same unpacking as QwenImage21Transformer2DModel._forward. The trailing `zero` entry is
        # only consumed by the fused inference kernels, which `in_training` bypasses.
        scale1, gate1, scale2, gate2 = mod.chunk(4, dim=-1)
        block_mod = (
            qi21_model._split_rows(scale1),
            qi21_model._split_rows(gate1.tanh()),
            qi21_model._split_rows(scale2),
            qi21_model._split_rows(gate2.tanh()),
            None,
        )

        self.offloader.wait_for_block(self.block_idx)
        hidden_states = self.block(hidden_states, block_mod, pe, _masked_attention(attn_mask), prefix_len)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, pe, mod, temb, attn_mask, sizes)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, pe, mod, temb, attn_mask, sizes = inputs
        prefix_len, H, W = sizes.tolist()
        bs = hidden_states.shape[0]

        out = self.norm_out(hidden_states[:, prefix_len:], temb[:-1])
        out = self.proj_out(out)
        return out.transpose(1, 2).reshape(bs, self.out_channels, H, W)
