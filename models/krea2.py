import os
import sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

import peft

from models.base import ComfyPipeline, make_contiguous, tokenize
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, is_main_process
from utils.offloading import ModelOffloader
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding


# Krea2 ("K2") is a single-stream MMDiT: text tokens produced by a Qwen3-VL-4B 12-layer
# "txtfusion" adapter and patchified image tokens are concatenated into one sequence and run
# through `blocks` shared SingleStreamBlocks (AdaLN-single, GQA + per-head QK-norm +
# sigmoid-gated attention, SwiGLU, 3-axis RoPE). It's a ComfyUI-native model (comfy/ldm/krea2),
# shares the Qwen-Image VAE (Wan21 16ch latent format), and is ModelType.FLUX (flow-matching),
# so training mirrors the flux2/qwen_image flow-matching setup: target = noise - latents,
# timestep in [0, 1] fed straight to comfy's timestep_embedding (which applies the 1000x factor
# internally), with sampling shift = 1.15.
class Krea2Pipeline(ComfyPipeline):
    name = 'krea2'
    checkpointable_layers = ['InitialLayer', 'TransformerLayer']
    # LoRA-target the 28 SingleStreamBlocks AND the text-conditioning pathway (the TextFusion
    # transformer + the txtmlp projection, the latter handled by name in configure_adapter). Both
    # the official upstream diffusion-pipe krea2 and ai-toolkit train this text pathway — freezing
    # it (blocks-only) prevents a trigger-word concept from binding into the image stream.
    adapter_target_modules = ['SingleStreamBlock', 'TextFusionTransformer']
    # These stay bf16 (not fp8) — but txtfusion/txtmlp are still LoRA-targeted above.
    keep_in_high_precision = ['first', 'tmlp', 'tproj', 'txtfusion', 'txtmlp', 'last']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def configure_adapter(self, adapter_config):
        # Like ComfyPipeline.configure_adapter, but also picks up the `txtmlp` Sequential (which is
        # not its own class in adapter_target_modules) by name — matching upstream diffusion-pipe.
        target_model = self.diffusion_model
        target_linear_modules = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules and 'txtmlp' not in name:
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

    def get_call_vae_fn(self, vae):
        # The krea2 VAE is the qwen-image VAE — a 3D/temporal VAE (comfy latent_dim==3). Comfy's
        # VAE.encode turns a 4D image batch (b,c,h,w) into ONE b-frame clip (batch axis -> temporal),
        # yielding a single latent per batch and breaking len(latents)==len(metadata) during caching.
        # So we feed each image as its own length-1 clip (5D input, which comfy leaves untouched),
        # giving one 5D latent (b,c,1,h,w) per image. We keep the temporal axis here because the
        # Wan21 latent format's process_latent_in is 5D; prepare_inputs squeezes it after that step.
        @torch.inference_mode()
        def fn(images):
            images = images.to('cuda')
            images = images.movedim(1, -1)    # (b,c,h,w) -> (b,h,w,c)
            images = images.unsqueeze(1)      # -> (b,1,h,w,c): one 1-frame clip per image
            images = (images + 1) / 2         # comfy expects pixels in [0,1]
            latents = vae.encode(images)      # comfy VAE.encode -> (b,c,1,h,w), kept 5D
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        # Same as ComfyPipeline.get_call_text_encoder_fn, but the Krea2 text encoder drops the
        # attention_mask from `extra` when every token is valid (no padding). That happens whenever
        # a caching batch is a single caption (or all captions share a length), so we must fall
        # back to an all-ones mask instead of KeyError-ing.
        te_idx = None
        for i, te in enumerate(self.text_encoders):
            if text_encoder == te:
                te_idx = i
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool]):
            tokenizer = getattr(text_encoder.tokenizer, text_encoder.tokenizer.clip)

            max_length = 0
            for text in captions:
                tokens = tokenize(text_encoder, text)
                for v in tokens.values():
                    max_length = max(max_length, len(v[0]))

            # Pad to max length in the batch ourselves, or the ComfyUI backend concats
            # variable-length tensors and fails.
            tokenizer.min_length = max_length
            tokens_dict = defaultdict(list)
            for text in captions:
                tokens = tokenize(text_encoder, text)
                for k, v in tokens.items():
                    tokens_dict[k].extend(v)

            o = text_encoder.encode_from_tokens_scheduled(tokens_dict)

            text_embeds = o[0][0]
            extra = o[0][1]
            attention_mask = extra.get('attention_mask')
            if attention_mask is None:
                attention_mask = torch.ones(
                    text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device
                )
            return {
                f'text_embeds_{te_idx}': text_embeds,
                f'attention_mask_{te_idx}': attention_mask,
            }

        return fn

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        # The Wan21 latent format's process_latent_in is 5D (latents_mean/std carry a temporal axis).
        # Cached latents may be 4D (bs,c,h,w) or 5D (bs,c,1,h,w) depending on when they were cached,
        # so ensure a length-1 frame axis, apply process_latent_in, then squeeze back to 4D for the
        # rest of prepare_inputs + the (image-mode) DiT.
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        latents = self.model_patcher.model.process_latent_in(latents)
        if latents.ndim == 5:
            latents = latents.squeeze(2)
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

        bs, c, h, w = latents.shape
        device = latents.device

        # (bs, txt_len) bool text mask. The image portion of the joint mask is all-ones and gets
        # appended inside InitialLayer where the image token count is known.
        text_attention_mask = attention_mask.to(torch.bool)

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

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        return (noisy_latents, t, text_embeds, text_attention_mask), (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.blocks = None
        diffusion_model.to('cuda')
        diffusion_model.blocks = blocks
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
        self.first = model.first
        self.tmlp = model.tmlp
        self.tproj = model.tproj
        self.txtfusion = model.txtfusion
        self.txtmlp = model.txtmlp
        self.pe_embedder = model.pe_embedder
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timesteps, context, text_mask = inputs

        bs, c, H_orig, W_orig = x.shape
        patch = self.patch
        # Pad the latent up to a multiple of patch (as Flux/QwenImage do); crop back in FinalLayer.
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        # context arrives as (B, seq, txtlayers*txtdim); reshape to (B, seq, txtlayers, txtdim).
        context = self._unpack_context(context)

        img = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch)
        img = self.first(img)

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(context, mask=None)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)

        # Position ids: text at 0, image at (0, h_idx, w_idx).
        device = combined.device
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
        imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
        imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
        imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
        pos = torch.cat((txtpos, imgpos), dim=1)
        freqs = self.pe_embedder(pos)

        # Joint attention mask: [text mask, image all-ones], broadcastable over SDPA
        # heads/query dims as (bs, 1, 1, txt+img).
        img_mask = torch.ones((bs, imglen), dtype=torch.bool, device=device)
        attention_mask = torch.cat((text_mask, img_mask), dim=1).view(bs, 1, 1, -1)

        sizes = torch.tensor([txtlen, imglen, h_, w_, H_orig, W_orig], device=device)

        outputs = make_contiguous(combined, tvec, t, freqs, attention_mask, sizes)
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
        combined, tvec, t, freqs, attention_mask, sizes = inputs

        self.offloader.wait_for_block(self.block_idx)
        combined = self.block(combined, tvec, freqs, attention_mask)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(combined, tvec, t, freqs, attention_mask, sizes)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.last = model.last
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        combined, tvec, t, freqs, attention_mask, sizes = inputs
        txtlen, imglen, h_, w_, H_orig, W_orig = sizes.tolist()
        patch = self.patch

        final = self.last(combined, t)
        out = final[:, txtlen:txtlen + imglen, :]
        out = rearrange(
            out, 'b (h w) (c ph pw) -> b c (h ph) (w pw)',
            h=h_, w=w_, ph=patch, pw=patch, c=self.channels,
        )
        out = out[:, :, :H_orig, :W_orig]
        return out
