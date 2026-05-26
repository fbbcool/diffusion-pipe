from pathlib import Path
import re
import tarfile
import os
import sys
from collections import defaultdict
import types
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import peft
import torch
from torch import nn
import torch.nn.functional as F
import safetensors.torch
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms
import imageio

from utils.common import is_main_process, VIDEO_EXTENSIONS, round_to_nearest_multiple, round_down_to_multiple
import comfy.utils
import comfy.sd
import comfy.sd1_clip
from comfy.sd1_clip import SD1Tokenizer
from comfy import model_management
# Avoids using comfy_kitchen RoPE implementations that don't have backward defined
model_management.in_training = True


def make_contiguous(*tensors):
    return tuple(x.contiguous() for x in tensors)


def extract_clips(video, target_frames, video_clip_mode):
    # video is (channels, num_frames, height, width)
    frames = video.shape[1]
    if frames < target_frames:
        # TODO: think about how to handle this case. Maybe the video should have already been thrown out?
        print(f'video with shape {video.shape} is being skipped because it has less ({frames}) than the target_frames {target_frames}')
        return []

    if video_clip_mode == 'single_beginning':
        return [video[:, :target_frames, ...]]
    elif video_clip_mode == 'single_middle':
        start = int((frames - target_frames) / 2)
        assert frames-start >= target_frames
        return [video[:, start:start+target_frames, ...]]
    # elif video_clip_mode == 'multiple_overlapping':
    #     # Extract multiple clips so we use the whole video for training.
    #     # The clips might overlap a little bit. We never cut anything off the end of the video.
    #     num_clips = ((frames - 1) // target_frames) + 1
    #     start_indices = torch.linspace(0, frames-target_frames, num_clips).int()
    #     return [video[:, i:i+target_frames, ...] for i in start_indices]
    else:
        raise NotImplementedError(f'video_clip_mode={video_clip_mode} is not recognized')


def convert_crop_and_resize(pil_img, width_and_height):
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')

    # add white background for transparent images
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    return ImageOps.fit(pil_img, width_and_height)


class PreprocessMediaFile:
    def __init__(self, config, support_video=False, framerate=None, round_height=16, round_width=16, round_frames=4):
        self.config = config
        self.video_clip_mode = config.get('video_clip_mode', 'single_beginning')
        print(f'using video_clip_mode={self.video_clip_mode}')
        self.pil_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.support_video = support_video
        self.framerate = framerate
        print(f'using framerate={self.framerate}')
        self.round_height = round_height
        self.round_width = round_width
        self.round_frames = round_frames
        if self.support_video:
            assert self.framerate
        self.tarfile_map = {}

    def __del__(self):
        for tar_f in self.tarfile_map.values():
            tar_f.close()

    def __call__(self, spec, mask_filepath, size_bucket=None):
        is_video = (Path(spec[1]).suffix in VIDEO_EXTENSIONS)

        if spec[0] is None:
            tar_f = None
            filepath_or_file = str(spec[1])
        else:
            tar_filename = spec[0]
            if tar_filename not in self.tarfile_map:
                self.tarfile_map[tar_filename] = tarfile.TarFile(tar_filename)
            tar_f = self.tarfile_map[tar_filename]
            filepath_or_file = tar_f.extractfile(str(spec[1]))

        if is_video:
            assert self.support_video
            num_frames = 0
            for frame in imageio.v3.imiter(filepath_or_file, fps=self.framerate):
                num_frames += 1
                height, width = frame.shape[:2]
            video = imageio.v3.imiter(filepath_or_file, fps=self.framerate)
        else:
            num_frames = 1
            pil_img = Image.open(filepath_or_file)
            height, width = pil_img.height, pil_img.width
            video = [pil_img]

        if size_bucket is not None:
            size_bucket_width, size_bucket_height, size_bucket_frames = size_bucket
        else:
            size_bucket_width, size_bucket_height, size_bucket_frames = width, height, num_frames

        height_rounded = round_to_nearest_multiple(size_bucket_height, self.round_height)
        width_rounded = round_to_nearest_multiple(size_bucket_width, self.round_width)
        frames_rounded = round_down_to_multiple(size_bucket_frames - 1, self.round_frames) + 1
        resize_wh = (width_rounded, height_rounded)

        if mask_filepath:
            mask_img = Image.open(mask_filepath).convert('RGB')
            img_hw = (height, width)
            mask_hw = (mask_img.height, mask_img.width)
            if mask_hw != img_hw:
                raise ValueError(
                    f'Mask shape {mask_hw} was not the same as image shape {img_hw}.\n'
                    f'Image path: {spec[1]}\n'
                    f'Mask path: {mask_filepath}'
                )
            mask_img = ImageOps.fit(mask_img, resize_wh)
            mask = torchvision.transforms.functional.to_tensor(mask_img)[0].to(torch.float16)  # use first channel
        else:
            mask = None

        resized_video = torch.empty((num_frames, 3, height_rounded, width_rounded))
        for i, frame in enumerate(video):
            if not isinstance(frame, Image.Image):
                frame = torchvision.transforms.functional.to_pil_image(frame)
            cropped_image = convert_crop_and_resize(frame, resize_wh)
            resized_video[i, ...] = self.pil_to_tensor(cropped_image)

        if hasattr(filepath_or_file, 'close'):
            filepath_or_file.close()

        if not self.support_video:
            return [(resized_video.squeeze(0), mask)]

        # (num_frames, channels, height, width) -> (channels, num_frames, height, width)
        resized_video = torch.permute(resized_video, (1, 0, 2, 3))
        if not is_video:
            return [(resized_video, mask)]
        else:
            videos = extract_clips(resized_video, frames_rounded, self.video_clip_mode)
            return [(video, mask) for video in videos]


class BasePipeline:
    framerate = None
    pixels_round_to_multiple = 16

    def load_diffusion_model(self):
        pass

    def get_vae(self):
        raise NotImplementedError()

    def get_text_encoders(self):
        raise NotImplementedError()

    def configure_adapter(self, adapter_config):
        target_linear_modules = set()
        for name, module in self.transformer.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.add(full_submodule_name)
        target_linear_modules = list(target_linear_modules)

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(self.transformer, peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

        # ---- runtime_adapters ---------------------------------------------
        # Load additional LoRAs that contribute to the forward pass but are
        # NOT trained or saved with this run. Use case: train a new LoRA
        # whose effective base includes one or more existing LoRAs at their
        # inference strengths, so the trained LoRA learns a delta that
        # COEXISTS with them rather than one that erodes them in pursuit of
        # the dataset distribution.
        #
        # Contrast with `merge_adapters` (qwen_image.py): that path bakes
        # adapter deltas into base weights at load. The trained LoRA then
        # has full freedom to undo those merged contributions because they
        # are inside its trainable parameter space. `runtime_adapters` keeps
        # the existing adapters STRUCTURALLY OUTSIDE the trainable space.
        #
        # Each entry is a dict:
        #   path:    either a PEFT-style adapter directory (adapter_config.json
        #            + adapter_model.safetensors) OR a single .safetensors file
        #            in ComfyUI/inference format, OR a directory containing one
        #            such .safetensors. The latter two cases derive rank/alpha
        #            from the tensors directly (matches `merge_adapters`
        #            behavior in qwen_image.py).
        #   name:    optional adapter name (default: file/dir basename)
        #   weight:  optional float strength multiplier (default: 1.0)
        #
        # Example TOML:
        #   [[adapter.runtime_adapters]]
        #   path   = '/path/to/xlasm10.safetensors'
        #   weight = 0.8
        runtime_entries = adapter_config.get('runtime_adapters', []) or []
        runtime_specs: list[tuple[str, float]] = []
        for entry in runtime_entries:
            if not isinstance(entry, dict):
                raise RuntimeError(
                    'runtime_adapters entry must be a dict with keys '
                    f'{{path, name?, weight?}}, got: {entry!r}'
                )
            raw_path = entry.get('path')
            if not raw_path:
                raise RuntimeError(
                    f"runtime_adapters entry missing 'path' key: {entry!r}"
                )
            adapter_path = Path(raw_path)
            if adapter_path.is_file():
                adapter_dir = adapter_path.parent
                explicit_file = adapter_path
            elif adapter_path.is_dir():
                adapter_dir = adapter_path
                explicit_file = None
            else:
                raise RuntimeError(
                    f'runtime_adapters path does not exist: {raw_path}'
                )
            adapter_name = entry.get('name') or (
                adapter_path.stem if explicit_file else adapter_dir.name
            )
            if adapter_name == 'default':
                raise RuntimeError(
                    "runtime_adapters entry name 'default' conflicts with "
                    'the trainable adapter; choose a different name'
                )
            weight = float(entry.get('weight', 1.0))

            has_peft_config = (adapter_dir / 'adapter_config.json').exists()
            if has_peft_config:
                # PEFT directory layout (e.g. produced by diffusion-pipe's
                # own save_adapter). PEFT reads the config, applies the
                # saved target_modules, and loads the safetensors.
                if is_main_process():
                    print(
                        f'[runtime_adapters] loading {adapter_dir} as frozen '
                        f'adapter "{adapter_name}" (weight={weight}, PEFT-dir)'
                    )
                self.lora_model.load_adapter(
                    str(adapter_dir), adapter_name=adapter_name
                )
            else:
                # ComfyUI / inference single-file format: no adapter_config.json.
                # Locate the .safetensors, derive rank from key shapes, build a
                # synthetic LoraConfig matching the trainable adapter's target
                # modules, and load weights with manual key remapping.
                if explicit_file is not None:
                    sft_file = explicit_file
                else:
                    sft_files = list(adapter_dir.glob('*.safetensors'))
                    if len(sft_files) != 1:
                        raise RuntimeError(
                            f'runtime_adapters dir {adapter_dir} has no '
                            'adapter_config.json and must contain exactly one '
                            f'.safetensors file (found {len(sft_files)})'
                        )
                    sft_file = sft_files[0]

                sd = safetensors.torch.load_file(str(sft_file))
                # Strip prefix to match base parameter naming (same regex
                # convention as load_adapter_weights and merge_adapters).
                normalized = {}
                a_ranks: list[int] = []
                for k, v in sd.items():
                    k2 = re.sub(r'^(transformer|diffusion_model)\.', '', k)
                    normalized[k2] = v
                    if k2.endswith('.lora_A.weight'):
                        a_ranks.append(v.shape[0])
                if not a_ranks:
                    raise RuntimeError(
                        f'runtime_adapters file {sft_file} has no '
                        "'lora_A.weight' tensors — not a recognizable LoRA"
                    )
                # Common case: all lora_A tensors share rank. If they don't
                # (rare), use the max — PEFT will allocate that many rank
                # slots; smaller lora_A tensors are padded by load_state_dict
                # (no, actually strict=False just skips them — pre-warn).
                rank = a_ranks[0]
                if any(r != rank for r in a_ranks):
                    if is_main_process():
                        print(
                            '[runtime_adapters] WARN: non-uniform rank in '
                            f'{sft_file} (ranks={sorted(set(a_ranks))}); '
                            f'using rank={rank}. Modules with a different '
                            'rank will fall back to PEFT init (zero contribution).'
                        )

                synthetic_config = peft.LoraConfig(
                    r=rank,
                    lora_alpha=rank,  # alpha=rank → scale=1.0 baseline
                    lora_dropout=0.0,
                    bias='none',
                    target_modules=target_linear_modules,
                )
                if is_main_process():
                    print(
                        f'[runtime_adapters] loading {sft_file} as frozen '
                        f'adapter "{adapter_name}" (weight={weight}, '
                        f'inferred rank={rank})'
                    )
                self.lora_model.add_adapter(adapter_name, synthetic_config)

                # Now overwrite the freshly-added adapter's lora_A/lora_B
                # parameters with the saved weights. PEFT names them
                #   <module_path>.lora_A.<adapter_name>.weight
                # so we rewrite the keys from the saved-file convention
                # (<module_path>.lora_A.weight) to that form.
                model_param_names = set(
                    n for n, _ in self.transformer.named_parameters()
                )
                rewritten = {}
                n_loaded = 0
                n_skipped = 0
                for k, v in normalized.items():
                    if k.endswith('.lora_A.weight'):
                        k_new = k[: -len('.lora_A.weight')] + f'.lora_A.{adapter_name}.weight'
                    elif k.endswith('.lora_B.weight'):
                        k_new = k[: -len('.lora_B.weight')] + f'.lora_B.{adapter_name}.weight'
                    else:
                        # Stray non-LoRA key (rare); ignore.
                        n_skipped += 1
                        continue
                    if k_new in model_param_names:
                        rewritten[k_new] = v
                        n_loaded += 1
                    else:
                        n_skipped += 1
                if not rewritten:
                    raise RuntimeError(
                        f'runtime_adapters file {sft_file}: no LoRA keys '
                        'matched any module in the wrapped transformer '
                        '(target_modules mismatch or unexpected key prefixes)'
                    )
                missing, unexpected = self.transformer.load_state_dict(
                    rewritten, strict=False
                )
                if is_main_process():
                    print(
                        f'[runtime_adapters] "{adapter_name}": loaded '
                        f'{n_loaded} LoRA tensors, skipped {n_skipped} '
                        f'(load_state_dict: {len(unexpected)} unexpected keys)'
                    )

            runtime_specs.append((adapter_name, weight))

        if runtime_specs:
            # Activate trainable 'default' alongside all runtime adapters
            # so they all contribute to the forward pass. PEFT applies
            # their deltas in parallel within each LoraLayer.
            active = ['default'] + [n for n, _ in runtime_specs]
            # PeftModel.set_adapter rejects lists on many PEFT versions
            # (`TypeError: unhashable type: 'list'` from a `name in dict`
            # check inside PeftModel.set_adapter). Bypass it and call
            # set_adapter on each tuner layer directly — every LoraLayer's
            # set_adapter accepts a list and activates all listed adapters
            # simultaneously. This is also what the tuner-level
            # LoraModel.set_adapter wrapper does internally, just with
            # stricter validation upstream.
            from peft.tuners.tuners_utils import BaseTunerLayer
            for module in self.lora_model.modules():
                if isinstance(module, BaseTunerLayer):
                    module.set_adapter(active)

            # Freeze runtime adapter parameters. PEFT's set_adapter may
            # re-enable gradients on active adapters in some versions, so
            # we freeze AFTER set_adapter to guarantee the final state.
            # Both the saver (utils/saver.py) and DeepSpeed gradient
            # collection gate on requires_grad, so this is the single
            # switch that excludes them from training.
            n_frozen_total = 0
            for adapter_name, _ in runtime_specs:
                n_frozen = 0
                key_a = f'.{adapter_name}.weight'
                key_b = f'.{adapter_name}.bias'
                for pname, p in self.lora_model.named_parameters():
                    if key_a in pname or key_b in pname:
                        p.requires_grad = False
                        # Cast to trainable adapter's dtype for compute
                        # uniformity; PEFT defaults loaded adapters to
                        # fp32, which mismatches the bf16 forward pass.
                        p.data = p.data.to(adapter_config['dtype'])
                        n_frozen += 1
                n_frozen_total += n_frozen

            # Apply per-adapter strength. PEFT's per-adapter scaling lives
            # on each LoraLayer instance as a dict; multiplying it by the
            # user weight is the most version-portable handle (the
            # set_scale / set_adapter_scale helpers have come and gone
            # across PEFT releases).
            for module in self.lora_model.modules():
                scaling = getattr(module, 'scaling', None)
                if not isinstance(scaling, dict):
                    continue
                for adapter_name, weight in runtime_specs:
                    if adapter_name in scaling and weight != 1.0:
                        scaling[adapter_name] = scaling[adapter_name] * weight

            # Defensive: ensure every parameter has `original_name` set
            # (the saver relies on it for any param that ever flips to
            # requires_grad=True; runtime params don't, but future code
            # might walk all params).
            for name, p in self.transformer.named_parameters():
                if not hasattr(p, 'original_name'):
                    p.original_name = name

            if is_main_process():
                print(
                    f'[runtime_adapters] active adapter list: {active} '
                    f'(froze {n_frozen_total} runtime params across '
                    f'{len(runtime_specs)} adapter(s))'
                )
                # Re-print trainable count so the user sees that the
                # frozen adapters did NOT bump the trainable parameter
                # tally.
                self.lora_model.print_trainable_parameters()

    def save_adapter(self, save_dir, peft_state_dict):
        raise NotImplementedError()

    def load_adapter_weights(self, adapter_path):
        if is_main_process():
            print(f'Loading adapter weights from path {adapter_path}')
        safetensors_files = list(Path(adapter_path).glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise RuntimeError(f'No safetensors file found in {adapter_path}')
        if len(safetensors_files) > 1:
            raise RuntimeError(f'Multiple safetensors files found in {adapter_path}')
        adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
        modified_state_dict = {}
        model_parameters = set(name for name, p in self.transformer.named_parameters())
        for k, v in adapter_state_dict.items():
            # Replace Diffusers or ComfyUI prefix
            k = re.sub(r'^(transformer|diffusion_model)\.', '', k)
            # Replace weight at end for LoRA format
            k = re.sub(r'\.weight$', '.default.weight', k)
            if k not in model_parameters:
                raise RuntimeError(f'modified_state_dict key {k} is not in the model parameters')
            modified_state_dict[k] = v
        self.transformer.load_state_dict(modified_state_dict, strict=False)

    def load_and_fuse_adapter(self, path):
        peft_config = peft.LoraConfig.from_pretrained(path)
        lora_model = peft.get_peft_model(self.transformer, peft_config)
        self.load_adapter_weights(path)
        lora_model.merge_and_unload()

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def get_call_vae_fn(self, vae):
        raise NotImplementedError()

    def get_call_text_encoder_fn(self, text_encoder):
        raise NotImplementedError()

    def prepare_inputs(self, inputs, timestep_quantile=None):
        raise NotImplementedError()

    def to_layers(self):
        raise NotImplementedError()

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    # Get param groups that will be passed into the optimizer. Models can override this, e.g. SDXL
    # supports separate learning rates for unet and text encoders.
    def get_param_groups(self, parameters):
        return [{'params': parameters}]

    # Default loss_fn. MSE between output and target, with mask support.
    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                if 'huber_delta' in self.config:
                    loss = F.huber_loss(output, target, reduction='none', delta=self.config['huber_delta'])
                elif 'smooth_l1_beta' in self.config:
                    loss = F.smooth_l1_loss(output, target, reduction='none', beta=self.config['smooth_l1_beta'])
                else:
                    loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()
            return loss
        return loss_fn

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError('Block swapping is not implemented for this model')

    def prepare_block_swap_training(self):
        pass

    def prepare_block_swap_inference(self, disable_block_swap=False):
        pass


def encode_token_weights(self, token_weight_pairs):
    to_encode = list()
    max_token_len = 0
    has_weights = False
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        max_token_len = max(len(tokens), max_token_len)
        has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
        to_encode.append(tokens)

    sections = len(to_encode)
    if has_weights or sections == 0:
        if hasattr(self, "gen_empty_tokens"):
            to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
        else:
            to_encode.append(comfy.sd1_clip.gen_empty_tokens(self.special_tokens, max_token_len))

    o = self.encode(to_encode)
    out, pooled = o[:2]

    # if pooled is not None:
    #     first_pooled = pooled[0:1].to(model_management.intermediate_device())
    # else:
    #     first_pooled = pooled
    assert pooled is None
    first_pooled = None

    output = []
    for k in range(0, sections):
        z = out[k:k+1]
        if has_weights:
            z_empty = out[-1]
            for i in range(len(z)):
                for j in range(len(z[i])):
                    weight = token_weight_pairs[k][j][1]
                    if weight != 1.0:
                        z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
        output.append(z)

    if (len(output) == 0):
        r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
    else:
        r = (torch.cat(output, dim=0).to(model_management.intermediate_device()), first_pooled)

    if len(o) > 2:
        extra = {}
        for k in o[2]:
            v = o[2][k]
            extra[k] = v

        r = r + (extra,)
    return r

# Handle batch of different prompts.
comfy.sd1_clip.ClipTokenWeightEncoder.encode_token_weights = encode_token_weights


class ModelWrapper:
    '''Simple wrapper that delays loading the model, since the model isn't needed if the training data is already cached.'''
    def __init__(self, load_fn):
        self._load_fn = load_fn
        self._model = None

    def __getattr__(self, name):
        if self._model is None:
            raise RuntimeError("Model wasn't loaded, this shouldn't ever happen.")
        return getattr(self._model, name)

    def load_model_if_needed(self):
        if self._model is None:
            self._model = self._load_fn()


def tokenize(text_encoder, text):
    # to make sure we force disable_weights=True even if the child class doesn't set it
    return SD1Tokenizer.tokenize_with_weights(text_encoder.tokenizer, text, disable_weights=True)


class ComfyPipeline:
    framerate = None
    pixels_round_to_multiple = 16
    keep_in_high_precision = []

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']

        # VAE
        def load_fn():
            sd = comfy.utils.load_torch_file(self.model_config['vae'])
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()

            def vae_encode_crop_pixels(self, pixels):
                if not self.crop_input:
                    return pixels

                downscale_ratio = self.spacial_compression_encode()

                dims = pixels.shape[-3:-1]
                for d in range(len(dims)):
                    x = (dims[d] // downscale_ratio) * downscale_ratio
                    x_offset = (dims[d] % downscale_ratio) // 2
                    if x != dims[d]:
                        pixels = pixels.narrow(d + 1, x_offset, x)
                return pixels

            # patch this to handle 5D video tensor (original code expects 4D even for video)
            vae.vae_encode_crop_pixels = types.MethodType(vae_encode_crop_pixels, self.vae)
            return vae

        self.vae = ModelWrapper(load_fn)

        # Text encoders
        self.text_encoders = []
        for te_config in self.model_config['text_encoders']:
            if 'paths' in te_config:
                paths = te_config['paths']
            elif 'path' in te_config:
                paths = te_config['path']
            else:
                raise ValueError('need text encoder path in config')
            if isinstance(paths, str):
                paths = [paths]
            clip_type = getattr(comfy.sd.CLIPType, te_config['type'].upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

            def load_fn():
                return comfy.sd.load_clip(ckpt_paths=paths, clip_type=clip_type, disable_dynamic=True)

            self.text_encoders.append(ModelWrapper(load_fn))

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        model_options = {}
        model_options['dtype'] = dtype
        self.model_patcher = comfy.sd.load_diffusion_model(self.model_config['diffusion_model'], model_options=model_options)

        for adapter_path in self.model_config.get('merge_adapters', []):
            if is_main_process():
                print(f'Merging adapter {adapter_path}')
            sd = comfy.utils.load_torch_file(adapter_path, safe_load=True)
            self.model_patcher, _ = comfy.sd.load_lora_for_models(self.model_patcher, None, sd, 1.0, 0.0)

        self.model_patcher.set_model_compute_dtype(dtype)
        with torch.no_grad():
            self.model_patcher.patch_model()
        self.diffusion_model = self.model_patcher.model.diffusion_model

        diffusion_model_dtype = self.model_config.get('diffusion_model_dtype', dtype)
        for name, p in self.diffusion_model.named_parameters():
            if any(keyword in name for keyword in self.keep_in_high_precision) or p.ndim == 1:
                continue
            p.data = p.data.to(diffusion_model_dtype)

        self.diffusion_model.train()
        for name, p in self.diffusion_model.named_parameters():
            p.original_name = name
            p.requires_grad_(True)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return self.text_encoders

    def configure_adapter(self, adapter_config):
        target_linear_modules = set()
        for name, module in self.diffusion_model.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.add(full_submodule_name)
        target_linear_modules = list(target_linear_modules)

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(self.diffusion_model, peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in self.diffusion_model.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

    def save_adapter(self, save_dir, sd):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        sd = {'diffusion_model.'+k: v for k, v in sd.items()}
        safetensors.torch.save_file(sd, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def load_adapter_weights(self, adapter_path):
        if is_main_process():
            print(f'Loading adapter weights from path {adapter_path}')
        safetensors_files = list(Path(adapter_path).glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise RuntimeError(f'No safetensors file found in {adapter_path}')
        if len(safetensors_files) > 1:
            raise RuntimeError(f'Multiple safetensors files found in {adapter_path}')
        adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
        modified_state_dict = {}
        model_parameters = set(name for name, p in self.diffusion_model.named_parameters())
        for k, v in adapter_state_dict.items():
            # Replace Diffusers or ComfyUI prefix
            k = re.sub(r'^(transformer|diffusion_model)\.', '', k)
            # Replace weight at end for LoRA format
            k = re.sub(r'\.weight$', '.default.weight', k)
            if k not in model_parameters:
                raise RuntimeError(f'modified_state_dict key {k} is not in the model parameters')
            modified_state_dict[k] = v
        self.diffusion_model.load_state_dict(modified_state_dict, strict=False)

    def load_and_fuse_adapter(self, path):
        raise NotImplementedError()

    def save_model(self, save_dir, sd):
        safetensors.torch.save_file(sd, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def get_call_vae_fn(self, vae):
        @torch.inference_mode()
        def fn(images):
            images = images.to('cuda')
            # move channel dim to end
            # works for both images (b c h w) and video (b c f h w)
            images = images.movedim(1, -1)
            # Pixels are in range [-1, 1], Comfy code expects [0, 1]
            images = (images + 1) / 2
            latents = vae.encode(images)
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
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
                # tokens looks like {'qwen3_4b': [[(0, 1.0), (1, 1.0), (2, 1.0)]]}
                for v in tokens.values():
                    max_length = max(max_length, len(v[0]))

            # Pad to max length in the batch. We need to do this ourselves or the ComfyUI backend code will fail (it concats tensors assumed to be the same length).
            tokenizer.min_length = max_length
            tokens_dict = defaultdict(list)
            for text in captions:
                tokens = tokenize(text_encoder, text)
                for k, v in tokens.items():
                    tokens_dict[k].extend(v)

            o = text_encoder.encode_from_tokens_scheduled(tokens_dict)

            text_embeds = o[0][0]
            extra = o[0][1]
            attention_mask = extra['attention_mask']
            return {
                f'text_embeds_{te_idx}': text_embeds,
                f'attention_mask_{te_idx}': attention_mask,
            }

        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        raise NotImplementedError()

    def to_layers(self):
        raise NotImplementedError()

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    # Get param groups that will be passed into the optimizer. Models can override this, e.g. SDXL
    # supports separate learning rates for unet and text encoders.
    def get_param_groups(self, parameters):
        return [{'params': parameters}]

    # Default loss_fn. MSE between output and target, with mask support.
    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                if 'huber_delta' in self.config:
                    loss = F.huber_loss(output, target, reduction='none', delta=self.config['huber_delta'])
                elif 'smooth_l1_beta' in self.config:
                    loss = F.smooth_l1_loss(output, target, reduction='none', beta=self.config['smooth_l1_beta'])
                else:
                    loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()
            return loss
        return loss_fn

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError('Block swapping is not implemented for this model')

    def prepare_block_swap_training(self):
        pass

    def prepare_block_swap_inference(self, disable_block_swap=False):
        pass