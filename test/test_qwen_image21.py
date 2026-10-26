# Synthetic smoke test for models/qwen_image21.py — no model weights needed (tiny random DiT / VAE).
#   1. layer split (Initial/Transformer/FinalLayer) == comfy's QwenImage21Transformer2DModel._forward
#   2. right-padded batch == per-sample unpadded reference (block-causal mask + per-caption RoPE start)
#   3. LoRA backward through the split: every lora_B gets gradient at step 1, every lora_A at step 2
#   3b. shows why comfy's in-place _gated_residual is patched (fp32 autograd error)
#   4. comfy VAE detection + encode path for the 2.1 VAE layout: (b, h, w, 3) -> (b, 64, h/16, w/16)
#   5. edit path: reference spliced at its image slot == comfy _forward(ref_latents, image_slots),
#      unpadded and right-padded; odd/even ref-vs-target parity; LoRA grads with a reference
# Run from the repo root with a venv that has torch/peft + the ComfyUI submodule requirements
# (comfy-kitchen, comfy-aimdo, scipy, pyyaml, av, aiohttp, yarl):
#   python test/test_qwen_image21.py            # CPU, exact
#   DEV=cuda python test/test_qwen_image21.py   # GPU, bf16 autocast (loose tolerance)
import os, sys, math
FORK = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
COMFY = os.environ.get('COMFY', os.path.join(FORK, 'submodules', 'ComfyUI'))
sys.path.insert(0, FORK)
import utils.common  # fork's utils is a namespace pkg; import it before ComfyUI's regular utils/ pkg lands on sys.path (train.py order)
sys.path.insert(0, COMFY)
import torch
from torch import nn
torch.manual_seed(0)

dev = torch.device(os.environ.get('DEV', 'cpu'))

from comfy import model_management
model_management.in_training = True
import comfy.ops
import comfy.ldm.qwen_image21.model as qm
orig_gated = qm._gated_residual  # capture the in-place original before the fork patches it
from models import qwen_image21 as pipe
assert qm._gated_residual is pipe._gated_residual_out_of_place, 'patch not applied'

def build(seed=0):
    torch.manual_seed(seed)
    m = qm.QwenImage21Transformer2DModel(in_channels=8, out_channels=8, num_layers=3, attention_head_dim=16,
        num_attention_heads=4, context_in_dim=24, mlp_ratio=2, fused_mlp=True, axes_dims_rope=(4, 6, 6), dtype=torch.float32, device=dev,
        operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1: p.zero_() if 'text_norm' in n else p.fill_(1.0)
            else: p.normal_(0, 0.05)
    return m.to(dev).float()

class Dummy:
    def wait_for_block(self, i): pass
    def submit_move_blocks_forward(self, i): pass

def chain(m):
    return [pipe.InitialLayer(m)] + [pipe.TransformerLayer(b, i, Dummy()) for i, b in enumerate(m.transformer_blocks)] + [pipe.FinalLayer(m)]

def run_chain(layers, x, t, ctx, mask, ref=None, slot=None):
    bs = x.shape[0]
    if ref is None:
        ref, slot = x.new_zeros((bs, x.shape[1], 0, 0)), torch.full((bs,), -1, dtype=torch.int64, device=x.device)
    inputs = (x, t, ctx, mask, ref, slot)
    for l in layers: inputs = l(inputs)
    return inputs

B, C, H, W, T = 2, 8, 5, 6, 7
m = build()
layers = chain(m)
x = torch.randn(B, C, H, W, device=dev); t = torch.rand(B, device=dev); ctx = torch.randn(B, T, 24, device=dev)
mask = torch.ones(B, T, dtype=torch.bool, device=dev)

# (1) unpadded: split vs comfy reference
with torch.no_grad():
    ref = m._forward(x, t, ctx) if dev.type=='cpu' else torch.autocast('cuda', dtype=torch.bfloat16)(m._forward)(x, t, ctx)
    out = run_chain(layers, x, t, ctx, mask)
print('1 unpadded max|diff| =', (out - ref).abs().max().item(), 'ref scale', ref.abs().mean().item())
TOL = 1e-4 if dev.type=='cpu' else 2e-2
assert (out - ref).abs().max().item() < TOL

# (2) padded batch: caption 0 has 4 real tokens, caption 1 has 7; compare each to unpadded single-sample reference
lens = [4, 7]
mask2 = torch.zeros(B, T, dtype=torch.bool, device=dev)
for i, n in enumerate(lens): mask2[i, :n] = True
ctx2 = ctx.detach().clone(); ctx2[0, 4:] = 123.0  # garbage in the pad slots must not leak
with torch.no_grad():
    out2 = run_chain(layers, x, t, ctx2, mask2)
    for i, n in enumerate(lens):
        ref_i = m._forward(x[i:i+1], t[i:i+1], ctx2[i:i+1, :n]) if dev.type=='cpu' else torch.autocast('cuda', dtype=torch.bfloat16)(m._forward)(x[i:i+1], t[i:i+1], ctx2[i:i+1, :n])
        d = (out2[i:i+1] - ref_i).abs().max().item()
        print(f'2 padded sample {i} (len {n}) max|diff| =', d)
        assert d < TOL

# (3) LoRA backward through the split (out-of-place residual), grads reach lora_B and img_in/txt_in LoRAs
import peft
targets = set()
for name, mod in m.named_modules():
    if mod.__class__.__name__ in pipe.QwenImage21Pipeline.adapter_target_modules or any(name == n or name.startswith(n + '.') for n in pipe.QwenImage21Pipeline.adapter_target_names):
        for full, sub in mod.named_modules(prefix=name):
            if isinstance(sub, nn.Linear): targets.add(full)
print('3 lora targets:', len(targets), sorted(targets)[:6], '...')
assert 'txt_in.in_layer' in targets and 'img_in' in targets and 'transformer_blocks.0.attn.to_out.0' in targets and 'transformer_blocks.0.img_mlp.gate_up' in targets
lm = peft.get_peft_model(m, peft.LoraConfig(r=4, lora_alpha=4, target_modules=sorted(targets)))
layers = chain(m)
out3 = run_chain(layers, x, t, ctx2, mask2)
loss = ((out3 - torch.randn_like(out3)) ** 2).mean(); loss.backward()
nz = [(n, p.grad.abs().sum().item()) for n, p in m.named_parameters() if p.requires_grad]
zero = [n for n, g in nz if g == 0]
# step 1: lora_B == 0 so lora_A grads are exactly 0 by construction; every lora_B must see gradient
assert all('lora_A' in n for n in zero) and not any('lora_B' in n for n in zero), zero
print('3 step1: trainable', len(nz), 'lora_B all nonzero, lora_A zero as expected:', len(zero))
with torch.no_grad():
    for n, p in m.named_parameters():
        if p.requires_grad and 'lora_B' in n: p -= 1.0 * p.grad
m.zero_grad()
with torch.autograd.set_detect_anomaly(True):
    out3 = run_chain(chain(m), x, t, ctx2, mask2); ((out3 - torch.randn_like(out3)) ** 2).mean().backward()
zero2 = [n for n, p in m.named_parameters() if p.requires_grad and p.grad.abs().sum().item() == 0]
print('3 step2 zero-grad params:', zero2)
assert len(zero2) == 0

# (3b) show that comfy's original in-place residual fails under autograd (why the patch exists)
qm._gated_residual = orig_gated
m.zero_grad()
try:
    out4 = run_chain(chain(m), x, t, ctx2, mask2); ((out4) ** 2).mean().backward(); print('3b in-place original: backward unexpectedly OK')
except RuntimeError as e:
    print('3b in-place original raises:', str(e).splitlines()[0][:120])
qm._gated_residual = pipe._gated_residual_out_of_place

# (4) VAE detection + encode path (random weights): 4D RGB in -> (b, 64, h/16, w/16) out
import comfy.sd
import comfy.ldm.wan.vae2_2
ddconfig = {"dim": 16, "dec_dim": 16, "z_dim": 64, "dim_mult": [1, 2, 4, 8, 8], "num_res_blocks": 1, "attn_scales": [], "temperal_downsample": [False, True, True, True], "dropout": 0.0, "image_channels": 4, "patch_size": 1, "temporal_kernel": 1}
sd = {k: v.float() for k, v in comfy.ldm.wan.vae2_2.WanVAE(**ddconfig).state_dict().items()}
vae = comfy.sd.VAE(sd=sd)
print('4 vae: latent_dim', vae.latent_dim, 'channels', vae.latent_channels, 'down', vae.downscale_ratio, 'pad', vae.pad_channel_value, 'out_ch', vae.output_channels)
img = torch.rand(2, 32, 48, 3)
with torch.no_grad(): lat = vae.encode(img)
print('4 vae.encode', tuple(img.shape), '->', tuple(lat.shape))
assert tuple(lat.shape) == (2, 64, 2, 3)

# (5) edit path: one reference at slot S, compared to comfy's build_sequence/_forward per sample
def comfy_fwd(*a, **k):
    return m._forward(*a, **k) if dev.type=='cpu' else torch.autocast('cuda', dtype=torch.bfloat16)(m._forward)(*a, **k)
m = build(1)
layers = chain(m)
S = 3
for (h, w) in [(H, W), (4, 7)]:  # same size as the target (training case) and other parity on both axes
    ref_lat = torch.randn(B, C, h, w, device=dev)
    slot = torch.full((B,), S, dtype=torch.int64, device=dev)
    with torch.no_grad():
        out5 = run_chain(layers, x, t, ctx, mask, ref_lat, slot)
        for i in range(B):
            r = comfy_fwd(x[i:i+1], t[i:i+1], ctx[i:i+1], ref_latents=[ref_lat[i:i+1]], image_slots=[S])
            d = (out5[i:i+1] - r).abs().max().item()
            print(f'5 ref {h}x{w} unpadded sample {i} max|diff| =', d)
            assert d < TOL
        # padded batch (pads after the slot, at the end of text[slot:])
        out6 = run_chain(layers, x, t, ctx2, mask2, ref_lat, slot)
        for i, n in enumerate(lens):
            r = comfy_fwd(x[i:i+1], t[i:i+1], ctx2[i:i+1, :n], ref_latents=[ref_lat[i:i+1]], image_slots=[S])
            d = (out6[i:i+1] - r).abs().max().item()
            print(f'5 ref {h}x{w} padded sample {i} (len {n}) max|diff| =', d)
            assert d < TOL
        # the reference must actually condition the output
        out7 = run_chain(layers, x, t, ctx, mask, ref_lat + 1.0, slot)
        assert (out7 - out5).abs().max().item() > 1e-3
lm = peft.get_peft_model(m, peft.LoraConfig(r=4, lora_alpha=4, target_modules=sorted(targets)))
out8 = run_chain(chain(m), x, t, ctx2, mask2, torch.randn(B, C, H, W, device=dev), torch.full((B,), S, dtype=torch.int64, device=dev))
((out8 - torch.randn_like(out8)) ** 2).mean().backward()
zero8 = [n for n, p in m.named_parameters() if p.requires_grad and p.grad.abs().sum().item() == 0]
assert all('lora_A' in n for n in zero8) and not any('lora_B' in n for n in zero8), zero8
print('5 edit LoRA step1: lora_B all nonzero')
print('ALL OK')
