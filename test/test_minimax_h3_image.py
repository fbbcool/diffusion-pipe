# Synthetic smoke test for models/minimax_h3_image.py — no model weights needed (tiny random MiniMaxH3ImageModel).
#   1. layer split (Initial/Transformer/FinalLayer) with the fork's patched H3 blocks == comfy's ORIGINAL
#      MiniMaxH3ImageModel._forward, per PDD head (comfy selects the head interval from the sampler's sigma schedule,
#      we pass the grid index), for both AdaLN layouts (time embedder / adaln curve table)
#   2. right-padded batch (different caption lengths + different sigmas per sample) == per-sample unpadded comfy
#      reference (pad keys masked, refiner on unpadded tokens, per-caption RoPE start, per-sample PDD head)
#   3. prepare_inputs draws only grid sigmas, honours pdd_step_weights, timestep_quantile -> grid index
#   4. LoRA backward through the split: every lora_B gets gradient at step 1, every lora_A at step 2; targets cover
#      DiTBlock + RefinerBlock linears, never adaln / final_layer
# Run from the repo root with a venv that has torch/peft/deepspeed + the ComfyUI submodule requirements:
#   python test/test_minimax_h3_image.py            # CPU, exact
#   DEV=cuda python test/test_minimax_h3_image.py   # GPU, bf16 autocast (loose tolerance)
import os, sys, types
FORK = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
COMFY = os.environ.get('COMFY', os.path.join(FORK, 'submodules', 'ComfyUI'))
sys.path.insert(0, FORK)
import utils.common  # fork's utils is a namespace pkg; import it before ComfyUI's regular utils/ pkg lands on sys.path
sys.path.insert(0, COMFY)
import torch
from torch import nn
torch.manual_seed(0)

dev = torch.device(os.environ.get('DEV', 'cpu'))

from comfy import model_management
model_management.in_training = True
import comfy.ops
import comfy.ldm.minimax.model as mm
PATCHED_NAMES = ('Attention', 'AdalnProj', 'DiTBlock', 'FinalLayer', '_mod_scale_shift', '_mod_gate', 'unpack_audio')
orig = {n: getattr(mm, n) for n in PATCHED_NAMES}
from models import minimax_h3_image as pipe
from models.minimax_h3 import MinimaxH3Pipeline
patched = {n: getattr(mm, n) for n in PATCHED_NAMES}
assert all(orig[n] is not patched[n] for n in PATCHED_NAMES), 'patch not applied'

GRID = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
N = len(GRID) - 1
C = 8  # latent channels

def use(classes):
    for n, v in classes.items(): setattr(mm, n, v)

def build(classes, curves, seed=0):
    use(classes)
    torch.manual_seed(seed)
    m = mm.MiniMaxH3ImageModel(
        pdd_sigmas=GRID, pdd_steps=N, hidden_size=64, num_layers=3, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=32, ffn_hidden_size=96, latents_dim=C, text_dim=40,
        timestep_input_dim=32, time_embed_hidden_size=48, time_embed_dim=32, rope_inv_freq_len=4,
        adaln_curve_grid=17 if curves else None, dtype=torch.float32, device=dev, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1 and ('norm' in n): p.fill_(1.0)
            else: p.normal_(0, 0.05)
        m.rope.inv_freq.copy_(1.0 / (100.0 ** (torch.arange(4, dtype=torch.float32) / 4)))
        if curves:
            m.adaln_t_table.normal_(0, 1.0)
    use(patched)
    return m.to(dev).float()

def reference(m, x, sigma, ctx):
    # comfy's own forward with comfy's own blocks; the PDD interval comes from the sampler schedule (= the grid)
    use(orig)
    try:
        opts = {'sample_sigmas': torch.tensor(GRID)}
        with torch.no_grad():
            f = m._forward if dev.type == 'cpu' else torch.autocast('cuda', dtype=torch.bfloat16)(m._forward)
            return f(x, torch.tensor([sigma * 1000.0], device=dev), ctx, transformer_options=opts)
    finally:
        use(patched)

class Dummy:
    def wait_for_block(self, i): pass
    def submit_move_blocks_forward(self, i): pass

def split_forward(m, x, t, head_index, ctx, mask, lens):
    layers = [pipe.InitialLayer(m)] + [pipe.TransformerLayer(b, i, Dummy()) for i, b in enumerate(m.blocks)] + [pipe.FinalLayer(m)]
    out = (x, t, head_index, ctx, mask, lens)
    for l in layers:
        out = l(out)
    return out

tol = 1e-4 if dev.type == 'cpu' else 5e-2

def check(name, a, b):
    err = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-8)
    print(f'{name}: rel max err {err:.2e}')
    assert err < tol, name

for curves in (False, True):
    tag = 'curves' if curves else 'time_embedder'
    m_ours = build(patched, curves)
    m_ref = build(orig, curves)
    m_ref.load_state_dict(m_ours.state_dict())
    m_ref.to(dev).float()

    # --- 1. every PDD head, unpadded, batch 1
    lat = torch.randn(1, C, 1, 6, 8, device=dev)
    ctx = torch.randn(1, 7, 40, device=dev)
    for k in range(N):
        out = split_forward(m_ours, lat, torch.tensor([GRID[k]], device=dev), torch.tensor([k], dtype=torch.int32, device=dev),
                            ctx, torch.ones(1, 7, dtype=torch.bool, device=dev), torch.tensor([7], dtype=torch.int32))
        check(f'[{tag}] head {k} == comfy', out.detach(), reference(m_ref, lat, GRID[k], ctx))

    # --- 2. padded batch, different caption lengths and sigmas
    lens = [5, 9]
    ks = [1, 4]
    lat = torch.randn(2, C, 1, 6, 8, device=dev)
    ctxs = [torch.randn(1, n, 40, device=dev) for n in lens]
    L = max(lens)
    ctx_pad = torch.cat([torch.cat([c, torch.zeros(1, L - c.shape[1], 40, device=dev)], dim=1) for c in ctxs])
    mask = torch.stack([torch.arange(L, device=dev) < n for n in lens])
    out = split_forward(m_ours, lat, torch.tensor([GRID[k] for k in ks], device=dev), torch.tensor(ks, dtype=torch.int32, device=dev),
                        ctx_pad, mask, torch.tensor(lens, dtype=torch.int32))
    for i in range(2):
        check(f'[{tag}] padded batch sample {i} (len {lens[i]}, head {ks[i]}) == unpadded comfy',
              out[i:i + 1].detach(), reference(m_ref, lat[i:i + 1], GRID[ks[i]], ctxs[i]))

# --- 3. prepare_inputs timestep draw
p = types.SimpleNamespace(pdd_sigmas=torch.tensor(GRID[:-1]), pdd_step_weights=torch.tensor([0, 0, 1.0, 0, 0, 1.0]),
                          get_conds=lambda inputs: MinimaxH3Pipeline.get_conds(None, inputs))
inputs = {'latents': torch.randn(64, C, 1, 4, 4), 'mask': None,
          'text_embeds_0': [torch.randn(3, 40)] * 64, 'attention_mask_0': [torch.ones(3)] * 64}
(noisy, t, hi, ctx, cmask, lens), (target, lmask) = pipe.MinimaxH3ImagePipeline.prepare_inputs(p, inputs)
assert set(hi.tolist()) == {2, 5}, hi
assert torch.allclose(t, torch.tensor(GRID)[hi.long()])
assert lmask.numel() == 0 and lens.tolist() == [3] * 64
(_, t, hi, *_), _ = pipe.MinimaxH3ImagePipeline.prepare_inputs(p, inputs, timestep_quantile=0.99)
assert set(hi.tolist()) == {5}
print('prepare_inputs: grid-only sigmas, weights + quantile OK')

# --- 4. LoRA gradients through the split
from peft import LoraConfig, get_peft_model
m = build(patched, curves=True, seed=1)
dummy = types.SimpleNamespace(adapter_target_modules=pipe.MinimaxH3ImagePipeline.adapter_target_modules)
targets = MinimaxH3Pipeline.get_target_modules(dummy, m)
assert targets and not any('adaln' in t or 'final_layer' in t for t in targets)
assert any(t.startswith('token_refiner.') for t in targets) and any(t.startswith('blocks.') for t in targets)
m = get_peft_model(m, LoraConfig(r=4, lora_alpha=4, target_modules=targets, init_lora_weights=True)).base_model.model
lora = {n: p for n, p in m.named_parameters() if 'lora_' in n}
opt = torch.optim.SGD(lora.values(), lr=1e-2)
x0 = torch.randn(2, C, 1, 6, 8, device=dev)
for step in (1, 2):
    opt.zero_grad()
    out = split_forward(m, x0, torch.tensor([GRID[0], GRID[3]], device=dev), torch.tensor([0, 3], dtype=torch.int32, device=dev),
                        ctx_pad, mask, torch.tensor(lens, dtype=torch.int32))
    out.square().mean().backward()
    kind = 'lora_B' if step == 1 else 'lora_A'
    dead = [n for n, p in lora.items() if kind in n and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not dead, f'step {step}: no grad for {dead[:5]}'
    opt.step()
print(f'LoRA: {len(targets)} target linears, all lora_B (step 1) / lora_A (step 2) receive gradient')
print('ALL OK')
