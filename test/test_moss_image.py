# Synthetic smoke test for models/moss_image.py — no model weights needed (tiny random DiT).
#   1. layer split (Initial/Transformer/FinalLayer) with the pipeline's patched Block/TextBlock/attention
#      == comfy's ORIGINAL MOSSImage._forward (unpadded), for both q/k/v layouts (fused_qkv False/True)
#   2. right-padded batch == per-sample unpadded comfy reference (pad-key mask + per-caption RoPE start)
#   3. LoRA backward through the split: every lora_B gets gradient at step 1, every lora_A at step 2;
#      LoRA targets cover the text refiner blocks (TextBlock) as well as the DiT blocks
#   4. conditioning tokenization through the fork's tokenize path keeps MOSS's chat template
# Run from the repo root with a venv that has torch/peft/deepspeed + the ComfyUI submodule requirements:
#   python test/test_moss_image.py            # CPU, exact
#   DEV=cuda python test/test_moss_image.py   # GPU, bf16 autocast (loose tolerance)
import os, sys
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
import comfy.ldm.moss_image.model as mm
# capture comfy's originals before the pipeline module monkeypatches them
orig = {n: getattr(mm, n) for n in ('attention', 'TextBlock', 'TextEmbedder', 'Block')}
from models import moss_image as pipe
patched = {n: getattr(mm, n) for n in orig}
assert all(orig[n] is not patched[n] for n in orig), 'patch not applied'

def use(classes):
    for n, v in classes.items(): setattr(mm, n, v)

def build(fused_qkv, classes, seed=0):
    use(classes)
    torch.manual_seed(seed)
    m = mm.MOSSImage(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=24, latent_channels=8, fused_qkv=fused_qkv,
                     dtype=torch.float32, device=dev, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1: p.fill_(1.0)
            else: p.normal_(0, 0.05)
    use(patched)
    return m.to(dev).float()

def reference(m, *a):
    # comfy's own forward with comfy's own (mask-less) attention
    use(orig)
    try:
        with torch.no_grad():
            return m._forward(*a) if dev.type == 'cpu' else torch.autocast('cuda', dtype=torch.bfloat16)(m._forward)(*a)
    finally:
        use(patched)

class Dummy:
    def wait_for_block(self, i): pass
    def submit_move_blocks_forward(self, i): pass

def chain(m):
    opts = {'total_blocks': len(m.layers), 'block_type': 'single'}
    return [pipe.InitialLayer(m)] + [pipe.TransformerLayer(b, i, opts, Dummy()) for i, b in enumerate(m.layers)] + [pipe.FinalLayer(m)]

def run_chain(layers, *inputs):
    for l in layers: inputs = l(inputs)
    return inputs

B, C, H, W, T = 2, 8, 5, 6, 7
TOL = 1e-4 if dev.type == 'cpu' else 5e-2  # GPU: bf16 autocast, comfy's own bf16-vs-fp32 floor is ~3e-2
x = torch.randn(B, C, H, W, device=dev); t = torch.rand(B, device=dev); ctx = torch.randn(B, T, 24, device=dev)
lens = [4, 7]
mask2 = torch.zeros(B, T, dtype=torch.bool, device=dev)
for i, n in enumerate(lens): mask2[i, :n] = True
ctx2 = ctx.detach().clone(); ctx2[0, 4:] = 123.0  # garbage in the pad slots must not leak

for fused in (False, True):
    m = build(fused, orig)
    # the patched classes must load comfy's state_dict layout unchanged
    mp = build(fused, patched, seed=1)
    mp.load_state_dict(m.state_dict())
    layers = chain(mp)

    # (1) unpadded: split (patched blocks) vs comfy reference (original blocks)
    ref = reference(m, x, t, ctx)
    with torch.no_grad():
        out = run_chain(layers, x, t, ctx, torch.ones(B, T, dtype=torch.bool, device=dev))
    d = (out - ref).abs().max().item()
    use(orig)
    with torch.no_grad(): floor = (m._forward(x, t, ctx) - ref).abs().max().item()  # fp32 comfy vs ref: bf16 noise floor on GPU
    use(patched)
    print(f'1 fused_qkv={fused} unpadded max|diff| =', d, 'ref scale', ref.abs().mean().item(), 'comfy fp32-vs-ref floor', floor)
    assert d < TOL

    # (2) padded batch vs unpadded single-sample references
    with torch.no_grad():
        out2 = run_chain(layers, x, t, ctx2, mask2)
    for i, n in enumerate(lens):
        d = (out2[i:i+1] - reference(m, x[i:i+1], t[i:i+1], ctx2[i:i+1, :n])).abs().max().item()
        print(f'2 fused_qkv={fused} padded sample {i} (len {n}) max|diff| =', d)
        assert d < TOL

# (2b) what upstream's rope (image at the PADDED text length) would have done to the short caption
with torch.no_grad():
    pe_up = mp.rope(H, W, T, dev)
    pe_fix = pipe.rope_per_caption(mp, H, W, T, mask2.sum(1), dev)
print('2b image-RoPE diff vs padded-length start: sample 0', (pe_up[0] - pe_fix[0]).abs().max().item(), 'sample 1', (pe_up[0] - pe_fix[1]).abs().max().item())
assert (pe_up[0] - pe_fix[1]).abs().max().item() == 0  # full-length caption: identical to comfy

# (3) LoRA backward through the split
import peft
targets = set()
for name, mod in mp.named_modules():
    if mod.__class__.__name__ in pipe.MossImagePipeline.adapter_target_modules:
        for full, sub in mod.named_modules(prefix=name):
            if isinstance(sub, nn.Linear): targets.add(full)
print('3 lora targets:', len(targets), sorted(targets)[:4], '...')
assert 'text_embedder.blocks.0.qkv' in targets and 'layers.0.proj.qkv' in targets and 'layers.2.proj.ffn_down' in targets
lm = peft.get_peft_model(mp, peft.LoraConfig(r=4, lora_alpha=4, target_modules=sorted(targets)))
out3 = run_chain(chain(mp), x, t, ctx2, mask2)
((out3 - torch.randn_like(out3)) ** 2).mean().backward()
nz = [(n, p.grad.abs().sum().item()) for n, p in mp.named_parameters() if p.requires_grad]
zero = [n for n, g in nz if g == 0]
assert all('lora_A' in n for n in zero) and not any('lora_B' in n for n in zero), zero
print('3 step1: trainable', len(nz), 'lora_B all nonzero, lora_A zero as expected:', len(zero))
with torch.no_grad():
    for n, p in mp.named_parameters():
        if p.requires_grad and 'lora_B' in n: p -= 1.0 * p.grad
mp.zero_grad()
with torch.autograd.set_detect_anomaly(True):
    out3 = run_chain(chain(mp), x, t, ctx2, mask2); ((out3 - torch.randn_like(out3)) ** 2).mean().backward()
zero2 = [n for n, p in mp.named_parameters() if p.requires_grad and p.grad.abs().sum().item() == 0]
print('3 step2 zero-grad params:', zero2)
assert len(zero2) == 0

# (4) tokenization: the fork's patched SD1Tokenizer.tokenize_with_weights must not bypass MOSS's template
import models.base  # applies the tokenizer patch
import comfy.text_encoders.moss_image
tok = comfy.text_encoders.moss_image.MOSSImageTokenizer()
ids = [x[0] for x in tok.tokenize_with_weights('a cat')[tok.clip_name][0]]
text = getattr(tok, tok.clip).tokenizer.decode(ids)
print('4 tokens', len(ids), repr(text))
assert text == '<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n', text
print('ALL OK')
