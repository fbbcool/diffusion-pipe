# CLAUDE.md — fbbcool/diffusion-pipe fork · agent-lora

This repo is the workspace of **agent-lora**, the LoRA R&D agent. Claude Code
running here **is** the agent. The repo is a fork of `tdrussell/diffusion-pipe`
used to train the **factored gts LoRA stack**; the sections below are the stable
distilled kickoff, while the evolving detail lives in this project's persistent
memory store (auto-loaded each session; seeded 2026-08-20 from the aitools
agent's LoRA-training memory).

## Scope

- **Fork maintenance** — patches, branches, upstream rebases in THIS checkout;
  cross-repo coordination with aitools (templater / installer wiring) via board
  tickets.
- **Training techniques** — rank / LR / masking / module-targeting decisions and
  experiments; the research-toolkit judgment calls.
- **Eval & keepers** — keeper-epoch selection, s_step, cross-seed ranking,
  strength calibration.
- **Stack doctrine** — guardian of the factored architecture (§2): base stays
  vanilla, substrates stay frozen, merging is delivery-only.

Out of scope (other agents own it): dataset assembly (agent-aidb), run execution
on rented boxes (agent-train, planned), captioning/curation and the aidb
libraries (aitools), inference nodes (comfyui-fbbcool-suite).

## Rules

- Reply in English; absolute clock times in CEST.
- **Git commits/pushes are the operator's** — prepare changes, never
  commit/stage unless explicitly asked.
- Never kill the ComfyUI process to free GPU — report contention and wait.
- **Content:** adult-only, **fictional characters only, never a real person**.
  Public artifacts (README, model cards, HF) describe LoRA roles at the
  architectural level ("base-prior refinement / prompt-following"), never the
  specific rendering concept.
- Memory hygiene: keeper verdicts, calibration results, and doctrine changes go
  into the memory store, not only into chat.
- System packages: stop and ask the operator with the exact install command —
  never vendor/build system deps from source.
- Final image-generation prompts → pipe to `wl-copy`.
- Board (Vikunja, the household ticket system): comment every time you touch a
  ticket; writes via the `vikunja_board` helper (creds in `./.env`). Onboarded
  2026-08-20 as `agent-lora` — agent card is task 57 (#13 in `agents`); member
  of `1xlasm` (6) + `agents` (8). Never poll — board work only when triggered.

---

## 1. This checkout is canonical

**Use `/home/misw/Repos/fbbcool/diffusion-pipe` for all patch/commit/push work.**
Tracks `git@github.com:fbbcool/diffusion-pipe.git` (SSH push rights).

Do **not** treat these other on-disk clones as authoritative:
- `/home/misw/Volumes/docker/workspace/diffusion-pipe` — docker bind mount,
  root-owned, public-rights remote (can't push). Pull from origin/main to sync.
- `/home/misw/venv/diffpipe/diffusion-pipe` — abandoned, pre-`merge_adapters`.
- `/home/misw/venv/finetune/diffusion-pipe` — upstream tdrussell, not the fork.

### Branch layout
- `feature-krea2` — **current.** krea2 model support (off `feature-xlasm-frozen`).
- `feature-xlasm-frozen` — single-file ComfyUI-format runtime_adapter support
  (reads safetensors directly, derives rank, builds synthetic `LoraConfig`,
  `add_adapter` + `load_state_dict`).
- `main` — tracks upstream-ish base.

`000_install/aitools.sh` `REPOS_TRAINER_BRANCH` selects which branch
`train_install` clones + which ComfyUI submodule commit gets pulled.

---

## 2. North star — the factored architecture

**"The entire stack is LoRAs, the base stays minimal forever."** (committed
2026-05-26.) This governs every "where should this concept live?" decision.

- Base model stays **vanilla** — foundational concepts are **never merged in**
  during development. Merging is reserved for delivery-time packaging only.
- Foundational concepts live as **permanent frozen `runtime_adapter`s**: loaded
  at BOTH training and inference time, structurally outside any later LoRA's
  trainable parameter space. **Train regime = inference regime, always.**
- New concepts train against `base + (all relevant prior LoRAs as frozen
  runtime_adapters at their inference strengths)`.

**Decision rule for a new LoRA:** own its own LoRA (never bundle) → smallest rank
that encodes the target deficit (default r4–8) → dataset/rank matched to the
layer's target signal → trained against base + relevant priors as runtime
adapters → lives in the library as a frozen runtime_adapter → collapse-merge only
for distribution.

### gts stack naming (fixed per layer)
| layer | name | rank | role |
|---|---|---|---|
| 0 | base (`snofs` on Qwen-Image; krea2-snofs for the krea2 track) | — | vanilla, never modified |
| 1 | `gts-atomic` | 4 | proportional substrate — `xlasm` differential-size trigger only |
| 2 | `gts-domain` | 32 (v3 ep31 keeper) | corrective residual / prompt-following refinement |
| 3 | `gts-app-<name>` | varies | sibling applications (identity, aesthetic, styling, …) |

`xlasm10` is **legacy** (monolithic r32); re-cast as factored LoRAs, don't build
new tracks against it.

---

## 3. LoRA training tracks — three tools, not interchangeable

Choose by the new LoRA's **relationship to the substrate/xlasm10**:

| relationship | tool (TOML) | placement |
|---|---|---|
| **child** — morphs existing behavior (body-type/prop variants) | `merge_adapters` in `[model]` | stacked with parent at inference |
| **sibling** — adds a feature orthogonal to the substrate (identity, style) | `runtime_adapters` in `[adapter]` | frozen at train + inference |
| **next generation** — replaces the substrate (xlasm v2/v3) | `init_from_existing` in `[adapter]` | standalone |

- **merge** rides the parent's low-rank pathway (rank-efficient morphs) but lets
  the child *erode* the parent — off-target drift is the tell.
- **runtime** write-protects the substrate ("stay away from my weights, search
  your own") — the fix for jezebeth-v1-style erosion. Costs more rank/data to
  learn cancellation in the child's own space.
- **init_from_existing** warm-starts trainable weights from a prior LoRA; rank
  must match (r32). **Rank-bump breaks it** — train from scratch if going r32→64.

### runtime_adapter = conditional residual (working theory, not proven)
A frozen runtime_adapter during training acts as a feature extractor; the
trainable LoRA learns the **residual** — substrate owns "**what**" (composition,
presence of concepts), trainable owns "**how well**" (rendering quality, defect
correction, non-overlap extension). Consequences:
- "Trainable underrepresents the substrate's central concept" is **gradient
  starvation, not failure** — the loss is already ~zero on the substrate's
  covered directions.
- "Trainable alone" (substrate off) is **OOD inference**, not standalone
  capability — don't read it as "what the LoRA knows."
- **Unit of composition is `(substrate, residual)` bundles trained together**,
  NOT arbitrary recombination of independently-trained weights. Can't swap
  substrates under a residual, can't stack independently-fit residuals cleanly.

### Rank vs base-freedom
Rank is **both** concept capacity **and** base-silencing. r4 preserves base
creativity (pose/composition variety); r32 silences it intrinsically —
independent of overcook. Goal is "atomic-style base freedom + targeted narrow
detail," **NOT** "match xlasm10 detail with factorization." Prefer multiple
narrow r4–8 LoRAs over one wide r32. The r8 gts-domain failure = rank-too-low
fragmenting across several residual axes (silenced base without adding clean
signal) — the fix is narrow-rank-on-narrow-**subset**, not more rank.

---

## 4. Resume + runtime_adapters behavior (operational)

`deepspeed --num_gpus=1 train.py --deepspeed --config <toml> --resume_from_checkpoint [run_id]`

- Runtime_adapters (frozen LoRAs) are **NOT in the checkpoint** — they're
  `requires_grad=False`, so DeepSpeed never serializes them. They're **reloaded
  from the TOML paths on every run start**. **The TOML is the source of truth.**
- Editing the TOML between original and resume **silently changes** the resumed
  run: removing a `[[adapter.runtime_adapters]]` entry runs without it (no
  error); changing `weight` applies the new weight (no warning); moving the file
  hard-errors. Changing trainable rank/target_modules fails on shape/key mismatch.
- Verify on every start: look for
  `[runtime_adapters] loading … as frozen adapter … active adapter list: ['default', …]`.
- For the factored stack: gts-atomic is correctly re-inserted + frozen on every
  resume as long as the TOML + safetensors path stay put. Don't move
  runtime_adapter files mid-run.

---

## 5. Masked training support (in the fork)

Landed as `edfb9ba`. Enables masked diffusion loss for narrow concept LoRAs.
- `mask_path = '/path/to/masks'` in the `[[directory]]` block of the dataset TOML.
- Masks matched to source images by **stem** (any ext); R channel → weight
  (white=1 train, black=0 ignore, grey=fractional).
- Loss math: `models/base.py:540` (BasePipeline) and `base.py:871` (ComfyPipeline)
  — `loss *= mask`, then `.mean()`; `models/qwen_image.py:566-570` handles
  latent-res broadcast + packing.
- Missing mask → warns + skips that image. Optional `default_mask_file`.
- **No code work to use it** — drop PNG masks in the dir, set `mask_path`, train.

`target_modules` selection is a **class-level constant** per model
(`adapter_target_modules`, e.g. `['QwenImageTransformerBlock']`), walked in
`configure_adapter` (`base.py:436-439` BasePipeline, `720-723` ComfyPipeline).
NOT TOML-exposed — late-block-only targeting needs a ~10-30
line patch to filter `target_linear_modules` by a regex/block-index field.

---

## 6. Model ports

### krea2 — task #90, IMPLEMENTED, smoke-testing
The fork **vendors ComfyUI as a submodule** (`submodules/ComfyUI`); several models
(`z_image`, `ernie_image`, `flux2`) are `ComfyPipeline` subclasses that reuse
comfy's native model/VAE/text-encoder loaders. krea2 follows this — **no
diffusers re-impl, no weight remap**.

- `models/krea2.py` — `Krea2Pipeline(ComfyPipeline)`. SingleStreamDiT split into
  InitialLayer / per-block TransformerLayer / FinalLayer. Flow-matching like
  flux2 (`target = noise - latents`, `t∈[0,1]` → comfy `timestep_embedding`
  applies ×1000). `shift = 1.15`.
- **Latent space: Qwen-Image VAE, 16-channel, Wan2.1 latent format** — 3D VAE,
  kept 5D `(b,c,1,h,w)` through `process_latent_in`, squeezed to 4D after. This
  is fixed by the architecture: **do not swap to the Wan2.2 VAE (48-channel) —
  it's a different latent space and the DiT projections are shaped for 16ch.**
  `qwen_image_vae.safetensors` is the only usable VAE (confirmed vs ComfyUI +
  Comfy-Org/Krea-2). (Note: "Krea 2" here ≠ FLUX.1-Krea-dev, which uses the Flux
  `ae.safetensors` — different model family.)
- **LoRA target modules MUST include the text-fusion path:**
  `['SingleStreamBlock', 'TextFusionTransformer']` + a `configure_adapter`
  override name-matching `txtmlp` (commit `c7c0fd4`). Blocks-only training learns
  loss-wise but produces a **dead LoRA** (overfits a frozen text representation,
  no concept transfer). Verify a trained LoRA contains `diffusion_model.txtfusion.*`.
- Training base is **plain e4m3 fp8** (`krea2-raw-snofs0.75-fp8.safetensors`, NOT
  the scaled inference variant); template sets `diffusion_model_dtype='float8'`,
  keeping `first/tmlp/tproj/txtfusion/txtmlp/last` + norms in bf16.
- LR: use **5e-5** (the 2e-4 in older templates is too high). raw = train base,
  turbo = inference base (raw→turbo transfer is fine; snofs proves it).
- **Frozen `runtime_adapters` are supported** — the machinery was extracted from
  `BasePipeline` into module-level `load_runtime_adapters()` (models/base.py) and
  is called by `ComfyPipeline.configure_adapter` and krea2's override (so frozen
  LoRAs match txtfusion/txtmlp keys too). Same TOML syntax as qwen_image
  (`[[adapter.runtime_adapters]]`). Point `path` at the `.safetensors` file
  directly; an explicit file is always parsed as single-file ComfyUI format,
  even inside an `epochN/` dir with `adapter_config.json` next to it.
- Consider adopting upstream `tdrussell` krea2.py wholesale later (cleaner 5D
  latent handling + LoKr) — caveat: its `prepare_inputs` skips `process_latent_in`
  (relies on the base.py VAE fn), so check that against this fork's base.py first.

### ideogram4 — parked
Single-stream DiT 34L, flow-matching, FP8 + NF4 open weights, Qwen3-VL-8B text
encoder (13-layer concat), dual conditional+unconditional transformer. Template:
`models/qwen_image.py`. Target FP8 (H100 native, quality). **Two go/no-go gates
before any port work:** (1) vanilla inference smoke-test must show *materially
better* gts priors vs Qwen-snofs on the stress prompts; (2) willingness to spend
2–3 months re-training the whole stack (base-locked, nothing transfers).
Non-commercial license. ~600–1000 lines. Check upstream for existing adapter code
first.

---

## 7. Hyperparameter defaults (Qwen-Image LoRA-on-pretrained)

These are project-tuned defaults, not from-scratch rules. **Do not auto-change
LR/batch when hardware changes.**

- **LR — no linear scaling with batch size.** Cold-start anchor **5e-5–7e-5**;
  warm-start (`init_from_existing`) anchor ~1e-4; 2e-4 = upper edge for
  warm-start; ≥4e-4 = "expect cosmetic artifacts," never for keepers. The parent
  is the most LR-sensitive node — corruption propagates to all children via
  `init_from_existing`.
- **Batch sweet spot is U-shaped, not "biggest that fits":** **8 on H100 NVL, 2
  on 5090** (4 at rank ≤8). Bigger batch = lower-variance grad but fewer
  steps/epoch, and you can't compensate with LR.
- **GPU pairing:** layer-3 gts-app + small substrates → **5090** (H100 NVL only
  ~30% faster, not worth the rental). gts-domain (r64, ~1000 imgs) → **H100 NVL**.
  - H100 NVL: `micro_batch_size_per_gpu=8`, `activation_checkpointing=false`,
    `compile=true`.
  - 5090: `micro_batch_size_per_gpu=2`, `activation_checkpointing=true`,
    `compile=true` — 32 GB has NO headroom to disable checkpointing (OOMs → batch 1).
- **Overcook tells:** knees redder/pinker than surrounding skin (also
  elbows/nipples), and a "plasticky" flattened skin micro-texture. First
  hypothesis when a LoRA "feels off" — cheaper to test than rank-up/more-data.
- **Rented GPU OOM:** check baseline reservation first (`nvidia-smi
  --query-gpu=memory.used,memory.free`; if used but no compute-apps → platform
  reserved 3–4 GiB, unrecoverable). Switch rentals if it eats >5-10%; don't
  redesign ZeRO/rank.
- Keeper prediction: epoch ≈ 25 × (batch / 2) at the same LR (preserves total
  step count, not image-views). gts-atomic v1 ref: batch 2, 225 imgs, 5e-5 →
  keeper ep25.

---

## 8. Eval methodology

- **Strip size/height cues from diagnostic prompts.** "looking up at her", "at
  her feet", "towering over", relative-size words ("smaller/bigger") all
  solo-induce the size differential via base-model perspective priors — they
  confound whether the *LoRA* learned it. Use neutral phrasing ("a xlgts woman
  standing next to a xlasm man"). Leaky cues are fine for showcase gens, never
  for progress diagnosis or A/B checkpoint comparison.
- **s_step diagnostic** (rank-4-8 atomic LoRAs with a discrete toggle): track the
  strength threshold where the concept abruptly activates. Target with-trigger
  s_step **0.7–0.9**. Caveat: at fixed seed s_step is noisy (4× spread across
  seeds) — for r32+ or substrate-supported LoRAs use **cross-seed ranking** on a
  stress-test prompt, not single-seed extrapolation.

---

## 9. Rejected / research

- **FFT rejected (2026-06-08):** voids the factored north star, exceeds
  single-H100 VRAM (~240 GB full / ~100 GB 8bit-Adam), needs a
  general-distribution anti-forgetting mix we don't have. Uniquely fixes the
  pixel-budget bottleneck — but do NOT re-propose until reconstruction-LoRA AND
  DPO-on-LoRA both plateau and infra/dataset/stack-reset are all committed.
- **Research toolkit** for going past atomic without rank-32 base-silencing:
  **Concept Sliders** preservation-loss (train invariance on preservation
  prompts — the highest-EV next experiment), **OPLoRA** orthogonal-complement
  gradient projection (2-line optimizer add; atomic's SVD is tiny), **Mix-of-Show**
  spatial gating at inference (detail LoRA active only on small-subject latents).
  Skepticism filter — do NOT pursue for this goal: LoKr/LoHa/LyCORIS, VeRA,
  ZipLoRA, Chain-of-LoRA, DoRA (single-LoRA only), BOFT (breaks A@B). All buy
  *expressive rank* or *post-hoc composition*, both anti-patterns here.

---

## 10. Codebase map & commands

### Commands
```bash
# Train (DeepSpeed is mandatory — the whole script is built on its pipeline
# parallelism, even for --num_gpus=1):
deepspeed --num_gpus=1 train.py --deepspeed --config /path/to/config.toml
# RTX 4000-series needs: NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
# VRAM-tight runs: prepend PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Resume (config on the CLI is the source of truth, see §4):
#   ... --resume_from_checkpoint [run_id]

# Caching flags: --cache_only (cache then exit), --regenerate_cache,
# --trust_cache (skip fingerprint check on huge datasets).

# After pulling: git submodule update   (ComfyUI + model submodules move)
```
There is no test suite or linter. `test/` is a DeepSpeed init debug script;
`tools/` holds one-off VAE round-trip sanity scripts.

### Architecture
Two TOML files drive everything: the main config (`[model]`, `[adapter]`,
`[optimizer]`, top-level train settings) and the dataset config it points to
(`[[directory]]` blocks, resolutions, AR bucketing). `examples/main_example.toml`
and `examples/dataset.toml` are the documented references;
`docs/supported_models.md` documents per-model options.

- **`train.py`** — single entry point. Dispatches on `model.type` via an
  if/elif chain (~line 304) to a pipeline class, builds a DeepSpeed
  `PipelineEngine` from the model's layer list, runs the epoch/step loop,
  eval, and Tensorboard logging. `utils/patches.py` monkeypatches are applied
  at startup.
- **`models/base.py`** — the two base classes every model subclasses:
  - `BasePipeline` — diffusers/HF-loader models (qwen_image, flux, wan, …).
  - `ComfyPipeline` — models loaded through the vendored `submodules/ComfyUI`
    native loaders (z_image, ernie_image, flux2, **krea2**). No weight remap.
  - The subclass contract: `load_diffusion_model` / `get_vae` /
    `get_text_encoders` / `configure_adapter` / `prepare_inputs` (builds the
    flow-matching target from cached latents+embeddings) / `to_layers` (splits
    the transformer into a layer list for pipeline parallelism) /
    `get_loss_fn` / `save_adapter`. Adding a model = new subclass + an elif in
    train.py + example config.
  - Fork-specific machinery lives here too: `runtime_adapters` loading/freezing
    (module-level `load_runtime_adapters`, `base.py:166`, called from both
    classes' `configure_adapter`) and the masked-loss math (§5).
    `merge_adapters` is in `models/qwen_image.py`.
- **`utils/dataset.py`** — pre-caching + data feeding. VAE latents and text
  embeddings are cached to disk via HF Datasets (a `cache/` dir inside each
  dataset directory) *before* training, so VAE/text encoders are never resident
  during training (hence: no text-encoder LoRA support). Aspect-ratio and size
  bucketing (`ARBucketDataset`/`SizeBucketDataset`), `DatasetManager`
  orchestrates multi-GPU caching, `PipelineDataLoader` feeds the engine and
  carries resumable dataloader state.
- **`utils/saver.py`** — two distinct outputs per run dir: DeepSpeed
  checkpoints (`global_stepN/`, full training state, not usable for inference)
  and saved LoRAs (`epochN/`, safetensors + PEFT config + the run's TOML).
- **`optimizers/`** — custom optimizers (`AdamW8bitKahan`, automagic,
  GenericOptim) selected by `[optimizer] type`.
- **`utils/pipeline.py`** — `ManualPipelineModule`, the DeepSpeed
  `PipelineModule` subclass that consumes `to_layers()` output.

---

## Upstream README

The original tdrussell/diffusion-pipe usage docs remain in `README.md` and
`docs/`. This file is additive project context, not a replacement.
