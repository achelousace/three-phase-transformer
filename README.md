# Three-Phase Transformer

![alt text](paper/figs/into.png)

**A Residual-Stream Structural Prior for Decoder-Only Transformers**

Mohammad R. Abu Ayyash - [Brains Build Research](https://github.com/achelousace), Ramallah, Palestine.

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](Paper/three_phase_paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Three-Phase Transformer (3PT) is a residual-stream structural prior for decoder-only Transformers. The d_model hidden vector is partitioned into 3 equally-sized cyclic channels ("phases A/B/C"), each maintained by a small number of phase-respecting operations scattered through every block. The channel partition geometrically carves out a one-dimensional DC subspace orthogonal to the channels, into which a fixed Gabriel's horn profile r(p) = 1/(p+1) is injected as an absolute-position side-channel that composes orthogonally with RoPE's relative-position rotation in attention. The architecture is a self-stabilizing equilibrium between scrambling (attention, FFN) and re-imposition (the phase-aware ops), not a bolted-on module.

At 123M parameters on WikiText-103, 3PT achieves a **−7.20% perplexity reduction (−2.62% BPB)** over a matched RoPE-Only baseline at **+1,536 trainable parameters (0.00124% of total)**, with a **1.93× step-count convergence speedup** (1.64× wall-clock, after a 17% per-step overhead). Every other modification beyond the rotation thetas is parameter-free or parameter-neutral.

## Architecture

3PT adds five coordinated structural modifications on top of a standard SwiGLU + RMSNorm + RoPE + GQA backbone:

**1. Three-phase channel partition** - The d_model residual stream is split into 3 contiguous equal-width stripes interpreted as 120°-offset components in the three-phase AC sense.

**2. Gabriel's horn DC injection** - At every forward pass, the embedding's per-position cross-phase DC mean is computed across the three phases, subtracted, and replaced by the value of a fixed analytic profile r(p) = 1/(p+1). Non-learnable buffer; zero trainable parameters. The 1D tunnel opened by the channel partition becomes an absolute-position side-channel orthogonal to where content lives.

**3. PhaseRotationLayer** - Inserted between attention and FFN inside every block, **non-residually**. Each layer holds a learnable parameter theta of shape `[d_phase/2]`, initialized to a depth-linear schedule θ_i = (i+1)·π/(2L). At forward time, each phase is rotated independently by its own theta + i·(2π/3) offset using a 2D Givens rotation. Because the layer is an orthogonal map, gradients flow through without attenuation or amplification.

**4. Phase-aligned GQA** - GQA is configured so that each attention head slice lies entirely within a single phase. At 123M: n_q = 12, n_kv = 3 (4 Q heads + 1 KV head per phase). Configuration constraint, not a separate mechanism. Adds zero parameters.

**5. PhaseAwareRMSNorm** - Replaces global RMSNorm everywhere it appears with three independent `RMSNorm(d_phase)` instances applied to the three phases and concatenated. Identical total parameter count to a single `RMSNorm(d_model)`.

![alt text](paper/figs/architecture_diagram.png)

## Key Results

**(123M on WikiText-103, 30k steps, seed 42):**

| Model | Final PPL | Final BPB | Params | Time |
|---|---|---|---|---|
| RoPE-Only Vanilla 123M | 17.31 | 1.1148 | 123,489,024 | 6,636s |
| **Three-Phase Transformer 123M** | **16.06** | **1.0855** | **123,490,560** | **7,777s** |
| Δ | **−7.20%** | **−2.62%** | +1,536 (+0.00124%) | +17.2% |

Convergence: 3PT reaches PPL 17.45 at step 14,000; RoPE-Only does not reach 17.45 until step 27,000 - a **1.93× step-count speedup** or **1.64× wall-clock speedup** after accounting for per-step overhead.

**Horn-orthogonality witness:** When the horn is active, the cross-phase residual at every eval is mathematically pinned at exactly NUM_PHASES × mean(horn) = 3 · H_1024 / 1024 ≈ 0.0220, matching the analytic value to 6 decimal places at every checkpoint. The cleanest possible empirical proof that the horn lives in a 1D subspace orthogonal to the three-phase decomposition.

**U-shaped depth profile of rotation drift (12 layers, seed 42):** Min L2 drift at block 2 (0.069), max at block 11 (1.833). The depth-linear schedule overshoots aggression in late blocks and undershoots in early blocks; the implied optimal schedule is sub-linear with a slight S-curve.

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA, ≥48 GB VRAM
- Auto-installed on first run: `transformers`, `datasets`, `tqdm`, `numpy`

### Run

```bash
python ThreePhaseTransformer123M.py
```

Trains 3PT 123M on WikiText-103-raw-v1 with the Llama-2 BPE tokenizer, seq_len 1024, effective batch 32 sequences = 32,768 tokens/step, 30,000 steps, cosine LR with 500 warmup, seed 42. Expected final BPB: 1.0855 (PPL 16.06). Runtime ~2.2 hours on Colab G4.

The first run pre-tokenizes WikiText-103 to a uint16 `.bin` file and memmaps it for training. Checkpoints every 500 steps, keeps the most recent 3, resumes automatically if re-run.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `D_MODEL` | 768 | Hidden dimension (must be divisible by 3) |
| `D_FF` | 2048 | SwiGLU intermediate dimension |
| `N_LAYERS` | 12 | Transformer blocks |
| `N_Q_HEADS` | 12 | Query heads (phase-aligned: 4 per phase) |
| `N_KV_HEADS` | 3 | KV heads (phase-aligned: 1 per phase) |
| `SEQ_LEN` | 1024 | Training context |
| `BATCH_SIZE` | 8 | Per-device batch |
| `GRAD_ACCUM` | 4 | Effective batch 32 sequences |
| `TRAIN_STEPS` | 30000 | Optimizer steps |
| `WARMUP_STEPS` | 500 | Linear warmup |
| `LR` | 3e-4 | Peak learning rate (cosine decay to 10%) |
| `WEIGHT_DECAY` | 0.1 | AdamW weight decay |
| `BETA1, BETA2` | 0.9, 0.95 | AdamW momentum |
| `GRAD_CLIP` | 1.0 | Gradient clipping norm |
| `NUM_PHASES` | 3 | Cyclic Z_N partition (architectural invariant) |

## Hardware

Validated on Colab G4 with NVIDIA RTX Pro 6000 Blackwell (96 GB VRAM). Minimum viable: any CUDA GPU with ≥48 GB VRAM (with bf16 + Flash Attention 2 via SDPA).

## Citation

If you use this work, please cite:

```bibtex
@article{abuayyash2026threephase,
  title   = {Three-Phase Transformer},
  author  = {Abu Ayyash, Mohammad R.},
  journal = {arXiv preprint arXiv:2604.14430},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.14430}
}
```

Paper: https://arxiv.org/abs/2604.14430
Code: https://github.com/achelousace/three-phase-transformer

## License

[MIT License](LICENSE) - Copyright (c) 2026 Mohammad Abu Ayyash
