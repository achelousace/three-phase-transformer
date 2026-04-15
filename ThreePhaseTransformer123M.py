"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Three-Phase Transformer (123M)                                          ║
║  Author: Mohammad R. Abu Ayyash - Brains Build Research, Ramallah        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, math, time, json, random, glob
import numpy as np

# T4 fragmentation fix - must be set BEFORE torch is imported
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm.auto import tqdm

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42

# Model dimensions (123M scale)
D_MODEL      = 768              # must be divisible by NUM_PHASES (3)
D_FF         = 2048
N_LAYERS     = 12
N_Q_HEADS    = 12               # must be divisible by NUM_PHASES (4 Q per phase)
N_KV_HEADS   = 3                # one KV head per phase, 4:1 GQA ratio
DROPOUT      = 0.0              
SEQ_LEN      = 1024             
MAX_SEQ_LEN  = 2048             # RoPE + mask buffer capacity, > SEQ_LEN for generation
BATCH_SIZE   = 8                
GRAD_ACCUM   = 4                # effective batch 8 x 4 = 32 sequences
LR           = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
BETA1, BETA2 = 0.9, 0.95

# Training schedule
TRAIN_STEPS  = 30000            # optimizer steps
EVAL_EVERY   = 1000
WARMUP_STEPS = 500
LOG_EVERY    = 50               # print train metrics every N steps
CKPT_EVERY   = 500              # rolling checkpoint every 500 steps (keep 3)
CKPT_KEEP    = 3                # number of recent checkpoints to retain
EVAL_BATCHES = 50               # val batches per eval

# Three-phase invariants
NUM_PHASES   = 3
ROPE_BASE    = 10000.0

# Dataset / tokenizer
DATASET_NAME     = "wikitext"
DATASET_CONFIG   = "wikitext-103-raw-v1"
DATASET_TEXT_KEY = "text"
STREAM_SHUFFLE   = True          # False = read rows sequentially 
STREAM_SHUFFLE_BUFFER = 10000
TOKENIZER_NAME   = "NousResearch/Llama-2-7b-hf"  # Llama-2 byte-level BPE

# Paths / checkpointing
RUN_NAME         = "threephase_123m_wikitext103"
OUT_DIR          = f"./runs/{RUN_NAME}"
CKPT_PATH        = f"{OUT_DIR}/checkpoint.pt"
BEST_PATH        = f"{OUT_DIR}/best.pt"
METRICS_PATH     = f"{OUT_DIR}/metrics.jsonl"
RESULTS_PATH     = f"{OUT_DIR}/results.json"
RESUME           = True          # resume from CKPT_PATH if it exists

# Mixed precision
USE_AMP          = True
AMP_DTYPE        = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ════════════════════════════════════════════════════════════════════════
#  DATA — streaming HuggingFace dataset + Llama/Gemma-style BPE tokenizer
# ════════════════════════════════════════════════════════════════════════

def load_tokenizer():
    """
    Loads a byte-level BPE tokenizer (Llama-2 style) via HuggingFace.
    Falls back to gpt-neox-20b if the primary mirror is unreachable.
    Returns (tokenizer, vocab_size, eos_id, pad_id).
    """
    from transformers import AutoTokenizer
    candidates = [TOKENIZER_NAME, "EleutherAI/gpt-neox-20b"]
    last_err = None
    for name in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|pad|>"
            eos_id = tok.eos_token_id if tok.eos_token_id is not None else 0
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
            print(f"  [Tokenizer] Loaded {name} | vocab={tok.vocab_size:,} | eos={eos_id} | pad={pad_id}")
            return tok, tok.vocab_size, eos_id, pad_id
        except Exception as e:
            last_err = e
            print(f"  [Tokenizer] Failed to load {name}: {e}")
    raise RuntimeError(f"Could not load any tokenizer. Last error: {last_err}")


def stream_dataset(split):
    """
    Opens a streaming HuggingFace dataset. When STREAM_SHUFFLE is False, rows
    are yielded in file order — useful for appending more data without re-
    shuffling the entire corpus.
    """
    from datasets import load_dataset
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split,
        streaming=True,
    )
    if STREAM_SHUFFLE and split == "train":
        ds = ds.shuffle(seed=SEED, buffer_size=STREAM_SHUFFLE_BUFFER)
    return ds


class StreamingTokenDataset(IterableDataset):
    """
    Streams rows from a HuggingFace dataset, tokenizes them on the fly with
    a BPE tokenizer, concatenates tokens across rows with an EOS separator,
    and yields fixed-length (x, y) sequence pairs of length SEQ_LEN.
    Supports both shuffled and strictly-sequential streaming.
    """
    def __init__(self, split, tokenizer, seq_len, eos_id, max_rows=None):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.max_rows = max_rows

    def __iter__(self):
        ds = stream_dataset(self.split)
        # Shard the stream across DataLoader workers so each worker gets a
        # disjoint slice (otherwise all workers would emit the same batches).
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        buf = []
        rows_seen = 0
        for row in ds:
            text = row.get(DATASET_TEXT_KEY, "")
            if not text or not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            buf.extend(ids)
            buf.append(self.eos_id)
            rows_seen += 1
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf = buf[self.seq_len + 1:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
            if self.max_rows is not None and rows_seen >= self.max_rows:
                break


# ════════════════════════════════════════════════════════════════════════
#  PRE-TOKENIZED .BIN PIPELINE (FAST PATH FOR TRAINING)
# ════════════════════════════════════════════════════════════════════════

def prepare_bin(split, tokenizer, eos_id, out_path, max_rows=None):
    """
    Stream the HuggingFace dataset, tokenize each row, and write a flat
    uint16 .bin file of concatenated token ids with EOS separators. Skips
    if the file already exists.
    """
    if os.path.exists(out_path):
        nbytes = os.path.getsize(out_path)
        ntokens = nbytes // 2
        print(f"  [bin] {split}: found existing {out_path} ({ntokens:,} tokens, "
              f"{nbytes / 1e6:.1f} MB)")
        return ntokens

    print(f"  [bin] {split}: tokenizing to {out_path} ...")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ds = stream_dataset(split)
    # Buffer chunks of ~10M tokens before writing, to amortize file IO
    chunk_buf = []
    chunk_limit = 10_000_000
    total = 0
    rows_seen = 0
    t0 = time.time()
    with open(out_path, "wb") as f:
        for row in ds:
            text = row.get(DATASET_TEXT_KEY, "")
            if not text or not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            chunk_buf.extend(ids)
            chunk_buf.append(eos_id)
            rows_seen += 1
            if len(chunk_buf) >= chunk_limit:
                arr = np.array(chunk_buf, dtype=np.uint16)
                f.write(arr.tobytes())
                total += len(chunk_buf)
                chunk_buf = []
                dt = time.time() - t0
                print(f"    [bin] {split}: {total:,} tokens written  "
                      f"({rows_seen:,} rows)  [{dt:.0f}s, {total/max(dt,1e-6):,.0f} tok/s]")
            if max_rows is not None and rows_seen >= max_rows:
                break
        if chunk_buf:
            arr = np.array(chunk_buf, dtype=np.uint16)
            f.write(arr.tobytes())
            total += len(chunk_buf)
    dt = time.time() - t0
    print(f"  [bin] {split}: done. {total:,} tokens in {dt:.0f}s "
          f"({total/max(dt,1e-6):,.0f} tok/s avg), {rows_seen:,} rows")
    return total


class MMapTokenDataset(IterableDataset):
    """
    Memmap-backed fixed-length window sampler. Opens the .bin file as a
    np.memmap of uint16, then yields random (x, y) pairs of length seq_len
    where y is x shifted by 1. This is the fast training path.

    For the training split we use random window offsets.
    For validation we use strictly sequential non-overlapping windows so
    the eval set is deterministic and complete.
    """
    def __init__(self, bin_path, seq_len, mode="train", seed=0):
        super().__init__()
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.mode = mode
        self.seed = seed
        self.ntokens = os.path.getsize(bin_path) // 2  # uint16

    def __iter__(self):
        data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        max_start = self.ntokens - self.seq_len - 1
        if max_start <= 0:
            return
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        nworkers = worker_info.num_workers if worker_info is not None else 1
        if self.mode == "train":
            # Random offsets, each worker gets its own stream
            rng = random.Random(self.seed + wid * 9973)
            while True:
                start = rng.randint(0, max_start)
                window = data[start:start + self.seq_len + 1].astype(np.int64)
                x = torch.from_numpy(window[:-1])
                y = torch.from_numpy(window[1:])
                yield x, y
        else:
            # Sequential non-overlapping windows, sharded across workers
            step = self.seq_len + 1
            offset = wid * step
            while offset + step <= self.ntokens:
                window = data[offset:offset + step].astype(np.int64)
                x = torch.from_numpy(window[:-1])
                y = torch.from_numpy(window[1:])
                yield x, y
                offset += step * nworkers


def estimate_bytes_per_token(tokenizer, split="validation", n_rows=200):
    """
    Computes the average number of raw UTF-8 bytes each tokenizer token
    represents on this dataset. Used to convert cross-entropy loss into
    bits-per-byte (bpb), the standard scale-invariant LM metric.
    """
    ds = stream_dataset(split)
    total_bytes, total_tokens = 0, 0
    for i, row in enumerate(ds):
        if i >= n_rows:
            break
        text = row.get(DATASET_TEXT_KEY, "") or ""
        if not text.strip():
            continue
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
    ratio = total_bytes / max(total_tokens, 1)
    print(f"  [BPB] {split}: {total_bytes:,} bytes / {total_tokens:,} tokens = {ratio:.4f} bytes/token")
    return ratio


# ════════════════════════════════════════════════════════════════════════
#  ROPE
# ════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=MAX_SEQ_LEN, base=ROPE_BASE):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x, seq_len, offset=0):
        # offset: starting position, > 0 during cached generation
        return (self.cos_cached[:, :, offset:offset + seq_len, :],
                self.sin_cached[:, :, offset:offset + seq_len, :])


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# ════════════════════════════════════════════════════════════════════════
#  NORMS
# ════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class PhaseAwareRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        d_phase = d_model // NUM_PHASES
        self.norms = nn.ModuleList([RMSNorm(d_phase, eps) for _ in range(NUM_PHASES)])

    def forward(self, x):
        phases = x.chunk(NUM_PHASES, dim=-1)
        return torch.cat([self.norms[i](phases[i]) for i in range(NUM_PHASES)], dim=-1)


# ════════════════════════════════════════════════════════════════════════
#  SWIGLU
# ════════════════════════════════════════════════════════════════════════

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=DROPOUT):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# ════════════════════════════════════════════════════════════════════════
#  THREE-PHASE EMBEDDING WITH FIXED GABRIEL'S HORN DC INJECTION
# ════════════════════════════════════════════════════════════════════════

class ThreePhaseChannelStructureEmbedding(nn.Module):
    """
    Three-phase channel-structure embedding with fixed Gabriel's horn DC
    injection.

    The d_model embedding is split into NUM_PHASES = 3 contiguous channels.
    On every forward pass, the cross-phase DC mean is replaced with the
    value of the fixed analytic profile

                        r(p) = 1 / (p + 1)

    at token position p. The horn is a non-learnable buffer; it adds zero
    trainable parameters. The cross-phase mean of the embedding output is
    pinned at the horn value at every position, while every direction
    orthogonal to the all-ones DC subspace is left unchanged.
    """
    def __init__(self, vocab_size, d_model, max_len=SEQ_LEN + 1):
        super().__init__()
        self.d_model = d_model
        self.d_phase = d_model // NUM_PHASES
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Gabriel's horn profile r(p) = 1/(p+1):
        positions = torch.arange(0, max_len, dtype=torch.float32)
        horn = 1.0 / (positions + 1.0)
        self.register_buffer("horn_profile", horn.view(1, max_len, 1))

    def forward(self, x):
        B, T = x.shape
        combined = self.token_emb(x)

        # Subtract the current cross-phase DC mean and add back the horn.
        # The added value (horn - total_mean) lies entirely in the 1D.
        phase_means = []
        for i in range(NUM_PHASES):
            s, e = i * self.d_phase, (i + 1) * self.d_phase
            phase_means.append(combined[:, :, s:e].mean(dim=-1, keepdim=True))
        total_mean = sum(phase_means) / NUM_PHASES        # [B, T, 1]
        horn = self.horn_profile[:, :T, :]                # [1, T, 1]
        return combined + (horn - total_mean)

    def get_zero_sum_loss(self, x):
        """Diagnostic only. Measures the cross-phase residual of x; when
        the horn is active this equals NUM_PHASES * mean(horn[:T]) at
        every eval, every seed -- the orthogonality witness."""
        phases = x.chunk(NUM_PHASES, dim=-1)
        return sum(p.mean(dim=-1) for p in phases).pow(2).mean()


class PhaseRotationLayer(nn.Module):
    def __init__(self, d_model, layer_idx, n_layers):
        super().__init__()
        d_phase = d_model // NUM_PHASES
        self.theta = nn.Parameter(
            torch.ones(d_phase // 2) * (layer_idx + 1) * math.pi / (2 * n_layers)
        )

    def forward(self, x):
        d_phase = x.shape[-1] // NUM_PHASES
        phases = list(x.chunk(NUM_PHASES, dim=-1))
        for i in range(NUM_PHASES):
            offset = i * (2 * math.pi / NUM_PHASES)
            p = phases[i]
            x1, x2 = p[..., :d_phase:2], p[..., 1:d_phase:2]
            c, s = torch.cos(self.theta + offset), torch.sin(self.theta + offset)
            phases[i] = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)
        return torch.cat(phases, dim=-1)


# ════════════════════════════════════════════════════════════════════════
#  GROUPED QUERY ATTENTION WITH ROPE
# ════════════════════════════════════════════════════════════════════════

class GroupedQueryAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, max_len=MAX_SEQ_LEN):
        super().__init__()
        assert d_model % n_q_heads == 0, f"d_model {d_model} not divisible by n_q_heads {n_q_heads}"
        assert n_q_heads % n_kv_heads == 0, f"n_q_heads {n_q_heads} not divisible by n_kv_heads {n_kv_heads}"
        self.n_q, self.n_kv = n_q_heads, n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads
        self.d_head = d_model // n_q_heads
        self.q = nn.Linear(d_model, n_q_heads * self.d_head, bias=False)
        self.k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o = nn.Linear(n_q_heads * self.d_head, d_model, bias=False)
        self.drop = nn.Dropout(DROPOUT)
        self.scale = math.sqrt(self.d_head)
        self.rotary = RotaryEmbedding(self.d_head, max_len=max_len)

    def forward(self, x, mask=None, past_key_value=None, use_cache=False):
        """
        Training path (default): past_key_value=None, use_cache=False.
        Returns just the output tensor.

        Inference path: past_key_value=(past_k, past_v), use_cache=True.
        Returns (output, new_kv).

        Uses F.scaled_dot_product_attention which auto-selects Flash
        Attention 2 on Ampere+ GPUs with bf16.
        """
        B, T, C = x.shape
        past_len = 0 if past_key_value is None else past_key_value[0].shape[2]

        q = self.q(x).view(B, T, self.n_q, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.d_head).transpose(1, 2)

        # Apply RoPE at the correct absolute positions. During training past_len=0.
        cos_q, sin_q = self.rotary(x, T, offset=past_len)
        q = q * cos_q + rotate_half(q) * sin_q

        cos_k, sin_k = self.rotary(x, T, offset=past_len)
        k = k * cos_k + rotate_half(k) * sin_k

        # Concatenate with cached K/V if generating
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Repeat KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k_full = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_q, k.shape[2], self.d_head)
            v_full = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_q, v.shape[2], self.d_head)
        else:
            k_full, v_full = k, v

        # Flash Attention 2 via SDPA. is_causal=True during training (past_len=0);
        # during incremental generation past_len>0 so the current query attends
        # to all cached keys and cannot see future tokens — but since q has T=1
        # in that case and all cached keys are valid past, we can pass is_causal=False.
        is_causal = (past_key_value is None)
        out = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=None,
            dropout_p=DROPOUT if self.training else 0.0,
            is_causal=is_causal,
        )
        out = self.o(out.transpose(1, 2).reshape(B, T, C))

        if use_cache:
            return out, new_kv
        return out

# ════════════════════════════════════════════════════════════════════════
#  THREE-PHASE TRANSFORMER
# ════════════════════════════════════════════════════════════════════════

class ThreePhaseTransformer(nn.Module):
    """
    Three-Phase Transformer (3PT)

    A standard SwiGLU + RMSNorm + RoPE + GQA decoder block, modified by five
    coordinated phase-respecting operations that impose and maintain a cyclic
    Z_3 partition on the residual stream.

        1. Three-phase channel partition
              The d_model hidden vector is interpreted as 3 contiguous stripes
              ("phases A/B/C"), each of width d_phase = d_model / 3.

        2. Gabriel's horn DC injection                 (in the embedding)
              The 1D cross-phase DC subspace is replaced at every forward
              pass with the fixed analytic profile r(p) = 1/(p+1), serving
              as an absolute-position side-channel orthogonal to RoPE's
              relative-position rotation in attention.

        3. PhaseRotationLayer  (between attention and FFN, NON-residual)
              Each layer rotates each phase by theta + i*(2*pi/3) using a
              2D Givens rotation per consecutive (even, odd) channel pair.
              Theta is initialized depth-linearly: theta_i = (i+1)*pi/(2*L).
              Non-residual placement is safe because the layer is an
              orthogonal map; gradients flow through unattenuated.

        4. Phase-aligned GQA attention
              n_q_heads and n_kv_heads must both be divisible by NUM_PHASES,
              so each query/KV head lies entirely within a single phase.
              At 123M: n_q_heads=12, n_kv_heads=3 (4 Q heads / 1 KV head per phase).

        5. PhaseAwareRMSNorm  (replaces RMSNorm at every norm site)
              Three independent RMSNorm(d_phase) instances, one per phase,
              concatenated. Identical total parameter count to a single
              RMSNorm(d_model).
    """
    def __init__(self, vocab_size, d_model, n_layers, d_ff,
                 n_q_heads=N_Q_HEADS, n_kv_heads=N_KV_HEADS,
                 max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        assert d_model % NUM_PHASES == 0, \
            f"d_model {d_model} not divisible by NUM_PHASES ({NUM_PHASES})"
        assert n_q_heads % NUM_PHASES == 0, \
            f"phase-aligned GQA requires n_q_heads ({n_q_heads}) divisible by NUM_PHASES ({NUM_PHASES})"
        assert n_kv_heads % NUM_PHASES == 0, \
            f"phase-aligned GQA requires n_kv_heads ({n_kv_heads}) divisible by NUM_PHASES ({NUM_PHASES})"

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads

        self.embedding = ThreePhaseChannelStructureEmbedding(vocab_size, d_model)
        self.drop = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "attn": GroupedQueryAttentionWithRoPE(d_model, n_q_heads, n_kv_heads, max_len=max_seq_len),
                "ff":   SwiGLUFFN(d_model, d_ff),
                "n1":   PhaseAwareRMSNorm(d_model),
                "n2":   PhaseAwareRMSNorm(d_model),
                "pr":   PhaseRotationLayer(d_model, i, n_layers),
            }))

        self.norm_f = PhaseAwareRMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x, targets=None, return_aux=False):
        """
        Standard pre-norm decoder forward pass with the PhaseRotationLayer
        applied non-residually between attention and FFN. The non-residual
        placement is what forces the rotation to be load-bearing rather than
        bypassable.

        return_aux=True additionally returns the cross-phase residual of the
        embedding output as a diagnostic. With the horn active, this value
        is mathematically pinned at NUM_PHASES * mean(horn[:T]) at every
        forward pass.
        """
        B, T = x.shape
        x_emb = self.embedding(x)
        h = self.drop(x_emb)
        for b in self.blocks:
            h = h + b["attn"](b["n1"](h))
            h = b["pr"](h)                          # non-residual rotation
            h = h + b["ff"](b["n2"](h))
        logits = self.head(self.norm_f(h))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
            if return_aux:
                # Diagnostic: cross-phase residual (horn-orthogonality witness).
                # Computed for logging only; never added to the training loss.
                aux = self.embedding.get_zero_sum_loss(x_emb)
                return logits, loss, loss.detach(), aux.detach()

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation with grouped-query KV cache. The full
        three-phase forward path (PhaseRotationLayer between attention and
        FFN, PhaseAwareRMSNorm at every norm site) is preserved step-by-step.

        NOTE: incremental decoding with the horn DC injection has a known
        caveat. Each generated token is embedded individually with shape
        [B, 1], so the horn lookup reads horn_profile[:, :1, :] = horn[0]
        regardless of the token's true sequence position. This is fine for
        sanity-check generations but is not a faithful absolute-position
        signal during step-by-step decoding. Training and full-sequence
        evaluation pass [B, T] inputs and use the horn profile correctly.
        """
        self.eval()
        ids = prompt_ids
        past_kvs = [None] * len(self.blocks)

        # Warm the cache with the full prompt in one pass.
        h = self.drop(self.embedding(ids))
        for i, b in enumerate(self.blocks):
            attn_out, new_kv = b["attn"](b["n1"](h), past_key_value=None, use_cache=True)
            h = h + attn_out
            h = b["pr"](h)
            h = h + b["ff"](b["n2"](h))
            past_kvs[i] = new_kv
        logits = self.head(self.norm_f(h))[:, -1:, :]

        # Step generation.
        for _ in range(max_new_tokens):
            next_id = self._sample(logits, temperature, top_k)
            ids = torch.cat([ids, next_id], dim=1)
            h = self.embedding(next_id)
            for i, b in enumerate(self.blocks):
                attn_out, new_kv = b["attn"](b["n1"](h),
                                              past_key_value=past_kvs[i],
                                              use_cache=True)
                h = h + attn_out
                h = b["pr"](h)
                h = h + b["ff"](b["n2"](h))
                past_kvs[i] = new_kv
            logits = self.head(self.norm_f(h))
        return ids

    @staticmethod
    def _sample(logits, temperature, top_k):
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP AND UTILITIES
# ════════════════════════════════════════════════════════════════════════

def get_lr(step):
    """Cosine LR schedule with linear warmup and a 10% floor."""
    if step < WARMUP_STEPS:
        return LR * step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(TRAIN_STEPS - WARMUP_STEPS, 1)
    progress = min(1.0, max(0.0, progress))
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def measure_phase_means(model, x_sample):
    """
    In-training diagnostic. Returns the per-phase channel mean of the
    embedding output and their sum (the cross-phase residual) With the
    horn active.
    """
    if not hasattr(model, "embedding") or not hasattr(model.embedding, "get_zero_sum_loss"):
        return None
    was_training = model.training
    model.eval()
    x_emb = model.embedding(x_sample)
    d_phase = x_emb.shape[-1] // NUM_PHASES
    phase_means = []
    for i in range(NUM_PHASES):
        s, e = i * d_phase, (i + 1) * d_phase
        phase_means.append(x_emb[:, :, s:e].mean().item())
    zs = sum(phase_means)
    if was_training:
        model.train()
    return {"phase_means": phase_means, "zero_sum_residual": zs}


@torch.no_grad()
def theta_drift_snapshot(model):
    """Compact snapshot of PhaseRotationLayer theta drifts across all blocks."""
    if not hasattr(model, "blocks") or "pr" not in model.blocks[0]:
        return None
    n_layers = len(model.blocks)
    drifts = []
    for i, block in enumerate(model.blocks):
        theta = block["pr"].theta.detach().cpu()
        init_val = (i + 1) * math.pi / (2 * n_layers)
        drift = (theta - init_val).abs()
        drifts.append({
            "block": i,
            "init": float(init_val),
            "mean_final": float(theta.mean().item()),
            "l2_drift": float(drift.norm().item()),
            "max_drift": float(drift.max().item()),
        })
    return drifts


@torch.no_grad()
def evaluate_model(model, val_iter_fn, device, max_batches=EVAL_BATCHES, bytes_per_token=None):
    """
    Evaluates a model on a fresh val iterator. Returns val_loss, val_ppl,
    val_bpb (bits per byte, if bytes_per_token provided), and total tokens seen.
    """
    model.eval()
    total_loss, total_tokens, count = 0.0, 0, 0
    val_iter = val_iter_fn()
    for i, (x, y) in enumerate(val_iter):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP and device == "cuda"):
            logits, _ = model(x, targets=None)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=0,
            )
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        count += 1
    model.train()
    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg, 20.0))
    bpb = avg / (math.log(2) * bytes_per_token) if bytes_per_token else None
    return {"loss": avg, "ppl": ppl, "bpb": bpb, "batches": count, "tokens": total_tokens}


def save_checkpoint(path, model, optimizer, step, best_val_loss, extra=None):
    """Saves a training checkpoint. The payload includes the model state_dict,
    optimizer state_dict, current step, best validation loss, and any extra."""
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, device):
    """Loads a training checkpoint if it exists. Returns (step, best_val_loss).
     If the checkpoint or optimizer state is incompatible, returns (0, inf) and prints a warning."""
    if not os.path.exists(path):
        return 0, float("inf")
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception as e:
            print(f"  [Resume] Optimizer state incompatible, starting fresh optimizer: {e}")
    step = payload.get("step", 0)
    best = payload.get("best_val_loss", float("inf"))
    print(f"  [Resume] Loaded checkpoint at step {step}, best val loss {best:.4f}")
    return step, best


def train_model(model, train_iter_fn, val_iter_fn, device, name="", bytes_per_token=None,
                ckpt_path=None, best_path=None, metrics_path=None, resume=True):
    """
    Production training loop:
      - bf16/fp16 mixed precision
      - gradient accumulation (BATCH_SIZE x GRAD_ACCUM effective batch)
      - gradient clipping (GRAD_CLIP)
      - checkpointing every CKPT_EVERY steps
      - best-val checkpoint tracked separately
      - eval every EVAL_EVERY steps with val_loss / ppl / bpb
      - per-LOG_EVERY step train logs (loss, ce, aux, grad_norm, tokens/sec, lr)
      - JSONL metrics log at metrics_path
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
        fused=torch.cuda.is_available(),
    )

    start_step, best_val_loss = 0, float("inf")
    if resume and ckpt_path and os.path.exists(ckpt_path):
        start_step, best_val_loss = load_checkpoint(ckpt_path, model, optimizer, device)

    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and AMP_DTYPE == torch.float16 and device == "cuda"))

    metrics_file = open(metrics_path, "a") if metrics_path else None

    def log_jsonl(d):
        if metrics_file is None:
            return
        metrics_file.write(json.dumps(d) + "\n")
        metrics_file.flush()

    model.train()
    step = start_step
    t0 = time.time()
    t_last_log = time.time()
    tokens_since_log = 0
    loss_since_log = 0.0
    ce_since_log = 0.0
    aux_since_log = 0.0
    grad_norm_since_log = 0.0
    log_count = 0

    val_history = []
    train_iter = iter(train_iter_fn())

    print(f"\n  [Train] Starting {name} at step {step}/{TRAIN_STEPS}")
    print(f"  [Train] Effective batch = {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} sequences")
    print(f"  [Train] AMP: {USE_AMP}, dtype: {AMP_DTYPE}")

    pbar = tqdm(total=TRAIN_STEPS, initial=step, desc=name[:40], dynamic_ncols=True, smoothing=0.1)

    while step < TRAIN_STEPS:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_ce = 0.0
        accum_aux = 0.0
        accum_tokens = 0

        for micro in range(GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_iter_fn())
                x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP and device == "cuda"):
                # Train loss
                _, loss, ce_val, aux_val = model(x, targets=y, return_aux=True)
                loss = loss / GRAD_ACCUM

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item() * GRAD_ACCUM
            accum_ce += ce_val.item()
            accum_aux += aux_val.item()
            accum_tokens += y.numel()

        # LR schedule
        lr_now = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Gradient clipping + step
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        step += 1
        pbar.update(1)
        tokens_since_log += accum_tokens
        loss_since_log += accum_loss / GRAD_ACCUM
        ce_since_log += accum_ce / GRAD_ACCUM
        aux_since_log += accum_aux / GRAD_ACCUM
        grad_norm_since_log += grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        log_count += 1

        # Train metrics log
        if step % LOG_EVERY == 0:
            now = time.time()
            dt = now - t_last_log
            toks_per_sec = tokens_since_log / max(dt, 1e-6)
            avg_loss = loss_since_log / log_count
            avg_ce = ce_since_log / log_count
            avg_aux = aux_since_log / log_count
            avg_grad = grad_norm_since_log / log_count
            elapsed = now - t0
            tqdm.write(f"    step {step:>6d}/{TRAIN_STEPS}  loss {avg_loss:.4f}  ce {avg_ce:.4f}  "
                  f"aux {avg_aux:.4e}  grad {avg_grad:.3f}  lr {lr_now:.2e}  "
                  f"{toks_per_sec:,.0f} tok/s  [{elapsed:.0f}s]")
            pbar.set_postfix(loss=f"{avg_loss:.3f}", ppl=f"{math.exp(min(avg_ce, 20.0)):.2f}",
                             lr=f"{lr_now:.1e}", tok_s=f"{toks_per_sec:.0f}")
            log_jsonl({
                "type": "train", "step": step, "loss": avg_loss, "ce": avg_ce,
                "aux": avg_aux, "grad_norm": avg_grad, "lr": lr_now,
                "tokens_per_sec": toks_per_sec, "elapsed_sec": elapsed,
            })
            tokens_since_log = 0
            loss_since_log = 0.0
            ce_since_log = 0.0
            aux_since_log = 0.0
            grad_norm_since_log = 0.0
            log_count = 0
            t_last_log = now

        # Eval
        if step % EVAL_EVERY == 0:
            eval_out = evaluate_model(model, val_iter_fn, device, bytes_per_token=bytes_per_token)
            elapsed = time.time() - t0
            if eval_out["bpb"] is not None:
                print(f"    >> EVAL step {step:>6d}  val_loss {eval_out['loss']:.4f}  "
                      f"ppl {eval_out['ppl']:.2f}  bpb {eval_out['bpb']:.4f}")
                print(f"       elapsed {elapsed:.0f}s  val_batches {eval_out['batches']}  "
                      f"val_tokens {eval_out['tokens']:,}")
            else:
                print(f"    >> EVAL step {step:>6d}  val_loss {eval_out['loss']:.4f}  "
                      f"ppl {eval_out['ppl']:.2f}")

            # Phase-constraint health check
            sample_batch = next(iter(val_iter_fn()))[0][:2].to(device)
            phase_diag = measure_phase_means(model, sample_batch)
            if phase_diag:
                print(f"       phase_means {['%+.5f' % m for m in phase_diag['phase_means']]}  "
                      f"zero_sum_residual {phase_diag['zero_sum_residual']:+.5e}")
            theta_diag = theta_drift_snapshot(model)
            if theta_diag:
                drifts_str = " ".join(f"b{d['block']}:{d['l2_drift']:.3f}" for d in theta_diag)
                print(f"       theta L2 drift: {drifts_str}")
            # Horn values (fixed Gabriel's horn, constant across training)
            emb = model.embedding
            if hasattr(emb, "horn_profile"):
                h0 = emb.horn_profile[0, 0, 0].item()
                hT = emb.horn_profile[0, -1, 0].item()
                print(f"       horn (fixed):  horn[0]={h0:.4f}  horn[-1]={hT:.6f}")

            val_entry = {
                "type": "eval", "step": step, "val_loss": eval_out["loss"],
                "val_ppl": eval_out["ppl"], "val_bpb": eval_out["bpb"],
                "val_tokens": eval_out["tokens"], "elapsed_sec": elapsed,
            }
            if phase_diag:
                val_entry["phase_means"] = phase_diag["phase_means"]
                val_entry["zero_sum_residual"] = phase_diag["zero_sum_residual"]
            if theta_diag:
                val_entry["theta_drifts"] = theta_diag
            log_jsonl(val_entry)
            val_history.append(val_entry)

            # Save best
            if eval_out["loss"] < best_val_loss and best_path:
                best_val_loss = eval_out["loss"]
                save_checkpoint(best_path, model, optimizer, step, best_val_loss,
                                extra={"val_ppl": eval_out["ppl"], "val_bpb": eval_out["bpb"]})
                print(f"       *** new best val loss {best_val_loss:.4f}, saved to {best_path}")

        # Rolling checkpoint (keep most recent CKPT_KEEP)
        if step % CKPT_EVERY == 0 and ckpt_path:
            # ckpt_path is used as a prefix: actual files are
            #   {ckpt_path}.step{N}
            # and the most recent CKPT_KEEP are retained. Oldest gets deleted.
            stepped_path = f"{ckpt_path}.step{step}"
            save_checkpoint(stepped_path, model, optimizer, step, best_val_loss)
            # Also save a stable "latest" symlink-like copy at ckpt_path for resume
            save_checkpoint(ckpt_path, model, optimizer, step, best_val_loss)
            # Prune old stepped checkpoints
            existing = sorted(
                glob.glob(f"{ckpt_path}.step*"),
                key=lambda p: int(p.rsplit(".step", 1)[-1]) if p.rsplit(".step", 1)[-1].isdigit() else -1,
            )
            to_delete = existing[:-CKPT_KEEP] if len(existing) > CKPT_KEEP else []
            for old in to_delete:
                try:
                    os.remove(old)
                except OSError:
                    pass
            tqdm.write(f"    [ckpt] saved step {step} -> {stepped_path}  "
                       f"(keeping last {CKPT_KEEP})")

    pbar.close()

    # Final eval + final checkpoint
    eval_out = evaluate_model(model, val_iter_fn, device, bytes_per_token=bytes_per_token)
    elapsed = time.time() - t0
    final_entry = {
        "type": "final", "step": step, "val_loss": eval_out["loss"],
        "val_ppl": eval_out["ppl"], "val_bpb": eval_out["bpb"], "elapsed_sec": elapsed,
    }
    log_jsonl(final_entry)
    val_history.append(final_entry)
    print(f"\n  [Train] FINAL: val_loss {eval_out['loss']:.4f}  ppl {eval_out['ppl']:.2f}  "
          f"bpb {eval_out['bpb'] if eval_out['bpb'] else 'n/a'}  time {elapsed:.0f}s")

    if ckpt_path:
        save_checkpoint(ckpt_path, model, optimizer, step, best_val_loss)
    if metrics_file is not None:
        metrics_file.close()

    return val_history


# ════════════════════════════════════════════════════════════════════════
#  POST-TRAINING DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def inspect_embedding(model, model_name):
    """
    Post-training diagnostic snapshot for the canonical 3PT architecture.
    Prints:
      - Gabriel's horn profile (head/tail values, total energy)
      - PhaseRotationLayer theta drift per block (the U-shape from Figure 9b)
    """
    print(f"\n  ──── Diagnostic: {model_name} ────")
    emb = model.embedding
    print(f"    Embedding class: {type(emb).__name__}")

    # Gabriel's horn (fixed buffer; same value at every eval)
    if hasattr(emb, "horn_profile"):
        horn = emb.horn_profile.detach().cpu().view(-1)
        print(f"    Gabriel's horn (fixed buffer, non-learnable):")
        print(f"      horn[0]   = {horn[0].item():.6f}  (mouth)")
        print(f"      horn[-1]  = {horn[-1].item():.6f}  (tail at p={len(horn)-1})")
        print(f"      sum(horn) = {horn.sum().item():.4f}  (~ harmonic series H_{len(horn)})")

    # PhaseRotationLayer theta drift per block (the U-shape).
    print(f"    PhaseRotationLayer theta drift per block (init = (i+1)*pi/(2*L)):")
    n_layers = len(model.blocks)
    for i, block in enumerate(model.blocks):
        theta = block["pr"].theta.detach().cpu()
        init_val = (i + 1) * math.pi / (2 * n_layers)
        drift = (theta - init_val).abs()
        print(f"      block {i:>2d}: init={init_val:.4f}  "
              f"mean_final={theta.mean().item():+.4f}  "
              f"L2_drift={drift.norm().item():.4f}  "
              f"max_drift={drift.max().item():.4f}")


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Three-Phase Transformer                                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  d_model       : {D_MODEL:<52}║
║  d_ff          : {D_FF:<52}║
║  n_layers      : {N_LAYERS:<52}║
║  n_q_heads     : {N_Q_HEADS:<52}║
║  n_kv_heads    : {N_KV_HEADS:<52}║
║  seq_len       : {SEQ_LEN:<52}║
║  train_steps   : {TRAIN_STEPS:<52}║
║  warmup_steps  : {WARMUP_STEPS:<52}║
║  batch_size    : {BATCH_SIZE} x grad_accum {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} seqs{'':<25}║
║  ckpt_every    : {CKPT_EVERY} (keep last {CKPT_KEEP}){'':<35}║
║  horn_inject   : {'True  (fixed 1/(p+1) DC side-channel)':<52}║
║  device        : {DEVICE:<52}║
║  dataset       : {DATASET_NAME + '/' + str(DATASET_CONFIG):<52}║
║  tokenizer     : {TOKENIZER_NAME:<52}║
║  run           : {RUN_NAME:<52}║
╚══════════════════════════════════════════════════════════════════════╝
""")
    print(f"  AMP: {USE_AMP} (dtype={AMP_DTYPE})  |  Flash Attention 2 via SDPA")
    print(f"  Output dir: {OUT_DIR}")

    # Tokenizer
    print("\n  [1/4] Loading tokenizer ...")
    tokenizer, vocab_size, eos_id, pad_id = load_tokenizer()

    # Pre-tokenize to .bin files
    print("\n  [2/4] Preparing pre-tokenized .bin files ...")
    train_bin = f"{OUT_DIR}/train_tokens.bin"
    val_bin   = f"{OUT_DIR}/val_tokens.bin"
    n_train = prepare_bin("train", tokenizer, eos_id, train_bin)
    n_val   = prepare_bin("validation", tokenizer, eos_id, val_bin)
    print(f"    train: {n_train:,} tokens  ({n_train * 2 / 1e6:.1f} MB)")
    print(f"    val:   {n_val:,} tokens  ({n_val * 2 / 1e6:.1f} MB)")

    def make_train_iter():
        ds = MMapTokenDataset(train_bin, SEQ_LEN, mode="train", seed=SEED)
        return iter(DataLoader(
            ds, batch_size=BATCH_SIZE,
            num_workers=2,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
        ))

    def make_val_iter():
        ds = MMapTokenDataset(val_bin, SEQ_LEN, mode="val", seed=SEED)
        return iter(DataLoader(
            ds, batch_size=BATCH_SIZE,
            num_workers=2,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
        ))

    print(f"    Seq len: {SEQ_LEN}, per-device batch: {BATCH_SIZE}, grad accum: {GRAD_ACCUM}")
    print(f"    Effective batch: {BATCH_SIZE * GRAD_ACCUM} sequences = "
          f"{BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,} tokens/optim step")

    # Bytes-per-token calibration for bpb metric
    print("\n  [3/4] Calibrating bytes-per-token for bpb metric ...")
    try:
        bytes_per_token = estimate_bytes_per_token(tokenizer, split="validation", n_rows=200)
    except Exception as e:
        print(f"    [BPB] Calibration failed ({e}); bpb metric disabled")
        bytes_per_token = None

    # Build model
    print("\n  [4/4] Building model ...")

    name = "ThreePhaseTransformer (123M)"

    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = ThreePhaseTransformer(
        vocab_size, D_MODEL, N_LAYERS, D_FF,
        n_q_heads=N_Q_HEADS, n_kv_heads=N_KV_HEADS,
        max_seq_len=MAX_SEQ_LEN,
    )
    n_params = model.count_params()

    print(f"\n\n{'═'*78}")
    print(f"  {name}")
    print(f"{'═'*78}")
    print(f"  Parameters: {n_params:,}  (~{n_params/1e6:.1f}M)")

    safe = name.replace(" ", "_").replace("/", "_").replace("+", "plus").replace("(", "").replace(")", "")
    ckpt = f"{OUT_DIR}/{safe}__checkpoint.pt"
    best = f"{OUT_DIR}/{safe}__best.pt"
    metrics = f"{OUT_DIR}/{safe}__metrics.jsonl"

    val_history = train_model(
        model,
        train_iter_fn=make_train_iter,
        val_iter_fn=make_val_iter,
        device=DEVICE,
        name=name,
        bytes_per_token=bytes_per_token,
        ckpt_path=ckpt,
        best_path=best,
        metrics_path=metrics,
        resume=RESUME,
    )

    final = val_history[-1] if val_history else {}
    result = {
        "params": n_params,
        "final_val_loss": final.get("val_loss"),
        "final_val_ppl": final.get("val_ppl"),
        "final_val_bpb": final.get("val_bpb"),
        "history": val_history,
        "checkpoint": ckpt,
        "best_checkpoint": best,
        "metrics_log": metrics,
    }

    model.eval()
    inspect_embedding(model, name)

    diag = {"embedding_class": type(model.embedding).__name__}
    diag["phase_rotation_drifts"] = theta_drift_snapshot(model)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary
    print(f"\n\n{'═'*78}")
    print(f"  FINAL RESULTS ({TRAIN_STEPS:,} steps, {N_LAYERS} layers, d_model {D_MODEL})")
    print(f"{'═'*78}\n")

    print(f"  {name}")
    print(f"    params:   {result['params']:>12,}  (~{result['params']/1e6:.1f}M)")
    print(f"    val_loss: {result['final_val_loss']:.4f}" if result['final_val_loss'] is not None else "    val_loss: n/a")
    print(f"    val_ppl:  {result['final_val_ppl']:.4f}" if result['final_val_ppl'] is not None else "    val_ppl:  n/a")
    print(f"    val_bpb:  {result['final_val_bpb']:.4f}" if result['final_val_bpb'] is not None else "    val_bpb:  n/a")
    print()

    # Save results JSON
    results_out = {
        "config": {
            "run_name": RUN_NAME,
            "d_model": D_MODEL, "d_ff": D_FF, "n_layers": N_LAYERS,
            "n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS,
            "seq_len": SEQ_LEN, "max_seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM,
            "train_steps": TRAIN_STEPS, "warmup_steps": WARMUP_STEPS,
            "eval_every": EVAL_EVERY, "ckpt_every": CKPT_EVERY,
            "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "beta1": BETA1, "beta2": BETA2, "grad_clip": GRAD_CLIP,
            "dropout": DROPOUT, "seed": SEED,
            "dataset": DATASET_NAME, "dataset_config": DATASET_CONFIG,
            "stream_shuffle": STREAM_SHUFFLE,
            "tokenizer": TOKENIZER_NAME,
            "vocab_size": vocab_size,
            "num_phases": NUM_PHASES,
            "backbone": "SwiGLU + PhaseAwareRMSNorm + RoPE + GQA",
            "amp": USE_AMP, "amp_dtype": str(AMP_DTYPE),
        },
        "results": {name: result},
        "diagnostics": {name: diag},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_PATH}")
    print(f"\n{'═'*78}\n  DONE\n{'═'*78}\n")


if __name__ == "__main__":
    main()
    
    