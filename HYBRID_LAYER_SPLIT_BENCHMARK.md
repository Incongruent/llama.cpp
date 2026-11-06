# A1 hybrid layer split A/B benchmark (measured)

Status: measured. `LLAMA_LAYER_SPLIT_HYBRID=attn:12` is adopted for the
development pair; it is the largest quota that fits the RTX 3080.

## Context

A1 places the first N full-attention layers of a hybrid-attention model on
device 0 and everything else on device 1 (see `FORK.md`, section A1). The
target model, Qwen3.8-27B (Q6_K, plus F16 mmproj), has 80 layers: 16
full-attention and 64 recurrent. The device pair is:

- device 0: RTX 3080 (10 GB, cc 8.6)
- device 1: Tesla P40 (24 GB, cc 6.1)
- plain PCIe, no P2P, CUDA 12.9

## Method

Two builds were compared: the current `a1-a4` build and a pre-A1 build
(vanilla `off`-equivalent; it predates A1, so the environment variable was
a no-op there). Identical server flags, model,
prompt, and sampling in every leg; each leg was a fresh server start after
full VRAM release.

Server: `--ctx-size 105000` (105,216 cells, single sequence), KV cache
q8_0 K and V, flash attention, `--batch-size 2048 --ubatch-size 1024`,
`--parallel 1`, draft `draft-mtp` with `--spec-draft-n-max 3`, `--fit on`,
`--n-gpu-layers 999`. The off legs additionally used the production
orientation `--tensor-split 0.22,0.78` with the top block and output
pinned to device 0 (`--override-tensor "^blk\.64\..*=CUDA0,^output\.weight=CUDA0"`);
hybrid legs ignore `tensor_split` by design.

Request: `/completion` with a fixed 358,000-character prompt of C++ source
= **100,490 tokens**, `n_predict 512`, greedy (`temperature 0`, `top_k 1`,
`seed 42`), `cache_prompt` off, one-shot per server start.

## Result

| Leg | Build | PP tok/s | PP wall | TG tok/s | TG wall | Total |
| --- | --- | --- | --- | --- | --- | --- |
| off | pre-A1 | 135.97 | 739.0 s | 12.95 | 39.5 s | 778.5 s |
| off | a1-a4 | 135.88 | 739.6 s | 13.17 | 38.8 s | 778.4 s |
| attn:11 | a1-a4 | 175.98 | 571.0 s | 16.37 | 31.2 s | 602.3 s |
| attn:12 | a1-a4 | **182.83** | 549.6 s | **16.23** | 31.5 s | **581.1 s** |

Against the same-build off leg:

- attn:12: PP **+34.6%**, TG **+23.2%**, total wall −25.3%
- attn:11: PP **+29.5%**, TG **+24.3%**, total wall −22.6%

Placement accounting from the load logs (weights / KV q8_0, both devices):

| Quota on dev 0 | dev 0 weights | dev 0 KV (attn layers) | dev 1 weights | dev 1 KV (attn layers) |
| --- | --- | --- | --- | --- |
| off (3 of 16, proportional) | 5917.83 MiB | 655.03 MiB | 14901.33 MiB | 2838.47 MiB |
| attn:11 | 4542.84 MiB | 2401.78 MiB | 16276.33 MiB | 1091.72 MiB |
| attn:12 | 4834.09 MiB | 2620.12 MiB | 15985.07 MiB | 873.38 MiB |

Costs per moved full-attention layer: **291.25 MiB** weights (Q6_K/Q5_K mix)
and **218.34 MiB** KV K+V q8_0 at 105,216 cells, i.e. ~510 MiB per layer
moved off or onto device 0.

## Observations

- Prefill scales monotonically with the quota (135.9 → 176.0 → 182.8 tok/s
  for off / 11 / 12): the attention layers dominate prefill compute.
- Decode is flat across the last interval (16.37 at 11 vs 16.23 at 12, run
  noise): decode is bounded below by the recurrent layers and the PCIe
  hops, not by attention placement.
- draft-mtp acceptance is unchanged ((0.930, 0.860, 0.776) off vs
  (0.937, 0.853, 0.783) hybrid, 3.57 mean accepted length in both): the
  gains are placement-driven, not speculative-decoding interplay.
- Greedy content differs by placement: hash `64e019eb7e437baf` for both off
  legs (old and new build identical), `ed19b98665f5f841` for both hybrid
  legs (11 and 12 identical). Device composition changes intermediate
  numerics; within one placement the pre-A1 and a1-a4 builds are bit-equal.
- The two builds at off are within noise (135.97/12.95 vs 135.88/13.17),
  so the entire delta is attributable to A1.
- The recurrent S-state buffer stayed on device 1 in every leg; the
  mmproj/vision pass and pipeline compute buffers remain on device 0 (GPU 0
  showed 9856 MiB used while serving attn:12).

## Fit boundary on the RTX 3080

Each extra full-attention layer on device 0 costs ~510 MiB of weights+KV,
on top of the vision pass and the ~0.8–1.2 GiB pipeline compute reserve.
The selection quota is context-size-dependent because the KV share scales
linearly with the number of cells: **at this workload's 105,216-cell q8_0
KV cache, 218.34 MiB per full-attention layer is the KV cost that fixes
the max fitting quota at 12** on the 10 GB card. Doubling the context
doubles the KV cost per layer and drops the max fitting quota by roughly
two layers; other KV types shift it the same way (f16 KV costs about 2×
the q8_0 KV at equal context). Re-derive the quota from
`(free dev-0 VRAM − vision − compute reserve) / (weights_per_layer +
kv_per_layer)` when context or KV type changes. Measured outcomes for
larger quotas on the 10 GB card:

| Quota | Outcome |
| --- | --- |
| attn (all 16) | clean fail: reserve attempt of 776.53 MiB on dev 0 fails, `failed to allocate compute pp buffers`, server exits |
| attn:15 | same clean reserve failure (814,252,032-byte attempt) |
| attn:14 | **GGML_ASSERT(buffer) failed** (`ggml/src/ggml-backend.cpp:328`) during load |
| attn:13 | same assert, GGML crash |
| attn:12 | loads and serves; one small (248.10 MiB) reserve iteration failed and the pass continued with smaller reserves |

Follow-up for the patch: quotas just over the fit boundary currently crash
through `ggml_backend_buffer` when an allocation returns null, instead of
taking the graceful error path that the larger quotas hit. A guard there
would turn a GGML abort into a clean `failed to allocate` message.

## Why `--override-tensor` cannot replace A1

An obvious alternative is per-tensor placement with `--override-tensor`
(`-ot`), pinning the first twelve attention blocks to device 0 by name.
It does not work, measured:

- `-ot` relocates **weight tensors only** (rules are first-match-wins,
  scoped to tensor names). The layer→device map (`dev_layer`) that drives
  KV-cache placement, recurrent-state placement, and graph scheduling is
  computed from `--tensor-split` and is not consulted by `-ot`.
- With the model's 16 full-attention blocks at stride 4
  (3, 7, 11, …, 63) and `-ot` rules for the first twelve,
  the load produced a contradictory layout — attention weights pinned to
  device 0 while the same layers' KV stayed on the contiguous
  0.22-split head, and several recurrent layers' states on device 0
  under weights pinned to device 1 — and the process **aborted during
  memory init** before serving.
- Even without the abort, weights and cache for the same layer would sit
  on different cards; A1's value is precisely co-location (weight, KV,
  and schedule move together), which no `-ot` scheme expresses today.

This is the mechanism argument for A1 against the "just use
`--override-tensor`" review objection.

## Attribution across A1–A4

The speedup is attributable to A1; the other changes are neutral or
not exercised in this workload:

- **A1** is the only env-gated change; the off ↔ attn:12 pair ran on the
  same binary, so PP +34.6% / TG +23.2% is A1's effect alone.
- **A2 (P2P tri-state)** was toggled on the same binary with the full
  off-leg protocol: `GGML_CUDA_P2P=0` measured PP 135.86 / TG 12.93 vs
  135.88 / 13.17 for the unset auto path, same greedy output. Neutral
  within noise — expected, since this pair has no P2P support; the change
  only alters init failure handling.
- **A4 (per-backend PP fences)** only acts on the pipeline-parallel
  graph-reuse path, which none of the measured legs enable; zero effect
  in these runs by construction.
- **A3 (vectorized Q4K/Q5K/Q6K→f16 dequant)** has no runtime toggle. It
  is bounded by the cross-build comparison, which covers A2+A3+A4
  jointly: pre-A1 build vs a1-a4 at off measured 135.97/12.95 vs
  135.88/13.17 (≈0% PP, +1.7% TG, bit-identical output). With A2 measured
  ~0 and A4 not engaged, A3's contribution sits inside the same noise
  band. Exact split would need per-commit builds.

| A2 toggle leg | PP t/s | TG t/s | output hash |
| --- | --- | --- | --- |
| `GGML_CUDA_P2P` unset (auto) | 135.88 | 13.17 | `64e019eb7e437baf` |
| `GGML_CUDA_P2P=0` | 135.86 | 12.93 | `64e019eb7e437baf` |

## Recommendation

For this model on the 3080 + P40 pair: set
`LLAMA_LAYER_SPLIT_HYBRID=attn:12`. The 12/16 boundary comes from the
105,216-cell q8_0 KV; if context length or KV type changes, re-derive the
quota from the per-layer costs above (~510 MiB per layer).
