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

## A1dig follow-up (2026-11-07): all 16 attention layers do fit — by moving the draft tower and output head off device 0

The original sweep only varied the attention quota and left the draft tower
(`blk.64`) and the output head pinned to device 0 via `--override-tensor`
(the production overrides). Those two tensors are worth **990.94 MiB
combined on device 0** (measured from the identical-shape loads below;
output.weight ≈ 511 MiB, draft tower ≈ 480 MiB), and they are the reason
attn:13 crashed there. Moving both to device 1 frees 991 MiB, which buys
almost two full attention layers outright, and the remaining gap was closed
by trimming the device-0 compute buffer (ubatch) or the context.

Measured device-0 stack assembly (27B Q6_K, q8_0 KV, ctx 105216 unless noted;
`load_tensors` / `llama_kv_cache` / `sched_reserve` log lines, -lv 4):

| segment | value |
| --- | --- |
| attention layer weights | 291.25 MiB per layer (Q6_K) |
| attention layer KV | 218.35 MiB per layer (16 layers -> 33.17 kB/token) |
| attention-layer-only model buffer | `attn:16` model CUDA0 = 4660.03 MiB = 16 × 291.25 + shared ≈ 0 |
| main-KV recurrent part | 655.03 MiB (48 recurrent layers) |
| recurrent state (RS) | 598.50 MiB (CUDA1, unchanged in all configs) |
| MTP draft-context KV | 411.00 MiB at ctx 105216 (377–391 at 96–100k) |
| compute buffer | 776.53 MiB at ub 1024; 548–569 at ub 512; 493–502 at ub 256 |
| allocatable on device 0 | ≈ 8576 MiB (10240 minus desktop + CUDA contexts measured via nvidia-smi) |

Fit ladder verified by one llama-server launch per point (exit = fail, "listening" = pass):

| config (override) | ctx/pad | ub | outcome | CUDA0 stack (model + KV + rot + comp) |
| --- | --- | --- | --- | --- |
| attn:13, draft+output on CUDA0 (prod style) | 105216 | 1024 | FAIL (exit) | 5125.34 + 2838.47 + 3.45 + 776.53 = 8732 MiB |
| attn:14, draft CUDA1, output CUDA0 | 105216 | 1024 | FAIL (OOM 884.62 MiB) | — |
| attn:15, draft CUDA1, output CUDA0 | 100096 | 512 | FAIL | — |
| attn:16, draft+output CUDA1 | 105216 | 1024 | FAIL (exit) | 4660.03 + 3493.50 + 4.25 + 776.53 = 8934 MiB |
| attn:16, draft+output CUDA1 | 105216 | 256 | FAIL (also fails the 884.62 MiB KV-extend retry) | 4660.03 + 3493.50 + 4.25 + 502.38 = 8660 MiB |
| **attn:15, draft+output CUDA1 (full prod shape)** | **105216** | **1024** | **PASS** (473 MiB nvidia-smi headroom) | 4368.78 + 3275.16 + 3.98 + 776.53 = 8424 MiB |
| **attn:16, draft+output CUDA1** | **103168** | **256** | **PASS** (307 MiB headroom) | 4660.03 + 3425.50 + 4.25 + 493.38 = 8583 MiB |
| **attn:16, draft+output CUDA1** | **100096** | **512** | **PASS** (371 MiB headroom) | 4660.03 + 3323.50 + 4.25 + 568.77 = 8557 MiB |

(The recurring "248.10 MiB cudaMalloc failed ... retry" line appears in every
passing config including production's own shape; it is a reserve-iteration
retry, not a fatal. A failed KV-extend retry of 884.62 MiB, by contrast, kills the process.)

Performance (server, 90% ctx fill + 512 generated tokens, draft-mtp on,
temp-0 identity probe identical across all configs, single rep per point):

| config | ctx | ub | PP t/s | TG t/s | MTP acc |
| --- | --- | --- | --- | --- | --- |
| attn:12, draft+output CUDA0 (current prod shape) | 105216 | 1024 | 184.5 | 8.34 | 0.278 |
| attn:15, draft+output CUDA1 | 105216 | 1024 | **217.8** (+18.1%) | 8.91 (+6.8%) | 0.294 |
| attn:16, draft+output CUDA1 | 103168 | 256 | 206.5 (+12.0%) | 9.26 (+11.0%) | 0.283 |
| attn:16, draft+output CUDA1 | 100096 | 512 | 212.1 (+15.0%) | **9.58** (+14.9%) | 0.291 |

Cross-check at the llama-bench shape (p2048/n128, r=3, no overrides, hybrid
placement only, draft+output auto on CUDA1) is bandwidth-bound and the
output head on Pascal costs more than the attention gain there: PP 257.4/250.1/249.3
and TG 9.18/8.95/8.96 for attn:12/15/16 — do NOT use the small-KV leg to
tune placement; the 90%-fill server numbers above match the production
regime. A short-prompt decode probe (512 generated, spec on) confirms the
server numbers' order: a16 12.84/13.13 t/s vs a15 12.58/11.19 t/s (2 reps each).

### Updated recommendation

- Max decode (TG): `attn:16` + draft/head on device 1 at ctx 100096, ub 512 —
  PP +15.0%, TG +14.9% over the current attn:12 production shape.
- Max prefill with zero ctx/ubatch regression: `attn:15` + both moved at the
  full 105216/ub1024 shape — PP +18.1%, TG +6.8%.
- attn:16 with output/head: no fit at any ctx >= 100k tested with ub <= 1024;
  the output head (≈511 MiB) is the binding resident on device 0.
- Full 105216 ctx with attn:16 is infeasible at measured ub values (8660 MiB
  at ub 256 vs ~8576 allocatable); only a KV-type reduction (e.g. q5/q4 KV)
  or a smaller ctx would close that.

## A1rig replay (2026-11-07): the original one-shot benchmark re-run on the current build

The original A/B (status header, above) was one leg of a fixed 358k-char
C++-source prompt (~100,490 tokens), `n_predict 512`, temp 0 / seed 42 /
top_k 1, one-shot per fresh server start. Those exact assets were replayed
on the current build for the three placement candidates. Fees: attn:15 runs
the exact original shape (105000, ub 1024); attn:16 must drop to ctx 103000
/ ub 256, the only fitting shape large enough for the original prompt
(100,490 + 512 <= 103,168 cells). Single rep per leg.

| leg | shape | PP tok/s | TG tok/s | total wall | vs rerun attn:12 |
| --- | --- | --- | --- | --- | --- |
| attn:12 (rerun, current build) | 105000, ub 1024, draft+head dev0 | 168.34 | 17.12 | 626.9 s | — |
| attn:15 | 105000, ub 1024, draft+head dev1 | **217.27** | 16.88 | **492.9 s** | PP +29.1%, TG −1.4%, wall **−21.4%** |
| attn:16 | 103000→103,168, ub 256, draft+head dev1 | 144.05 | **19.09** | 724.4 s | PP −14.4%, TG **+11.5%**, wall +15.6% |

Reference: the original build's attn:12 leg measured PP 182.83 / TG 16.23;
the rerun reproduces TG above that (+5.5%) while PP is ~8% lower — build
drift from A5–A16, not placement, since both PP and TG of subsequent legs
move in directions the old build did not show. Notably the old high-TG
regime survives on the current build (17–19 t/s): the low TG values in the
90%-fill sweep above were prompt/content-dependent, not a placement
property.

Venue conclusions:

- **attn:15 is the composite winner at equal context**: +29% prefill with
  decode within noise of the prod shape, total wall −21.4%. It needs no
  concession (full 105000 ctx, ub 1024 unchanged).
- **attn:16 is the decode winner (+11.5% TG)** but loses this specific
  long-prefill workload on wall time, because fitting its KV forces ub 256
  (4× more prefill micro-batches). If the real usage is many short prefills
  / long multi-turn decode, the decode edge can dominate instead.
- PP across legs is ubatch-confounded for attn:16; the only controlled
  comparison (attn:12 vs attn:15, both ub 1024) shows pure placement gains.

## mmproj-device offload (2026-11-07): freeing the vision tower from the fast GPU

The mmproj (vision tower + projector) is resident on the fast device
by default (`clip_ctx: CLIP using CUDA0 backend`) and its
worst-case memory footprint is logged at load: **1157.64 MiB**. That
footprint is part of what forced the earlier fit compromises: the
attn:16 quota at ctx 105216 / ub 1024 failed while the mmproj sat on
device 0 (failed KV-extend retry), and moving the vision tower to
device 1 with `--mmproj-device CUDA1` removed exactly that failure.

Fit outcomes (attn:16, draft+output on device 1, privileged to use the
same 994.63 MiB CPU_Mapped share that all both-moved configs carry):

| config | ctx | ub/b | result | device-0 stack (model + KV + rot + compute) |
| --- | --- | --- | --- | --- |
| attn:16 + mmproj CUDA1 | 105216 | 1024/2048 | PASS | 4660.03 + 3493.50 + 4.25 + 776.53 = 8934 (smi 9739) |
| attn:16 + mmproj CUDA1 | 105216 | 2048/4096 | PASS | 4660.03 + 3493.50 + 4.25 + 1142.06 = 9300 (smi 9931, ~309 free) |

Performance:

| bench | config | PP tok/s | TG tok/s | MTP acc |
| --- | --- | --- | --- | --- |
| 90% fill (93,115-token prompt, spec on) | attn:16 + mmproj CUDA1, ub 1024 | **235.4** | 9.29 | 0.279 |
| 90% fill | attn:16 + mmproj CUDA1, ub 2048 | 210.2 | 9.94 | 0.286 |
| 90% fill | attn:15 (mmproj on CUDA0), ub 1024 | 217.8 | 8.91 | 0.294 |
| 90% fill | attn:12 prod shape (mmproj on CUDA0), ub 1024 | 184.5 | 8.34 | 0.278 |
| original one-shot rig (100,490 tok, n=512) | attn:16 + mmproj CUDA1, ub 1024 | **232.54** | 18.05 | 460.5 s wall |
| original one-shot rig | attn:15, ub 1024 | 217.27 | 16.88 | 492.9 s wall |
| original one-shot rig | attn:12 prod shape, ub 1024 | 168.34 | 17.12 | 626.9 s wall |

Temp-0 identity probe identical across all configs. Vision smoke test
(one 64x64 image through the chat-completions vision path on device 1,
cc 6.1): dispatch clean, model answers the content correctly
(`finish=stop`). The ub 2048 leg fits on VRAM but **inverts prefill**
(−25 PP t/s vs ub 1024) while buying +0.65 TG at ~50 MiB free — ub 1024
stays the operating point.

Updated standing recommendation (fast device = 10 GB card):

```
--layer-split-hybrid attn:16
--override-tensor '^blk\.64\..*=CUDA1,^output\.weight=CUDA1'
--mmproj-device CUDA1
--ctx-size 105000
--batch-size 2048 --ubatch-size 1024
```

vs the attn:12 production shape: PP +27.6% (fill) / +38.1% (one-shot),
TG +11.4% (fill) / +5.4% (one-shot), one-shot wall −26.6%, full original
context retained, and the vision tower moves off the fast device.

