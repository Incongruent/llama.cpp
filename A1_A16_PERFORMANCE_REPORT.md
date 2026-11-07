# A1–A16 performance report: hybrid placement campaign

Scope: every measured optimization campaign on this fork for the two-device
production rig, branches `a1.1` -> `A5-A16`. All supporting measurements are
single-rep unless noted; every temp-0 identity probe across configs reproduced
identical output. Companion docs:
[A5_A16_ATTEMPTS_REPORT.md](A5_A16_ATTEMPTS_REPORT.md),
[HYBRID_LAYER_SPLIT_BENCHMARK.md](HYBRID_LAYER_SPLIT_BENCHMARK.md),
[FORK.md](FORK.md).

## Rig and workload

- Devices: NVIDIA RTX 3080 (10 GB, cc 8.6, fast) + Tesla P40 (24 GB, cc 6.1),
  plain PCIe, no peer access, CUDA 12.9 toolchain, SASS 61+86.
- Production model: 27B hybrid-attention (16 full-attention + 64 recurrent
  layers), Q6_K weights, F16 vision projector, q8_0 K/V KV cache,
  flash attention, MTP speculative decode (draft n=3).
- Three measurement rigs:
  1. **90 % context-fill server bench** — production-like: ~93k-token prompt,
     n_predict 512, spec on (the *fill rig*).
  2. **Original one-shot rig** — fixed 358k-char C++-source prompt
     (100,490 tokens), n_predict 512, temp-0/seed-42, fresh server per leg
     (the *A-rig*; this is the rig where the historical A1 numbers live).
  3. **llama-bench cross-check** — p2048/tg128: decode-bandwidth-bound and
     placement-insensitive for these configs; useful only to invalidate
     claims, never to choose placements (see pitfalls).

## Branch structure: what changed where

| Branch | Content | Effect character |
| --- | --- | --- |
| `a1.1` | **The only throughput-bearing source change**: hybrid-aware non-contiguous layer placement (`LLAMA_LAYER_SPLIT_HYBRID`, `--layer-split-hybrid attn:N`) routing N of 16 full-attention layers to the fast device; KV/recurrent-state/output placement follows automatically | main wins below |
| `A5-A16` (head) | A2–A16: P2P tri-state, half2 K-quant dequant, PP fences, ts-sweep harness, FA kernel audit log, AR spin/blocks knobs, BF16 reduce-wire verification, MTP draft pin, AR capture guard, output-layer pin, SASS arch gate + cc banner | hardening + diagnostics; measured neutral or off-path on throughput |
| follow-up digs | **No source changes at all** — configuration-only work (env/args: quotas, overrides, `--mmproj-device`, batch sizes) on branch binaries; repo delta in this period is documentation and a one-comment script scrub | config research |

## Placement campaign results

### Fill-rig benchmark (90 % of context, spec on)

| # | placement (only variables changed) | context / ub | PP tok/s | TG tok/s | MTP acc | delta vs baseline row |
| --- | --- | --- | --- | --- | --- | --- |
| 0a | vanilla (no placement args at all: no hybrid, no ts, no overrides, default free-memory split) | 105216/1024 | 144.0 | 6.37 / 6.64 (2 reps) | 0.246 / 0.271 | measured control on the current build |
| 0 | pre-hybrid orientation (ts 0.22, top/output on fast GPU, hybrid off) | 105216/1024 | 147.2 | 6.74 | 0.288 | from the ts-sweep measurement on this build |
| 1 | attn:12 quota + draft tower + output head on fast GPU (long-standing production) | 105216/1024 | 184.5 | 8.34 | 0.278 | reference row |
| 2 | attn:15 + draft & head on slow GPU | 105216/1024 | 217.8 | 8.91 | 0.294 | PP +18.1 %, TG +6.8 % |
| 3 | attn:16 + draft & head on slow GPU | 100096/512 | 212.1 | 9.58 | 0.291 | PP +15.0 %, TG +14.9 % |
| 4 | attn:16 + draft & head on slow GPU | 103168/256 | 206.5 | 9.26 | 0.283 | PP +12.0 %, TG +11.0 % |
| 5 | **attn:16 + draft & head + vision on slow GPU** | **105216/1024 (full prod shape)** | **235.4** | **9.29** | 0.279 | **PP +27.6 %, TG +11.4 %** |
| 6 | same, ub 2048/b 4096 | 105216/2048 | 210.2 | 9.94 | 0.286 | ub 2048 fits but inverts prefill |

### A-rig benchmark (one-shot 100,490-token prompt, n_predict 512)

| leg | shape | PP tok/s | TG tok/s | wall |
| --- | --- | --- | --- | --- |
| vanilla control (current build, no placement args) | 105216/1024 | 147.30 | 13.03 | 721.5 s |
| off / ts orientation (historical build) | 105216/1024 | 135.88 | 13.17 | 778.4 s |
| attn:12, draft+head fast GPU (long-standing production) | 105216/1024 | 182.83 | 16.23 | 581.1 s |
| attn:12 (current build, rerun control) | 105216/1024 | 168.34 | 17.12 | 626.9 s |
| attn:15, draft & head slow GPU | 105216/1024 | 217.27 | 16.88 | 492.9 s |
| attn:16, draft & head slow GPU | 103168/256, ub 1024→256 | 144.05 | 19.09 | 724.4 s |
| **attn:16 + vision slow GPU (recommended)** | **105216/1024** | **232.54** | **18.05** | **460.5 s** |

Recommended-row vs production-row deltas (same rig, same build):
**PP +38.1 %, TG +5.4 %, wall −26.6 %**. Against the genuine vanilla
configuration (current build, zero placement arguments, A-rig):
**PP +57.9 % (147.30 → 232.54), TG +38.5 % (13.03 → 18.05),
wall −36.2 % (721.5 s → 460.5 s)**; on the fill rig vs vanilla:
**PP +63 %, TG ~+43 %**. The vanilla default split reproduces the
historical ts-only orientation rows (147.3/13.0 one-shot vs 135.9/13.2
historical; 144.0/6.5 fill vs 147.2/6.74), so the old-orientation numbers
remain valid baselines and the placement deltas stack on them.

## Fit bookkeeping (why the ceilings moved)

Fast-device budget assembly at ctx 105216, q8_0 KV (all values measured from
load logs):

| item | size |
| --- | --- |
| per full-attention layer, weights (Q6_K) | 291.25 MiB |
| per full-attention layer, KV | 218.35 MiB |
| recurrent main-KV (64 layers) | 655.03 MiB |
| MTP draft-context KV | 411.00 MiB |
| recurrent state buffer (slow device) | 598.50 MiB |
| draft + output tensors (freed by moving both off dev 0) | 991 MiB |
| vision tower + projector, worst-case (freed by `--mmproj-device`) | 1157.64 MiB |
| compute buffer | 776.53 MiB @ ub 1024; 1142.06 @ ub 2048 |

Ceilings discovered: attn:16 alone fits at 100096/512 or 103168/256; the
vision tower on the fast device was the blocker that made full-shape
attn:16 infeasible (failed ~885 MiB KV-extend retry during load); moving it
(`--mmproj-device CUDA1`, supported in-branch flag, no source change) makes
full shape and even ub 2048 fit. Remaining caveats:

- Both-moved configs mmap ~994.63 MiB of model tensors via a CPU-mapped
  buffer on every load (standing observation, feature not regression; exact
  tensor attribution open).
- The benign `allocating 248.10 MiB ... retry` line appears in every passing
  load including the historical baseline; only the **884.62 MiB** failed
  extend kills the process (that distinction diagnosed the legacy ceiling).
- ub 2048 fits but prefill inverts (235.4 -> 210.2 PP) — batch tuning
  saturates at ub 1024 on the 10 GB card.
- KV quantization (q5/q4) unexplored; would buy another attention layer at
  full context.
- llama-bench tg/p1024-style legs show placement order flat or reversed
  (Pascal-side output head effect at pipe-short workloads); every placement
  decision above used the fill/A-rigs, not llama-bench.
- Vision path verified end-to-end on the slow device (cc 6.1): image chat
  dispatches cleanly and answers a color probe correctly (finish=stop).

## Per-item ledger (all forks rounds, A1.1→A16)

| item | what it is | measured status |
| --- | --- | --- |
| A1.1 hybrid split | farm quota of full-attn layers + followers (KV/RS/output/draft) | main win; see tables above |
| A2 P2P tri-state | enable peer access only where supported, tolerate failure | neutral (no-P2P pair) |
| A3 half2 dequant | 2 elems/thread f16-K dequant path, bit-exact | ~0 % in joint bound with A2/A4 |
| A4 PP fences | per-backend fences replace blanket sync | not exercised (no pipeline-parallel leg) |
| A5 MTP pin | pin draft to output device | neutral (acceptance identical) |
| A9 AR knobs | spin/blocks env tuning | defaults win |
| A10 BF16 wire | always-on F32↔BF16 reduce wire | PP +1 %, TG neutral, outputs identical |
| A11 ts harness | sweep tool | proved the production ts ratio is argmax; all TS gains exhausted → placement is the lever |
| A12 AR capture guard | refuse AR inside graph capture | guard trips = 0 across ~1.03 M-token stress |
| A13 output pin | pin logits head via env | works, costs −14 % TG on this shape → off |
| A15 FA audit | one-shot kernel-selection log | no CPU fallbacks on either device |
| A16 arch gate | SASS readback gate + cc banner | catches silent arch drift; surfaced silent scheduler fallbacks in earlier legs |
| mmproj offload | `--mmproj-device` (pre-existing flag) | enabled the recommended full-shape config (row #5) |

## Production impact summary

Long-standing production shape (attn:12, everything else on the fast GPU):
**PP 184.5 / TG 8.34** (fill) — was the best that fit.

Adopted production shape (config-only change on the same binaries):

```
layer-split-hybrid = attn:16
override-tensor    = ^blk\.64\..*=CUDA1,^output\.weight=CUDA1
mmproj-device      = CUDA1
# ctx, batch-size, ubatch-size unchanged (105000 / 2048 / 1024)
```

- PP **+27.6 %** / TG **+11.4 %** on the production-shaped fill rig
- PP **+38.1 %** / TG **+5.4 %** / wall **−26.6 %** on the one-shot rig
- full original context retained; vision inference moved to the large device
- rollback = revert three config lines (documented above); binaries unchanged

Attribution: the entire speedup is placement-driven (A1.1 mechanism +
configuration of quotas/overrides/mmproj-device). The A5–A16 code items
contribute infrastructural robustness (correctness guards, diagnostics,
build gates), not throughput.

## Open items

- CPU-mapped ~1 GiB model share in both-moved configs: characterize which
  tensors mmap to host and whether dropping the vision KV or draft changes it.
- One-shot-rig PP at attn:12 drifted −7.9 % between the a1-a4 build and the
  A5-A16 tip (TG +5.4 %); fill-rig PP unchanged — attribution to a specific
  A-item unresolved, does not affect the recommended config's ranking.
- KV q5/q4 at full context: expected +1 attention layer; unmeasured.
- ub 2048 prefill inversion: kernel-level explanation pending (suspect
  occupancy boundary on the fast device), measured, not fixed.
