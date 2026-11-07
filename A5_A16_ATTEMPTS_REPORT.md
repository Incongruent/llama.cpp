# A5–A16 attempts report (dual-GPU CUDA fork branch `A5-A16`)

Branch: `A5-A16` (the A5–A16 phase commits on top of `a1.1`), measured across early-November sessions.
Host: two-GPU Windows machine, RTX 3080 (cc 8.6, ~10 GB) + Tesla P40 (cc 6.1, ~24 GB),
no peer-to-peer path. Every change is opt-in via environment variables and defaults to
the previous behavior, so the branch is safe to merge piecemeal.

## Executive verdict

| Area | Final word |
|---|---|
| A1 hybrid layer placement `attn:12` | only measured speedup of the phase (PP +34.6%, TG +23.2%, see HYBRID_LAYER_SPLIT_BENCHMARK.md) |
| A11 tensor-split sweep | production ratio `0.22,0.78` **is the argmax**; every larger 3080 share fails VRAM-fit in both qualities — no more speed to be found on the ts axis |
| A9 AR spin/blocks | defaults best; larger spin and more blocks both ~1% TG loss; disabling AR ~2% PP loss (negative control confirms the AR path matters) |
| A10 BF16 AR wire | numerics verified; ~1% PP win vs F32 wire, TG neutral — default stays always-on |
| A5 MTP draft pin | works, but single-slot draft TG/acceptance is identical; keep default off |
| A13 output pin | works, but −14% TG on this topology; keep default off |
| A12 AR-under-capture guard | structural audit says AR can never be dispatched inside a capture; added a defensive guard anyway; 1M-token stress clean |
| A15 FA fallback audit | zero silent CPU/F16 fallbacks for FLASH_ATTN_EXT on either GPU across KV f16/bf16/q8_0 |
| A16 build gates | catches arch drift loudly; runtime demo exposed that a wrong-arch binary otherwise fails *silently* |

## Per-item detail

### A5 — MTP draft-layer output-device pin (`LLAMA_MTP_PIN`)

Semantics: pins the MTP draft tower layers (il >= n_layer) and their KV to the output
(logits) device instead of following frac-based placement. Draft hidden states already
live on the output device, so the idea was to remove a per-draft-step cross-GPU hop.

Measured (9B MTP surrogate, single slot, draft-mtp on, 2 reps each):
- Pin engaged confirmed by load log line: "1 draft layers pinned to output device CUDA1".
- Eval time identical: 1243 ms / 46 tokens / 36.19 vs 36.21 t/s; MTP acceptance per position
  byte-identical ((0.444, 0.185, 0.037) both modes).

Verdict: mechanism verified, benefit not observable at this scale — keep default off.
Caveat: only meaningful with `--spec-type draft-mtp`; requires n_layer_nextn > 0 and no
router layer (guarded).

### A9 — AllReduce poll-interval / kernel-blocks knobs

New envs in the internal AR path:
- `GGML_CUDA_AR_SPIN_CYCLES` — Pascal `clock()` spin before yielding; 0 disables the
  throttle; clamp 2^20; default 16384.
- `GGML_CUDA_AR_KERNEL_BLOCKS` — arrival-array blocks; clamp 1..32; constexpr default
  remains 8 (compiled-in fallback).

Matrix (9B surrogate, `-r 3`, pp2048/tg128, fixed two ratios):

| point | PP @0.22 | TG @0.22 | PP @0.78 | TG @0.78 |
|---|---|---|---|---|
| defaults (16384/8) | 4175 | 91.59 | 4156 | 91.64 |
| spin=65536 | 4143 | 90.55 | 4098 | 90.36 |
| blocks=32 | 4120 | 90.48 | 4120 | 90.44 |
| `GGML_CUDA_AR_DISABLE=1` | 4075 | 90.89 | 4083 | 91.07 |

Verdict: defaults win; larger spin or more blocks both cost ~1%; disabling the internal
AR costs ~2% PP — evidence the path is beneficial as configured. No tuning wins shipped.

### A10 — always-on BF16 reduction wire (`GGML_CUDA_AR_BF16_THRESHOLD`)

Wire semantics: F32 activations of any nonzero size take an F32->BF16->F32 round-trip
across the PCIe hop (halves the wire bytes). `GGML_CUDA_AR_BF16_THRESHOLD=0` forces the
unconverted F32/F16 wire; the "combined kernel" path keeps bit-equivalent accumulators.

Measured: num-based correctness (temp-0 sampled output identical between wire modes on
the 9B surrogate; also acceptance-profile identity under speculative decode), speed
r=3: F32-wire PP 4134/4125 vs BF16-wire 4175/4156 (~+1% for BF16), TG neutral.

Verdict: keep the BF16 wire always-on (already the default). A 27B perplexity delta was
not separately measured; correctness was established up-stack at temp 0.

### A11 — tensor-split ratio sweep

Harness: one llama-server per point at the section's own context, hybrid placement
knobs (layer-split-hybrid, override-tensor) removed so `--tensor-split` is the only
variable, 90% context fill + 512 predicted tokens.

Results (27B hybrid model, q8_0 KV):

| section | ctx | ts=0.22 | ts=0.30 | ts=0.35 | ts=0.40 | ts=0.45 |
|---|---|---|---|---|---|---|
| Q5_K_M | 155k | OK (tg 5.41, pp 117.4, MTP-acc 0.248) | FAIL | FAIL | FAIL | FAIL |
| Q6_K | 105k | OK (tg 6.74, pp 147.2, MTP-acc 0.288) | FAIL | FAIL | FAIL | FAIL |

Every point with a larger CUDA0 share fails to load (VRAM fit boundary). The raw
ts-only numbers are far below the A1 hybrid placement figures — which is the real
conclusion: **the balance axis is exhausted; extra A1-style token-placement work is
the only remaining lever.** `ts = 0.22,0.78` stays.

### A12 — AR-under-CUDA-graph-capture audit + guard

Structural audit: the internal AR dispatch sits in the meta backend's command
sequencer between per-device `ggml_backend_graph_compute_async` calls (never inside a
cudaStreamBeginCapture region), so a wedge from capturing the AR busy-wait kernel is
not structurally possible today.

Defense-in-depth added anyway: `ggml_cuda_ar_capture_guard()` at the AR entry point
checks `cudaStreamIsCapturing` on every backend stream and bails to the caller's plain
copy path before any AR state mutates (fixed warn-once). `GGML_CUDA_AR_GRAPH_GUARD=0`
disables the check for diagnostics only.

Stress (9B surrogate, graphs ON, ctx 4096, b2048/ub1024): 480 requests × ~2137 tokens
= ~1,025,760 tokens in 552 s, 0 request failures, 0 capture-guard trips, final health
OK, per-slot "graphs reused" counter reached 14,401.

### A13 — output-layer pin (`LLAMA_OUTPUT_PIN`)

Pins the logits head to the last CUDA device before draft resolution. Verified:
placement log line ("output layer pinned to device CUDA1"), temp-0 sampled output
identical to the unpinned run.

Measured cost on the 9B surrogate at `-ts 0.22,0.78` (r=3): TG 78.63 vs 91.59 (-14%),
PP 3929 vs 4175 (-6%). On this hardware the un-pinned default already places the head
where it works best.

Verdict: functional but harmful here — keep default off. Caveat: `LLAMA_OUTPUT_PIN`
overrides an explicit `output.weight` override-tensor (last device wins), which could
silently change a production launch; documented in the table row.

### A15 — flash-attention fallback audit (`GGML_CUDA_FA_DEBUG`)

One-shot per-device audit that prints, for the first FLASH_ATTN_EXT execution of each
(distinct device, kernel family, Q shape, K/V type, head geometry) combination:

    [FA] dev 1 cc 610 kernel VEC Q[256,2] kv q8_0/q8_0 ne10=256 ne11=256 gqa=4

Two things worth knowing:
1. Operationally, the Audit prints with `GGML_LOG_INFO` — run the server with
   `-lv 4`. At `-lv 3` these lines are filtered (learned the hard way; the early "no
   [FA] lines" scare was purely a verbosity artifact, fixed by an earlier commit).
2. The separate warn `resolve_fused_ops: Flash Attention over K/V rows ... missing
   support` refers to a different fused-op pass (whole-layer fused rows attention),
   and is correctly labeled as a WARN, not a silent attention fallback.

Measured selection map (attention never falls back to CPU in any config):
- KV f16 or bf16: cc 8.6 → `MMA_F16` (prefill) / `TILE` or `VEC` (decode); cc 6.1 →
  `TILE` both phases.
- KV q8_0: cc 8.6 → `MMA_F16` prefill / `VEC` decode; cc 6.1 → `VEC` prefill + decode.
- Also checked at the hybrid model's production shape (27B Q6_K, gqa=6, q8_0 KV):
  cc 6.1 VEC. Zero `NONE` kernels across all configs.

### A16 — build-time SASS gate + startup banner

- `scripts/check-arch.ps1`: post-build cuobjdump `-lelf` readback; fails the build when
  any expected arch is missing. Exit 0/1/2 = PASS/FAIL/tooling. Learned and fixed a
  PowerShell binding bug: `-Expected 61,86` arrives as the single literal string
  "61,86" inside a string[] param, so the list is now normalized
  (split on commas/semicolons/whitespace) before comparison.
- `ggml_cuda_init` compile-banner (`#ifdef __CUDA_ARCH_LIST__`): at startup it compares
  each device's compute capability against the compiled arch set and prints, e.g.:

      Device 1: cc 6.1 not in compiled SASS arch list (860), no compiled arch is compatible - device requires PTX JIT

  Silent when the compiled arch covers the device (verified: no banner on the correct
  61;86 dual build).

Runtime demonstration on an 86-only rebuild: the banner fired for the cc 6.1 device,
the gate FAILED the DLL, and — the important behavioral finding — **the backend
scheduler silently schedules around the incompatible device**: model buffers moved to
CUDA0 + CPU_Mapped (5562 + 667 MiB), no CUDA1 buffers at all, and nothing in the server
log flagged it (the server only dies if kernels are later forced onto the missing
device). With no banner and no gate, a mis-arched binary produces "plausible" bench
numbers while quietly dropping an entire GPU. The two gates plus a post-build `check-arch`
invoke are the only alarms; both are wired into the build wrapper.

### Flag-audit parenthetical (recorded earlier, unchanged)

`-DGGML_CUDA_ARCH=86` in a older wrapper was a no-op (not a llama.cpp CMake option) and
the wrapper lacked `-DCMAKE_CUDA_ARCHITECTURES`, producing a 3080-only drift — the
eventual reason A16 exists.

## Cross-cutting lessons

1. `-lv 3` filters ggml-domain INFO lines; any downstream diagnostic that prints at
   INFO needs `-lv 4` (or a WARN) to be visible in server logs.
2. Missing-SASS devices do not produce loud errors in the current scheduler — they are
   silently scheduled around. Always pair a bench session with the A16 gate.
3. Negative controls earn their keep: `GGML_CUDA_AR_DISABLE=1` losing ~2% PP is what
   actually proves the internal AR path is engaged and useful.
4. The ts axis is fully explored for this hardware pair (A11) — future dual-GPU gains
   must come from token placement (A1 family), not split ratios.

## Commit list (`a1.1`..`A5-A16`)

- `2d9b1510a` ggml-cuda : skip internal AllReduce under CUDA stream capture (A12)
- `d02a799c8` llama : add MTP draft-layer output device pin (A5)
- `97dc1ae48` llama : add output-layer device pin (A13)
- `fc61241dd` ggml-cuda : add startup cc banner and post-build SASS readback gate (A16)
- `813f5b68a` docs : extend fork notes for the A5-A16 phase (env vars, pending rows, A16 verified)
- `a3e4c4e64` ggml-cuda : print flash-attention audit at INFO level (A15)
- `f8be8b452` scripts : normalize -Expected arch list parsing in check-arch gate (A16)
- `4a24ebb1b` docs : record measured results for A5 A9 A10 A12 A13 A15 A16
- `ee0171ffb` docs : record A11 tensor-split sweep results (argmax remains production 0.22)
- `b3587086c` scripts : genericize toolkit path in check-arch usage doc (privacy scrub)
- plus this report.

## Environment variables introduced by this branch

| var | values | meaning |
|---|---|---|
| `LLAMA_MTP_PIN` | any non-empty != "off" | pin MTP draft tower + KV to the output device (A5) |
| `LLAMA_OUTPUT_PIN` | any non-empty != "off" | pin logits head to the last CUDA device (A13) |
| `GGML_CUDA_AR_SPIN_CYCLES` | 0..2^20 (default 16384) | Pascal AR poll spin in SM cycles; 0 = no throttle (A9) |
| `GGML_CUDA_AR_KERNEL_BLOCKS` | 1..32 (default 8) | arrival-array blocks for the AR kernel (A9) |
| `GGML_CUDA_AR_GRAPH_GUARD` | default on | refuse AR while any backend stream is capturing (A12) |
| `GGML_CUDA_AR_BF16_THRESHOLD` | default 1 | F32->BF16 wire threshold in bytes; 0 disables the wire (A10) |
| `GGML_CUDA_FA_DEBUG` | set to enable | one-shot per-device FA kernel-selection audit to stderr/I output (A15) |
