# Fork notes: mixed-generation multi-GPU support for llama.cpp

This fork carries a small, self-contained set of changes for running large
quantized models across two GPUs of *different generations* over plain PCIe:

- an NVIDIA RTX 3080 (10 GB, compute capability 8.6),
- an NVIDIA Tesla P40 (24 GB, compute capability 6.1),
- no NVLink/NVSwitch, no peer-to-peer memory access between the two.

The upstream multi-GPU paths assume reasonably homogeneous, modern hardware.
The patches here remove incidental assumptions that block this configuration,
add one placement knob for hybrid-attention models, and vectorize a hot
K-quant to f16 conversion path. Every change is opt-in or behavior-preserving
by default, and each is its own commit so it can be dropped independently.

Target workload: Qwen3.8-27B (hybrid attention, ~27B parameters) quantized
with Q6_K/Q5_K tensors, tensor-split across both devices. The changes are
generation-agnostic; the names above are the configuration this fork was
developed and measured on.

## Branch layout

| Branch | Tag | Contents |
| --- | --- | --- |
| `pre-a1` | `snapshot-pre-a1` | Pascal internal AllReduce; one-shot MMQ config debug log |
| `a1-a4` | `snapshot-a1-a4` | everything above, plus A1–A4 below |

`git diff pre-a1..a1-a4` is exactly the A1–A4 delta.

## pre-A1 changes

### 1. Internal AllReduce on Pascal (cc 6.1)

Upstream's chunked (pipeline) AllReduce requires Volta or newer, because the
kernel's polling loop uses `__nanosleep`, which does not exist before sm_70.
On Pascal the kernel compiled to `NO_DEVICE_CODE`, and the pipeline was
rejected outright.

Changes:

- The polling loop now falls back to a `clock()`-based delay on pre-Volta
  architectures. The delay is ~16k SM cycles (roughly 8–11 µs at 1.5–2 GHz)
  between polls.
- The pipeline minimum is lowered from Volta (sm_70) to DP4A (sm_61), which
  is the minimum NVIDIA class for which the rest of the pipeline is valid:
  mapped pinned memory, volatile system-scope fences, events, and the copy
  engine are all architecture-neutral.
- New `GGML_CUDA_AR_DISABLE` environment variable: when set, internal
  AllReduce is disabled and the code falls back to the previous path.

Why the throttle matters on Pascal: each poll is a volatile read of a
host-mapped flag, which is itself a PCIe round trip (~1.5–2.5 µs). With 8
participating blocks, an unthrottled spin would inject tens to hundreds of
MB/s of 64-byte read TLPs into the same link that carries the reduction
data. Since the reduction token's propagation latency is only a few µs, a
deliberate delay costs about half a poll interval of added latency per
AllReduce while keeping link traffic negligible.

Expected effect: the chunked internal AllReduce becomes usable on Pascal
instead of being rejected. This matters on systems without NCCL, where the
fallback multi-GPU path is otherwise markedly slower. No measured
before/after numbers are recorded yet.

### 2. One-shot MMQ config debug log

Adds a debug-level line reporting the selected MMQ tile configuration:
device, compute capability, source type, `ne01`/`ne11`, `J`, `I`,
`nthreads`, shared-memory bytes, `smpbo`, fallback flag, and fusion flag.

- Deduplicated with a small `std::set`: each unique
  (cc, type, J, fallback, gate) key is logged once, so it is usable during
  long runs without flooding the log.
- Diagnostic only; cost is one set insertion per MMQ call and no string
  formatting unless a new key appears (and debug logging is enabled).
- The log was used to capture the effective config of the Pascal device on
  the target workload: `Q6_K J=64 I=64 nthreads=256 sram=27432` and
  `Q8_0 J=64 I=64 nthreads=256 sram=28224`.

#### Follow-up experiment: MMQ tile width (evaluated, rejected)

An A/B experiment halved the Q6_K/Q8_0 tile width (`I` 64→32) to test
whether lower shared-memory use would raise Pascal occupancy. It was
rejected: occupancy stayed at 2 blocks per SM (register-limited), and prompt
processing was 16–25% slower. The edit was reverted. Details and
measurements: [MMQ_PASCAL_TILE_EXPERIMENT.md](MMQ_PASCAL_TILE_EXPERIMENT.md).

## A1–A4 changes

### A1 — hybrid attention layer placement (`LLAMA_LAYER_SPLIT_HYBRID`)

New environment variable: `LLAMA_LAYER_SPLIT_HYBRID=attn`, `attn:N`, or
`off` (default).

- Only active for hybrid-recurrent architectures and exactly 2 devices;
  otherwise it logs a warning and is ignored.
- `attn`: place all full-attention layers on device 0, everything else on
  device 1. `attn:N`: at most the first N full-attention layers on device 0.
- Layers assigned to the CPU by the normal offload rules stay on the CPU.
- When active, the fractional `tensor_split` is bypassed (a warning is
  logged). This is intentional: the point is a semantic split, not a
  size-proportional one.

Rationale: in hybrid models the full-attention layers are usually the
latency-critical, compute-heavy ones, while recurrent/MoE layers dominate
memory. On a fast-small + slow-large pair, the default proportional split
puts some attention layers on the slow device. This knob lets the operator
put all (or a bounded number of) attention layers on the fast device.

Measured on the target pair with the target workload (100,490-token
prefill + 512-token generation): `attn:12` on the RTX 3080 gives PP +34.6%
and TG +23.2% over the same build at `off`; it is also the largest quota
that fits the 10 GB card. Details and the fit boundary:
[HYBRID_LAYER_SPLIT_BENCHMARK.md](HYBRID_LAYER_SPLIT_BENCHMARK.md).

### A2 — P2P setup is tri-state and tolerant (`GGML_CUDA_P2P`)

Before: presence of the environment variable enabled P2P, absence disabled
it, and any failure to enable peer access aborted initialization.

Now:

- unset: auto-enable peer access on every pair that reports
  `cudaDeviceCanAccessPeer` support;
- `0`: disable even when supported (explicit opt-out);
- any other value: explicit opt-in, same as unset behavior.

Enabling failures (for example `cudaErrorPeerAccessAlreadyEnabled` from an
earlier context or driver component) are logged and tolerated instead of
aborting. Peer access is never forced on unsupported pairs, so mixed
generations behave exactly as before on paths without P2P.

### A3 — vectorized Q4_K/Q5_K/Q6_K to f16 dequantization

Adds dedicated kernels for the quantized-to-f16 conversion path:

- Each thread dequantizes two adjacent elements of a 256-element
  super-block and emits a single 32-bit `half2` store
  (`__floats2half2_rn`), instead of element-wise 16-bit stores.
- 128 threads per block; same float math and operation order as the generic
  helpers, so results are bit-identical to the previous path.
- Selected only when `fp16_available(cc)`; other devices keep the generic
  implementation. No change for non-K quant types.

Expected effect: half the store instructions and half the per-element index
math in the conversion kernel, which is bandwidth/latency sensitive for
small row sizes. Exact speedup is unmeasured; correctness is bit-exact by
construction.

### A4 — per-backend event fences for pipeline parallelism

With pipeline parallelism (`-pp`), graph reuse requires the previous
asynchronous compute to finish before `set_inputs` overwrites the input
buffers it is still reading. Upstream waits with a blanket
`ggml_backend_sched_synchronize`, which synchronizes every backend
including the CPU one.

Changes:

- At pipeline-parallel initialization, create one event per non-CPU backend.
- After each enqueued `graph_compute`, record the fence on each backend.
- On the graph-reuse path, synchronize only those fences. CPU backends run
  synchronously and are skipped.

Why not remove the barrier entirely: reused graphs share intermediate
buffers across successive micro-batches. Writing the next micro-batch's
inputs while the previous compute is still reading those buffers corrupts
results. Fully asynchronous execution would require ping-pong
double-buffering of all intermediates, which is left as future work.

Expected effect: identical synchronization semantics for GPU work (no
wall-clock win expected), slightly less host-side bookkeeping, and no CPU
backend stall. Also groundwork for finer-grained overlap later.

## Performance summary

| Change | Mechanism | Expected effect | Measured |
| --- | --- | --- | --- |
| Pascal internal AllReduce | `clock()` spin, cc gate lowered, opt-out env | Internal AR available on cc 6.1; lower AR latency than the no-NCCL fallback; ~8–11 µs poll interval keeps PCIe TLP traffic negligible | pending |
| MMQ log + tile A/B | one-shot log; `I` 64→32 experiment | log is diagnostic; the tile-width experiment was refuted (−16…−25% PP, no occupancy gain) and reverted | measured: rejected ([details](MMQ_PASCAL_TILE_EXPERIMENT.md)) |
| A1 hybrid placement | attention layers pinned to device 0 | operator-tunable balance: latency-critical layers on the fast device, capacity on the large device | measured: `attn:12` on 3080+P40, PP +34.6%, TG +23.2% vs off ([details](HYBRID_LAYER_SPLIT_BENCHMARK.md)) |
| A2 P2P tri-state | auto-enable where supported, tolerate failures | no regression; P2P used only where actually supported | measured: neutral on the no-P2P pair, PP 135.86 vs 135.88 / TG 12.93 vs 13.17, identical output ([details](HYBRID_LAYER_SPLIT_BENCHMARK.md)) |
| A3 half2 K-quant dequant | 2 elements/thread, one 32-bit store | fewer stores and index ops on the Q4_K/Q5_K/Q6_K to f16 path; bit-exact output | bounded: A2+A3+A4 jointly ≈ 0% PP / +1.7% TG vs the pre-A1 build, bit-identical output ([details](HYBRID_LAYER_SPLIT_BENCHMARK.md)); A1-engaging legs don't touch it |
| A4 PP fences | per-backend events replace blanket synchronize | same timing semantics; less bookkeeping; groundwork for async overlap | not exercised in the measured workload (no pipeline-parallel reuse path) |
| A11 `-ts` sweep harness | scripted benchmark sweep over tensor-split fractions | operator picks the ts optimum without hand-rolling runs | measured 2026-11-07 (27B Q5_K_M ctx155k + Q6_K ctx105k, q8_0 KV, 90% fill, n_predict 512, no hybrid placement/override): both sections: ts=0.22 OK (tg 5.41 / 6.74 t/s raw ts-only, pp 117.4 / 147.2, MTP-acc 0.248 / 0.288); every larger 3080 share (ts 0.30/0.35/0.40/0.45) FAILS to load (VRAM fit boundary) in both sections -> argmax is the production ratio 0.22; all further TG gains must come from placement (A1), not ts: `ts = 0.22,0.78` stays |
| A15 FA kernel one-shot log | `GGML_CUDA_FA_DEBUG` one-shot per-device kernel-selection audit | diagnostics only; no runtime cost when off | measured 2026-11-07: KV f16/bf16 -> P40 TILE prefill+decode; KV q8_0 -> P40 VEC prefill+decode (9B gqa=4 and 27B prod shape gqa=6); zero `NONE` lines = zero CPU fallbacks across both GPUs; logs require `-lv 4` (ggml-domain INFO is filtered at `-lv 3`); the `resolve_fused_ops` "Flash Attention over K/V rows" warn is a separate fused-op support gate, not a silent FA fallback |
| A9 AR spin/blocks env | `GGML_CUDA_AR_SPIN_CYCLES`, `GGML_CUDA_AR_KERNEL_BLOCKS` runtime knobs | latency tuning for the AR poll interval and SM occupancy on Pascal | measured 2026-11-07 (9B r=3, pp2048/tg128, -ts 0.22 and 0.78): defaults best (tg 91.59/91.64); spin=65536 about -1.1% TG, blocks=32 about -1.2% TG, `GGML_CUDA_AR_DISABLE=1` about -0.8% TG and -2% PP; no win found, defaults (16384/8) kept |
| A10 BF16 AR wire verification | verify the always-on F32–BF16 reduction wire | confirms A10's claimed on-wire saving and numerics | measured 2026-11-07: threshold=0 (F32 wire) vs default BF16 wire -> sampled temp-0 output identical; BF16-on PP +1.0%/+0.8% (4175/4156 vs 4134/4125), TG neutral; default stays BF16 wire always-on |
| A5 MTP draft pin | `LLAMA_MTP_PIN` pins draft layers + KV to the output device | removes the per-draft-step cross-GPU hidden-state hop | measured 2026-11-07: pin engaged ("1 draft layers pinned to output device CUDA1"); draft-run TG/acceptance identical (36.21 t/s, acc profile (0.444, 0.185, 0.037) both modes); default off stays |
| A12 AR capture guard | refuse AR under CUDA stream capture, fall back to plain copies | protects graph replay correctness if AR is ever invoked captured | audit done (dispatch sits at ggml-backend-meta.cpp:2553 between per-device computes, never inside a capture) + measured 2026-11-07: ~1,025,760 tokens over 480 requests in 552 s with graphs ON, 0 failures, zero guard trips, graphs_reused=14401 at the end |
| A13 output pin | `LLAMA_OUTPUT_PIN` pins the logits head to the last device | explicit, testable per-token logits residency | measured 2026-11-07: pin works ("output layer pinned to device CUDA1"), output identical at temp-0, but on the 9B -ts 0.22/0.78 shape it costs -14% TG (78.63 vs 91.59) and -6% PP; NOT recommended for this topology; caveat: it overrides an explicit `output.weight` override-tensor (last-device wins); default off stays |
| A16 arch gate | `scripts/check-arch.ps1` SASS readback + startup cc banner | catches silent `-DCMAKE_CUDA_ARCHITECTURES` drift; triages slow-device reports | measured 2026-11-07 on an 86-only rebuild: banner fires ("Device 1: cc 6.1 not in compiled SASS arch list (860), no compiled arch is compatible - device requires PTX JIT"); gate FAILs on missing 61; behavior note: on missing SASS the scheduler silently schedules around the P40 (model buffer 5562 MiB -> CUDA0 + 667 MiB CPU_Mapped, no CUDA1 buffers, no WARN in server log) — the banner (stderr at backend init, not in --log-file) + the SASS gate are the only alarms; `-Expected` list parsing normalized (was binding "61,86" as one literal string) |

Three measured results exist so far: the MMQ tile-width experiment
([details](MMQ_PASCAL_TILE_EXPERIMENT.md)), refuted and reverted; the A1
hybrid placement A/B ([details](HYBRID_LAYER_SPLIT_BENCHMARK.md)), with
`attn:12` at PP +34.6% / TG +23.2% over `off`; and the A2 P2P toggle,
neutral on the no-P2P pair. Attribution: A1 accounts for the entire
speedup; A2 is measured neutral, A4 is not exercised by the measured
workload, and A3 is bounded within the same noise band as A2+A3+A4
jointly. Everything still marked "pending" should be treated as expected
effects derived from the code, not as results.

## Validation status

- All modified translation units compile cleanly with CUDA 12.9 for
  `sm_61` + `sm_86` (MMQ, flash attention, CUDA graphs enabled) and with
  MSVC for the host-side changes.
- The MMQ tile-width A/B ([details](MMQ_PASCAL_TILE_EXPERIMENT.md)) was
  measured on a live build (log patch applied); the experiment itself was
  reverted.
- A full link and runtime pass is the next step; runtime validation
  (tensor-split correctness, AllReduce numerics, pipeline-parallel output
  stability) is still pending.

## Build notes

Typical configuration used for validation:

```
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_QUANTS=all \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="61;86"
cmake --build build --config Release
```

## Environment variables introduced

| Variable | Values | Effect |
| --- | --- | --- |
| `GGML_CUDA_AR_DISABLE` | set | disable internal AllReduce, use fallback |
| `LLAMA_LAYER_SPLIT_HYBRID` | `attn`, `attn:N`, `off` | hybrid-aware attention placement (A1) |
| `GGML_CUDA_P2P` | unset/`1`, `0` | auto/on, off (A2) |
| `GGML_CUDA_FA_DEBUG` | set | one-shot audit log of flash-attention kernel selection per device (A15) |
| `GGML_CUDA_AR_SPIN_CYCLES` | 0..2^20 (default 16384) | pre-Volta AR poll throttle in SM cycles; 0 removes the throttle (A9) |
| `GGML_CUDA_AR_KERNEL_BLOCKS` | 1..32 (default 8) | chunked AR kernel grid size (A9) |
| `GGML_CUDA_AR_GRAPH_GUARD` | unset/`1`, `0` | default on: skip internal AR when a participating stream is capturing (A12) |
| `LLAMA_MTP_PIN` | `off`/unset (default), any other value | pin MTP draft layers + their KV to the output device (A5) |
| `LLAMA_OUTPUT_PIN` | `off`/unset (default), any other value | pin the output/logits head to the last GPU device (A13) |

All other behavior is unchanged from upstream. No files are written, no
telemetry is added, and no persistent configuration is modified.
