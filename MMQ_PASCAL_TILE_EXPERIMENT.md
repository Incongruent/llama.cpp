# MMQ Pascal DP4A tile-width experiment (evaluated, rejected)

Status: rejected; the change was reverted. Only the one-shot MMQ config log
(`ggml/src/ggml-cuda/mmq.cu`) remains in this fork.

## Context

The one-shot MMQ config log added in `pre-a1` was used to capture the
effective config of the Tesla P40 (cc 6.1, GP102) on the target workload,
Qwen3.8-27B (Q6_K):

- `Q6_K`: `J=64 I=64 nthreads=256 sram=27432 fallback=0`
- `Q8_0`: `J=64 I=64 nthreads=256 sram=28224`

In the MMQ config table, `I` is the compile-time weight-tile width and `J`
is the activation-tile width. The tile config is selected per
`(type, J, fallback, precision)` and baked into the kernel at compile time
through `__device__ constexpr` lookups, so a table edit requires a rebuild.

The hypothesis was that the Pascal device is limited to 2 resident blocks
per SM by shared memory (27432 × 3 > 64 KiB), and that halving `I` to 32
would drop the per-block shared-memory footprint to ~18.4 KiB
(`sram` ≈ 18452 for Q6_K, ≈ 18848 for Q8_0), permitting 3 blocks per SM
(25% → 37.5% occupancy). The change was expected to be performance-only:
the DP4A integer dot products are exact, and the `I` field is documented as
not affecting results.

## Method

Single-file edit to `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh`: the
fourth `CASE` argument (`I`) was changed from 64 to 32 on all Q6_K and Q8_0
rows, both fallback groups. `nthreads`, `occupancy`, `J`, shared-memory
layout, and `K_vram` were left unchanged, as were all other quant types and
the RTX 3080's Ampere table (it was unaffected).

The table is defined once as `__host__ __device__` and is used by both host
dispatch and device dispatch, so host-side shared-memory/grid math and the
kernel's compile-time tiling stay consistent. The user rebuilt and
restarted the live server; the post-rebuild run confirmed the new config was
active (`I=32`, `sram=18452/18848` on the Tesla P40) and the RTX 3080 config
was unchanged.

## Result

The occupancy gain never materialized. `cuobjdump` on the `I=32` build
showed the Q6_K `J=64` kernel uses 88 registers per thread; at 256 threads
per block, three blocks require 88 × 256 × 3 = 67584 registers, above the
65536-register file, so occupancy remained at 2 blocks per SM. The
regression is therefore pure per-thread efficiency loss: with `I=32` each
warp covers one x-row instead of two, doubling shared y-loads per DP4A and
halving ILP.

Same-prompt prompt-processing comparison (cumulative tok/s):

| tokens | `I=64` baseline | `I=32` experiment | delta |
| --- | --- | --- | --- |
| 2048 | 403.3 | 302.7 | −25% |
| 8192 | 293.9 | 226.8 | −23% |
| 16384 | 259.6 | 204.4 | −21% |
| 24985 | 234.0 | 189.0 | −19% |
| 33177 | 216.3 | 177.6 | −18% |
| 41369 | 201.4 | 167.6 | −17% |
| 45465 | 194.8 | 163.1 | −16% |

The device was at its maximum clock with no competing compute during both
runs, and both runs used the same prompt, so the comparison is
like-for-like. Absolute throughput depends on the workload and is only
meaningful relative to the baseline column.

## Conclusion

- Rejected: `I=32` is uniformly 16–25% slower and does not improve
  occupancy.
- The edit was reverted; the table matches upstream, and the only surviving
  piece of this work is the config log.
- `I=64/J=64` is the local optimum for this Pascal DP4A configuration, and
  tile-table tuning on this device is considered exhausted. To gain
  occupancy here, register pressure would have to drop below 85
  registers/thread first; the measured alternatives do not.
