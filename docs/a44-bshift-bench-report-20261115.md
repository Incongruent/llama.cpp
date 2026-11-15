# A44 b_shift bench report — 2026-11-15

**VERDICT: bench gate FAILED (correctness) — NO MERGE to prod.**
The `--b-shift on` path produces degenerate temp-0 output (repetition
loops). The branch is kept for a fix cycle; see "Fix path" below.

## Summary

- b_shift replaces the explicit -INF causal-mask buffer with shifted-KV
  addressing on the single-seq, non-MTP, causal, non-ALiBi prefill path.
- Expected (row gate): PP +3-7% at ub1024 plus the mask-tensor VRAM
  reduction; identity output must remain unchanged at temp 0.
- Measured: PP +0% (gate not met) AND degenerate output (hard fail).
- The path WAS exercised (VRAM peak -174 MB = mask tensor gone), so this
  is a mask-correctness bug, not a gating/build miss.
- First empirical test of this feature; the impl report's "bit-identity"
  was analytic-only (impl work is barred from model runs). The bench
  caught the bug — the gate did its job.

## Bench setup

- Harness: a1b direct-launch, port 8901, fill = 0.9*ctx, n_predict 512,
  reps 3, temp 0, identity gate per leg.
- Geometry: Q6_K 27B, r_ctx 118000, ub 1024, b 2048, kvc q8_0,
  layer-split-hybrid attn:16, tensor-split 0.22/0.78 (3080 / P40).
- Feature binary built from this branch (clean 349/349).
- A/B: `--b-shift off` (leg `a44_off_q6k118`) vs `--b-shift on`
  (leg `a44_on_q6k118`).

## Results

| metric | off | on |
|---|---|---|
| identity (temp 0) | clean counting `1, 2, 3, ..., 14, 1` | **DEGENERATE** `1,2,3,4,5,5,6,7,8,5,7,8,1,2,3,4,1,2,3,1,2,3,1,1` |
| pp rep1 (real fill 104,474 tok) | 226.7 t/s | 225.7 t/s (**+0%**; gate wanted +3-7%) |
| tg mean (3 reps) | 9.39 (9.93/9.18/9.05) | 8.70 (8.77/8.16/9.18) |
| VRAM peak 3080 | 9945 MB | 9771 MB (**-174 MB** = mask tensor eliminated) |

## Diagnosis

- The ON identity matches OFF for the first ~5 tokens, then falls into
  repetition loops — the classic signature of a subtly broken causal
  mask (wrong offset at a boundary, or future-token leak).
- The probe prompt is ~15 tokens, well under one K-tile: points at the
  short-context / tile-boundary region of the shifted addressing.
- The open question from the impl report (IEEE 754 `0 * -INF = NaN` in
  the dense-mask path vs `-INF` in b_shift for masked entries) remains
  unvalidated empirically.
- Report defect: the `b_shift path taken` verification line never
  appears at `-lv 4` (zero log hits on the ON leg) — raise it to a
  visible level so the path is greppable in future benches.

## Fix path before any re-bench

1. Empirical kernel validation with short n_kv on BOTH the MMA_F16 and
   TILE kernels (the bench only exercises the path that is selected for
   this geometry).
2. A tiny identity/unit check on a ~10-token prompt (the failing regime)
   as a pre-bench gate.
3. Path-taken log line at visible level.
4. Re-run this exact A/B (config and harness are stable and reusable).

## Evidence

- Results CSV: `a1bench_results_20261115_011656.csv`
- Per-leg server logs: `a1bench_20261115_011656_a44_off_q6k118.log`,
  `a1bench_20261115_011656_a44_on_q6k118.log`
- Run log: `c1_a44_run.log`
- ON leg MTP stats (cumulative): `#acc rate/pos = (0.543, 0.275, 0.161,
  0.076)` class; no CUDA errors, no asserts in either leg.
- Sanitize: quad-anchored IP + drive-path + username sweep -> zero hits.
