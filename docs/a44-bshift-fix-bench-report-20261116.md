# A44 b_shift fix — A/B bench + applicability audit — 2026-11-16 09:10

## Verdict
REJECT / no merge to prod. The b_shift correctness fix is right and now SAFE (both
legs identity-clean), but the feature is inapplicable on this rig: the served 27B
model is a Qwen-VL VLM using M-RoPE (`ubatch.n_pos = 4`), and the fix's gate requires
`ubatch.n_pos == 1`, so the `--b-shift on` leg never takes the b_shift path and runs
the mask path identically to `off` (PP/TG flat, within noise). Confidence: high.
Flip condition: a 1D-RoPE model bench where the layer-0 probe fires (path actually
taken) AND PP improves by the +3-7% gate.

## Summary
- The A44 b_shift offset bug (hardcoded `0.0f` -> decode queries attended only the
  first token -> garbled identity) is fixed by `b_shift = ubatch.pos[0]` at the
  build_attn_mha call site; kernels untouched.
- The fix also restricts the gate to 1D positions (`ubatch.n_pos == 1`). This is what
  makes it safe: the ORIGINAL build let the path run for this M-RoPE (2D) model and
  produced the garbled output; the 1D-only gate now correctly excludes it.
- A/B bench (main 27B Q6_K @ 118k ctx, ub 1024, MTP self-draft, 2 legs x 2 reps) on
  the confirmed build (commit `1c72390c0`, verified via `llama-server --version`):
  both legs identity-clean; cold prefill PP 228.4 (on) vs 229.5 (off) t/s, decode TG
  9.33 vs 9.36 t/s — flat within noise. The +3-7% PP gate is NOT met.
- The on-leg `b_shift path taken` probe fired **0** times at verbosity=4: the path was
  provably not taken. Supporting: the on-leg VRAM peak is 9876 MiB (mask tensor
  present), versus the 11/15 buggy epoch's 9771 MiB (mask tensor freed when the path
  ran).
- Root cause of exclusion: the model declares M-RoPE (`rope type = 40`,
  `mrope sections = [11,11,10,0]`), so `n_pos_per_embd()` returns GGML_MROPE_SECTIONS
  = 4 -> `ubatch.n_pos = 4`; the gate requires `== 1`. `--b-shift` is a valid parsed
  arg, so this is a real gate-exclusion, not a missing-flag artifact.

## Findings
1. **Correctness fix is sound and verified.** `b_shift = ubatch.pos[0]` makes the
   kernel's implicit causal test (`key_col <= query_idx + b_shift`) reproduce the mask
   (`pos_get(key_col) <= query_abs_pos`). Where: src/llama-graph.cpp:3064-3073. Both
   legs now emit the exact identity string (`1, 2, ..., 14, 1`), versus the 11/15
   buggy build's garbled on-leg. Impact: the correctness gate (bit-identical at temp 0)
   is met. Risk: none observed. Effort: done.
2. **The original bug and the inapplicability share one root cause: M-RoPE 2D
   positions.** A b_shift offset assumes a single 1D causal ordering (key index <=
   query index + offset). M-RoPE positions are 2D (t, y, x); causality is defined on
   the time axis, not a flat index, so the 1D shift is WRONG for them. The original
   gate (single-seq causal prefill, no 1D check) let the path run on this model ->
   garbled output. The fix adds `ubatch.n_pos == 1` (src/llama-graph.cpp:1619) to
   exclude them. Impact: the fix is correct AND safe, but it removes the only model
   class (1D-RoPE) that this rig does not serve. Risk: low (fallback is the original
   correct mask path). Effort: done.
3. **The b_shift path is provably not exercised on the served model.** The gate
   (src/llama-graph.cpp:1618-1621) requires `n_pos == 1`; the main 27B model is a
   Qwen-VL VLM with M-RoPE, so `n_pos_per_embd()` returns GGML_MROPE_SECTIONS = 4
   (src/llama-hparams.cpp) and every prefill ubatch carries `n_pos = 4`. The gate
   returns false and falls back to the mask path (src/llama-graph.cpp:3078-3081).
   Impact: a flat PP is the EXPECTED outcome, not a missed optimization. Risk: none.
   Effort: none to verify (code + log).
4. **Runtime probe confirms the gate decision.** A layer-0 INFO log
   (src/llama-graph.cpp:3074-3077) prints only when the b_shift path is taken. It
   fired 0 times on the on-leg at verbosity=4, directly confirming `bshift_ok` was
   false. The on-leg VRAM peak (9876 MiB, mask present) vs the 11/15 epoch (9771 MiB,
   mask freed) corroborates that the path ran then but not now. Impact: converts
   "on == off" from an ambiguity into a confirmed gate-exclusion. Risk: none (log
   only). Effort: done.
5. **No applicable A/B exists on this rig.** Because the served model is M-RoPE, the
   one scenario A44 targets (1D-RoPE single-seq causal prefill) is absent here, so
   there is no bench that could show the +3-7% PP gain the success criterion demands.
   Impact: the item cannot clear its bench gate on this rig. Risk: none. Effort: n/a
   (would require a 1D-RoPE model, which the rig does not serve).

## Files Touched
- src/llama-graph.cpp (b_shift offset = pos[0]; gate + probe; call site)
- src/llama-graph.h (bshift_ok member; graph-reuse guard)
- src/llama-kv-cache.cpp / src/llama-kv-cache.h (is_prefix_dense_unshifted impl + decl)
- src/llama-memory.h (virtual is_prefix_dense_unshifted on the base memory interface)
- docs/a44-bshift-fix-bench-report-20261116.md (this report, in-repo copy)

## Open Questions
- If/when the rig serves a 1D-RoPE model, re-bench A44 to measure its real PP delta
  (the one scenario where the path actually runs).
- Keep the a44-bshift branch as a reference vs. delete it (recommend keep: the fix is
  correct and reusable; it just has no applicable workload on this rig).

## Evidence
- Build identity: `llama-server --version` -> `version: 0.8.0-dev (build 12153,
  commit 1c72390c0)`.
- Gate closed: on-leg server log (verbosity=4) has 0 hits for `b_shift path taken`.
- Model is M-RoPE: on-leg server log: `rope type = 40`, `mrope sections = [11, 11,
  10, 0]`.
- n_pos source: `n_pos_per_embd()` returns GGML_MROPE_SECTIONS (4) for M-RoPE
  (src/llama-hparams.cpp); the ubatch n_pos is set from it (src/llama-batch.cpp:325,
  :550, :994).
- Bench (results CSV `a1bench_results_20261116_084734.csv`; cold rep1 from the run
  log): off PP=229.5 TG=9.36; on PP=228.4 TG=9.33; both identity `1, 2, ..., 14, 1`,
  status OK. VRAM peak 3080=9876 MiB (on) / 9863 MiB (off); P40=19300 MiB (both).
- `--b-shift` is parsed: common/arg.cpp:1752 (`-bs/--b-shift [on|off|auto]`);
  `cparams.b_shift = params.b_shift` (common/common.cpp:1776).
