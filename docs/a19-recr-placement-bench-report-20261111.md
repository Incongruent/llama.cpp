# A19 bench report — `--layer-split-hybrid` recr placement ladder (2026-11-11)

Sanitized: no local usernames, LAN addresses, or machine names; hardware referenced
generically ("GA10x-class client GPU" = dev0, "pre-Ampere server GPU" = dev1).
Zero-hit sweep before commit.

## Scope

Hardware A/B of the A19 feature (`ba862cecd`, in `prod`): staging the first M
recurrent (GDN) layers onto dev0 via `--layer-split-hybrid attn:N,recr:M`, versus the
benchmark controls. Build: prod tip `444e8ecc4` (version stamp `f89402613`, one
commit stale in build-info; the cc-fix behavior was verified live). Single fill
(0.9x ctx), temp-0 identity gate per leg, 1 rep per leg.

Deviation from the 2026-11-08 plan: benched prod tip directly instead of the
`06e3526ba` snapshot (its base predates the whole A1+ campaign, and the feature had
landed properly as `ba862cecd`); direct server launch per leg instead of the
hotswapper. Some legs were run from a parallel session; two `attn:12` legs were
killed mid-fill during the session handoff and produced no data.

## Results

### Family B — 198k ctx, q4_0 KV

| leg | hyb | TG t/s | PP t/s | dev0 margin at load | status |
|---|---|---|---|---|---|
| B19_r0 | attn:16 | 9.24 | 189.9 | 89 MiB | OK, identity pass |
| B19_r1 | attn:16,recr:1 | — | — | — | **OOM at load** |

Failure: compute-buffer alloc of 1361.01 MiB on dev0 failed (margin 89 MiB).
Family B ceiling: M=0.

### Family A — 118k ctx, q8_0 KV

| leg | hyb | TG t/s | PP t/s | acc | dev0 margin | status |
|---|---|---|---|---|---|---|
| A19_r0 | attn:16 | 9.72 | 212.2 | 0.323 | 165 MiB | OK, identity pass |
| A19_r1 | attn:16,recr:1 | — | — | — | — | OOM (891.51 MiB buffer) |
| A19_r2 | attn:16,recr:2 | — | — | — | — | OOM (same) |
| A15_r0 | attn:15 | 8.78 | 217.0 | 0.291 | 432 MiB | OK |
| A15_r1 | attn:15,recr:1 | 8.84 | 219.0 | 0.294 | — | OK |
| A15_r2 | attn:15,recr:2 | — | — | — | — | OOM |
| A14_r2 | attn:14,recr:2 | — | — | — | — | boots (aborted at fill start, user call) |
| recrALL probe | attn:16,recr | — | — | — | — | OOM 19480 MiB (planned) |

Positive-control marker lines confirmed placement: `up to 16 full-attention and 0
recurrent layers on CUDA0` (control) and `up to 14 full-attention and 2 recurrent
layers on CUDA0` (A14_r2). The probe's marker prints the INT_MAX sentinel; the true
recurrent-layer count is 48.

## Findings

1. **OOM ceiling mechanics.** The GDN compute buffer required on dev0 is
   891.51 MiB at 118k and 1361.01 MiB at 198k — it scales with ctx (~5.9 MB per 1k
   ctx) but is **invariant to the recr count** (identical alloc at recr:1 and
   recr:2). It is the chunk-path workspace, sized by ctx/ub only. The marginal cost
   of each additional staged recr layer is therefore its weights only, ~550 MB
   (measured from the recr:2 OOM delta at attn:15).
2. **Feasible corners exist but are off-control.** Demoting one attn layer frees
   ~925 MB (weights ~800 + KV ~124 MiB at 118k q8_0 + workspace), so attn:15,recr:1
   boots and even attn:14,recr:2 boots. The ladder math extends to attn:0,recr:16
   (~14.8 GB freed vs ~8.8 GB spent + constant buffer).
3. **Exchange rate kills the ladder.** Measured at equal ctx in-session:
   demoting one attn layer costs ~0.94 TG (9.72 -> 8.78); staging one recr layer
   gains ~0.06 TG (8.78 -> 8.84). Every attn:recr exchange nets ~-0.88 TG. Linear
   projection for attn:12,recr:4 lands ~TG 6.2. There is no mechanism for
   superlinear recr gains (GDN per-layer compute is tiny) or sublinear attn costs
   (each demoted layer adds serial dev1 work per token). PP flips slightly upward
   with demotion (+2.3% at attn:15) but cannot compensate the TG loss.
4. **Net verdict vs the attn:16 control:** the best feasible recr corner
   (attn:15,recr:1, TG 8.84) is still -9% TG vs plain attn:16 (9.72). A19 recr
   placement is **REJECTED**; plain `attn:16` remains the production config at
   118k/198k. The feature stays in prod (opt-in, harmless) and may matter on rigs
   with a larger dev0.
5. Caveats: single rep per leg; the -0.94 TG demotion cost carries error bars, but
   the direction matches the A1 campaign history (attn:16 was the winner config)
   and the mechanism (cross-device hops per decode step).

## Artifacts (relative)

- Bench configs: `benchmark/a1b_configs_a19_B.json`, `a1b_configs_a19_A.json`,
  `a1b_configs_a19_attn15A.json` (parallel session), `a1b_configs_a19_A14r2.json`
- Results CSVs: `benchmark/a1bench_results_20261110_222914.csv` (family B),
  `a1bench_results_20261110_224849.csv` (family A),
  `a1bench_results_20261111_002413.csv` (A15 ladder)
- Per-leg server logs + VRAM watcher dirs alongside the CSVs (timestamped pairs)
- Original plan: `A19_BENCH_PLAN.md`, `models_a19_sweep.ini` (partially superseded,
  see deviations)
