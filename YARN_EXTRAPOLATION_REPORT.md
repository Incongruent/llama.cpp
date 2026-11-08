# YaRN extrapolation extension report — context beyond the 262144 training cap

Date: 2026-11-07. Follow-up to `T1_T5_TUNING_REPORT.md` (same capacity model, same fill-rig
conventions). Scope: the E-ladder — YaRN rope-scaling legs at N12 + q4_0 KV pushing usable
context past the model's training cap — plus small completion deltas to the parent report
(fit-variant ceiling probes, harness `ea` passthrough, preset scenario E). Config-only wave;
no deliverable source changes. Privacy posture unchanged: no local absolute paths, no hosts,
no ports, no tool names beyond those already documented in companion reports.

---

## 1. Approach

The gate that stops context growth at 262,144 is `n_ctx_train` in the gguf. Custom rope
scaling changes the gate's meaning in one specific way: when explicit `--rope-scaling yarn`
with a scale factor and an original-context is set, the loader **recomputes the training
cap as `orig_ctx * rope_scale`** (log line: `custom YaRN scaling detected, re-adjusting
n_ctx_train(...)`). Three consequences:

1. YaRN flags are *mandatory* to exceed the cap — without them the server warns
   `n_ctx_seq > n_ctx_train` and lays out raw positions (silent quality damage, no refusal).
2. With them, the overflow warning becomes cosmetic: the re-adjusted cap simply sits just
   above the requested slot size, so the warning reads "the full capacity of the model will
   not be utilized" and can be ignored safely.
3. YaRN costs nothing per token: rope scale is per-position, not per-token, so KV bytes per
   cell and the compute-per-cell slope from the parent report's capacity model are unchanged.

Per-cell costs therefore carry over verbatim: q4_0 attention KV ~1,154.8 B/cell/layer,
draft-tower KV fixed at ~4,101 B/cell, recurrent-state buffer flat (does not scale with
context), compute slope per-cell per quota (smaller N → smaller attention reserve — the
parent report's regression was fitted at N16 and *over*-predicts reserve at N12, which is
how the 280k rung landed inside its predicted failure point).

## 2. Ladder design

At depths past 262k the only continuing shape is the D family (Q6_K + q4_0 KV), and the
quota rung is exactly **N12**: each quota step up hands slow-card weight+KV to the fast
card until fast-side compute caps. Prediction → observation:

| rung | verdict | mechanism |
|---|---|---|
| N11 @ 280k | fail | slow-card first-round layout ≈ 24,837 MiB > 24,576 floor (~260 MiB over) |
| N12 @ 280k | **pass** | 268 MiB fast-card spare at load; the model's +55-MiB miss was the N16-fitted compute slope |
| N13 @ 280k | fail (predicted) | fast-card grows +290 weights/+442 KV against a 268 MiB spare → ~465 over |

## 3. Measured ladder

Fill rig (90% fill of slot, spec draft-mtp n=3, ub 1024 / b 2048, mmproj offloaded to slow
card, fit-aborted umbrella, single-rep):

| leg | scale | slot n_ctx | fill used | PP | TG | acc | identity | peaks (MiB) |
|---|---|---|---|---|---|---|---|---|
| 280,064 | 1.0687 | 280,064 | 247,816 | 127.9 | 6.56 | 0.313 | PASS | 9,495 / 24,260 |
| 290,048 | 1.10637 | 290,028 | 256,448 | 125.7 | 5.99 | 0.263 | PASS | 9,742 / 24,422 |
| 294,048 | 1.12148 | 293,989 | — | — | — | — | **FAIL (echo)** | 9,820 / 24,424 |

- **Max-loadable rung: 290,048** (slow-card margin 154 MiB at fill peak — the tightest
  configuration yet, `-np 1` mandatory as everywhere deep).
- **Mechanical death at 294k is real and logged**: `CUDA error: out of memory` ~25 s into
  the fill, fast card at 9,820/10,240.
- **Quality dies first, and the identity gate caught it**: at scale 1.12148 the temp-0
  probe *echoed the prompt tail* (`Alpha Bravo Charlie. Now count: 1, 2, 3.`) — the same
  echo signature as the parent report's T4 disqualification. Clean counting survives at
  scale 1.0687 and 1.10637. Extrapolation acceptance therefore sits between **1.10–1.12
  scale (≈290–292k tokens)** for this model, independent of the (overlapping) VRAM wall at
  294k. The two walls landing together is fortunate; geometry can't buy past quality here.
- Depth decay on TG: 6.56 → 5.99 t/s over +8.6k cells (−8.7%), ordinary ladder cost.
- The 280k TG read (6.56 at 247,816 cells) *edges* the parent report's D@262 read (6.18 at
  235,929) — deeper but slower config reading faster; both single-rep, inside the ±0.6 t/s
  run-variance band. Noted, not concluded; reps≥3 for any adoption claim.

## 4. Session-integrity post-mortem (operational note)

Two 290k attempts failed with an abrupt "connection closed" signature: server hung
mid-prefill (VRAM pinned stable, no error line in the server log, no driver reset event at
the failure time). Forensics resolved it to **cross-session interference**: a concurrently
confused operator session had spawned its own llama-server on the same GPUs mid-run (a
foreign-resident process with a fresh spawn time was found holding both cards), and the
clean redo completed normally with zero harness or code changes. Operational rule
re-confirmed — *the two benchmarking GPUs are single-tenant*: verify no foreign resident
processes before a run, and gate every long leg (the fit/bench harnesses do).

## 5. Harness and preset deltas

1. **Bench config JSON gained an optional `ea` field** (extra-args array passthrough,
   appended last). Rope-scaling flags are per-leg extra args, so this was the only harness
   change needed for the whole ladder. The fit-probe harness already had the equivalent
   `-ExtraArgs`.
2. **Preset scenario E added**: `[..._290kq4yarn]` — scenario D keys plus the three yarn
   keys at scale 1.10637. The preset header now documents E's guardrails verbatim:
   *never raise the scale past 1.10637 (echo at 1.12148), never lower the origin below
   262,144*, `-np 1` mandatory. The serving wrapper's header comment was updated (A-D →
   A-E). The wrapper forwards preset keys generically (verified: no key allow-list), so
   the yarn keys reach the server intact.

## 6. Parent-report deltas

- Fit-variant ceiling probes, previously unprobed: **(d)** no-override READY at 106,752
  cells; **(b)/(c)** output-only/draft-only READY at 113,152 cells. Fit-only knowledge —
  deep fill benches were not run for those variants (their buffer layouts are byte-
  identical to the (a) family at equal context, so their deep behavior is assumed, not
  measured).
- Everything else in the parent report stands unchanged; scenario D remains at its
  current preset depth (the 262,144-slot top-up stays a one-line pending decision).

## 7. Pending / handoffs

- T7 quality battery (owned elsewhere) — now has a concrete new target: depth-quality at
  the 280k/290k rungs, where extrapolated KV positions see scales up to 1.10637.
- The 262,000 → 262,144 scenario-D ctx top-up needs an explicit decision (margins shrink
  to ~0.5 GB slow-card / ~0.8 GB fast-card at `-np 1`).
- Single-rep transparency: every ladder number above is one rep; the parent report's
  variance discipline (reps≥3 for headline claims) applies equally here.
