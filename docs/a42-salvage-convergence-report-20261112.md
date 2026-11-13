# A42 — KV-stash salvage convergence (checkpoint-rollback restore) — campaign report — 2026-11-12

## Verdict

**KEEP and MERGE.** The converged salvage path is validated on controlled legs
(exact temp-0 identity, −35.3 s prefill on the win geometry) and live soak
traffic (11/17 salvage attempts bounded-tail wins, 0 behavior regressions).
Merged to `prod` as `749a264f5` (fast-forward from the A59 phase-2 tip
`12353cc26`); live-server swap is a separate quiet-window deploy decision.

## Background

Backlog A42 began as "recr catch-up replay" (rebuild 48 GDN recurrent layers
past the last in-RAM checkpoint at a ring-gap fork instead of forced full
re-prefill). The 2026-11-11 depth histogram deferred the in-child replay:
kv_stash (A39/A59) had absorbed the 21-23 do_resets/epoch trigger (0
slot-rewind resets in the A59 epoch, re-prefill median 577 tok).

The workstream re-converged on 2026-11-12 after the A59 phase-2 validation
report: interior-divergence salvage measured NET-NEGATIVE in 4/4 deep events
because phase 2 truncated attention KV at the divergence index F and dropped
the checkpoint appendix — the 48 GDN layers had no recurrent state at F, so
`forcing full prompt re-processing` fired and salvage saved 0 while paying the
state load (net = −(load ms) in every event).

## Design (the convergence)

On interior divergence F against a longer same-digest disk state:

1. Restore the FULL state (attention KV `[0, N_state)` + RS carry at
   `N_state−1`) — unchanged from the verbatim path.
2. Load the checkpoint appendix (each entry carries its own target + draft +
   spec blobs); skip only the live draft appendix (wrong branch tail).
3. The processing loop rolls back to the largest usable checkpoint via the
   existing newest-first search (`pos_min < threshold`, `pos_max <= pos_next`):
   checkpoints with `pos_max > F` are rejected as wrong-branch and erased; a
   `PARTIAL_ONLY` restore replants the hybrid RS (GDN) state at the checkpoint
   position; `n_past = ckpt.n_tokens`; the stale wrong-branch token tail is
   trimmed; decode re-prefills only `[P+1, N)`.
4. No usable checkpoint → `do_reset` full re-prefill (same as before; cost of
   the doomed state load is bounded by the `min_salvage_prefix = 4096` gate).

**Usability boundary (correctness):** a checkpoint at `pos_min = P` holds the
state after exactly `P+1` tokens, i.e. INCLUDING token P. It is usable for a
divergence at F iff `P < F`. F == P must refuse (the ckpt snapshot contains the
diverged token).

## Controlled legs (10,240-tok scratch geometry, seed state 10,263 tok, ckpts @9211/10235)

| leg | divergence | expected | observed | prefill | net |
|---|---|---|---|---|---|
| C1 win | F=9212, ckpt@9211 usable | bounded tail | salvage → ckpt rollback, no forcing-full, wrong-branch ckpt@10235 rejected + erased | **1,028 tok / 4,132.9 ms** | **−35.3 s** vs 39,474.8 ms fresh full prefill |
| C2 boundary | F=9211 == ckpt pos | full re-prefill | full re-prefill (10240 tok) | 39,325.5 ms | 0 (correct refusal) |
| C3 uncovered | F=6029, no ckpt ≤ F | full re-prefill | full re-prefill | 39,186.4 ms | −(state load), known residual |
| C4 reference | empty stash | fresh full prefill | 10240 tok | 39,474.8 ms | — |

- **Identity: C1 == C4 exact** (sha `d5a33cdc…`), same transport, temp 0.
- Stash integrity after C1: the graceful dump re-appended both checkpoints
  (ring re-captured during the bounded re-prefill).
- Unit tests `test-kv-stash`: 94 checks, 0 failures.

## Live soak evidence (a42conv build, prod INI + the prod KV-stash dir
(`kvstash_daily`), port 9712, window 16:25–22:47, ~6.4 h, 49 children)

Watch log: `soak_a42conv_watch.log` (filtered tail of the host-side server log
`P40+3080-SOAK-A42CONV-20261112162551.log`).

| event | count | detail |
|---|---|---|
| salvage attempts | 17 | `salvaging state … (divergence at token F, tail re-prefill via checkpoint rollback)` |
| — bounded-tail wins | 11 | restore replants ckpt appendix, then checkpoint rollback; NO forcing-full line |
| — forced-full losses | 6 | no covering checkpoint ≤ F in the restored appendix → forcing-full; net = −(restore ms) |
| gate rejections | 1 | `prefix 3190 < 4096 min, full re-prefill is cheaper` |
| verbatim full-match restores | 33 | restore 0.5–6.4 s incl. draft payloads (unchanged path) |

- Win example (child 54807): F=8118 vs 167,297-tok state → ckpt@8091 restored
  (`pos_min = 8091, n_tokens = 8092, n_past = 8092`), six wrong-branch ckpts
  (21k–167k) rejected + erased, tail bounded to ~12.6k of 20.7k tokens.
- Loss cluster: F ∈ {4368, 6020, 8902} with appendix ckpts all > F (e.g. F=8902
  vs ckpts @9743/9932/102779/103803) → forcing-full on the same request after a
  2.8–8.8 s restore. Six losses ≈ ~35 s total state-load overhead in the window;
  the same events would also have paid the load under the pre-convergence
  phase 2 (identical refusal economics), so the convergence adds no regression.
- Salvage restore times: 2.8–8.8 s typical; one 71.2 s outlier (cold-cache NVMe
  contention right after model load).
- 1 in-child rewind-gap forcing-full event (task 977) — the residual ring-gap
  class A42 originally targeted; unchanged by this work.

## Economics

- The dominant re-prefill cost on this rig is opencode's client-side
  re-serialization of the transcript between turns (recurring divergence
  clusters at F ≈ 4.4k, 6.0k, 8.1–9.0k, ~15k). When the divergence depth F sits
  below a captured checkpoint position, the convergence converts a full
  re-prefill into (state load + short tail) — the C1 shape (−35.3 s on a 10k
  prompt) scales with `(F − P)` tokens avoided.
- When no checkpoint covers F, salvage is load-cost-only; the 4096 gate keeps
  shallow divergences out.

## Residuals / follow-ups

1. **Positions peek (needs stash-format trailer):** uncovered interior
   divergences still pay the full-state load for nothing; a cheap
   ckpt-positions peek at scan time would refuse these before the load.
   Deferred (llama state-file header does not carry the appendix offset).
2. **Shallow usable ckpts (P ≪ F):** the gate keys on F, not on usable depth;
   the same trailer would let the gate refuse when `(P+1)·4.2 ms < load ms`.
3. **Live-deploy pending:** prod currently serves the pre-convergence build
   (build-a59); swapping needs a hot-swap in a quiet window (the a42conv soak
   was still running at merge time).
4. **F ≈ 6k cluster uncovered by the 8192-grid:** trigger (d) (turn-anchored
   capture at user-turn ends) remains the lever.
5. Related rows queued for the same quiet-window restart: A60 (prune_dir
   no-op fix, ~118 GB reclaim) and A61 (stop-timeout = 120 / dump-on-exit).

## Branch / commit state

- `a42conv` = `749a264f5` on prod tip `12353cc26` (A59 phase-2 gated salvage):
  `tools/server/server-context.cpp` (salvage restore: no truncation, load ckpt
  appendix, skip live draft), `tools/server/server-kv-stash.{h,cpp}` (doc
  wording), `tests/test-kv-stash.cpp` (2 note-wording assertions).
- Merged: `prod` fast-forwarded to `749a264f5`; both `prod` and `a42conv`
  pushed to the `fork` remote.
- Reports chain (host-side reports dir): `2026-11-11-1215-a42-phase0-scope.md`
  (scope), `2026-11-11-2049-a42-depth-histogram.md` (defer verdict),
  `2026-11-12-1530-a59-salvage-test.md` (phase-2 validation),
  `2026-11-12-1626-a42-salvage-convergence.md` (controlled legs).
