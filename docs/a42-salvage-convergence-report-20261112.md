# A42 — KV-stash salvage convergence (checkpoint-rollback restore) — campaign report — 2026-11-12

## Verdict

**KEEP and MERGE.** The converged salvage path is validated on controlled legs
(exact temp-0 identity, −35.3 s prefill on the win geometry) and on a complete
16.9 h production soak (61 stash consults: 31 verbatim restores, 17/28 salvage
wins, net **215 min saved ≈ 21% of epoch wall**). Merged to `prod` as
`749a264f5` (fast-forward from the A59 phase-2 tip `12353cc26`); the prod build
tree just needs the next routine rebuild/restart to serve the merged code.

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

## Live soak evidence — interim window (a42conv build, prod INI + the prod KV-stash dir (`kvstash_daily`), port 9712, window 16:25–22:47, ~6.4 h, 49 children)

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

## Final full-epoch soak analysis (log cut 2026-11-13 09:21, ~16.9 h wall)

Corpus: 77 children (61 children on the four 27B Q6K preset geometries + 16
children on the 9B side model), ~270 requests on the 27B side, 61 stash
consults at fresh-child re-entry — **every consult is cross-preset
(prev child preset != current)**; zero same-preset re-entries ever spawned a
fresh child, i.e. same-preset parent/child alternation never leaves the
resident child (RAM cache + in-RAM checkpoints carry it; stash idle by design).
The 9B side model served 16/16 children with NO stash consult and NO
no-match line: its INI block runs with the kv-stash disabled — the same
config asymmetry flagged in the A39 11-10 epoch, now re-confirmed at scale.

### Scenario verdicts (user's five)

| # | scenario | observed verdict |
|---|---|---|
| 1 | parent → child, different model/preset | every fresh-child re-entry consulted the disk stash (61/61 cross-preset): 31 full-match verbatim restores (all succeeded, exact tail-only prefill) + 28 salvage attempts + 1 gate rejection + 1 genuine no-match (new conversation/geometry) |
| 2 | parent → child, same model | no swap occurs — resident child survives; reuse via RAM prompt cache + in-child checkpoints (1926 cache-reuse events, 31 in-RAM checkpoint rollbacks); zero stash events (by design, nothing left VRAM) |
| 3 | child1 ‖ child2, different models | the 105k ↔ 198kq4 ping-pong (33 consults) is the parallel-subagent churn; the router serializes via kill/reload and the stash absorbs every re-entry; also 105k/118k/249kq4 crossings |
| 4 | child1 ‖ child2, same model | both sessions share the resident child (single-tenant); zero stash events; RAM-cache only |
| 5 | child returns focus to parent | = the return leg: 31 verbatim + 17/28 salvage wins absorbed it; the 9B side model blocks (which never consult) bracket those returns |

### A/B/C answers

- **A) checkpoints used successfully?** 17/28 salvage attempts achieved a
  genuine checkpoint rollback (restored context checkpoint pos_min < F,
  bounded tail, no forcing-full). In-child: 31 RAM checkpoint rollbacks on
  rewind worked as before. 9/28 salvage attempts found NO covering checkpoint
  (F below the smallest captured position) → forced full re-prefill.
- **B) stash used successfully?** 31/31 verbatim restores succeeded; 17/28
  salvage attempts succeeded; 1 rejected by the 4096 gate (correct); 1
  no-match (correct); 2 salvage attempts aborted mid-restore by client
  cancel (74 s restore spent, no leg completed).
- **C) full prefill after using cache/stash?** After verbatim stash restore:
  **0/31** (never). After salvage restore: **9/28** (the uncovered-divergence
  cluster, net −67.2 s). After RAM cache: 3 forcing-full events from in-child
  ring-gap rewinds (the residual class A42 originally targeted; the A59 epoch
  had 0, small-sample noise).
- **D) with/without-stash economics** (61 consult re-entries; per-request
  marginal prefill rate from each leg's own prompt-eval line):

  | class | n | tokens avoided | restore postage | net saved |
  |---|---|---|---|---|
  | verbatim restore | 31 | 1,778,176 | 70.7 s | **+12,417 s** |
  | salvage win (ckpt rollback) | 17 | 156,298 | 79.9 s | **+579.6 s** |
  | salvage loss (uncovered F) | 9 | 0 | 67.2 s | **−67.2 s** |
  | **total (consults)** | 57 | 1,934,474 | 217.8 s | **+12,929 s** |

  Counterfactual: the same 61 re-entries without the stash would have prefilled
  2,278,655 tokens ≈ **18,217 s**; actual (prefill 5,070 s + restore 217.8 s) =
  5,288 s → **215 min (3.6 h) saved ≈ 21.2% of the 16.9 h epoch wall; 71%
  reduction of re-entry prefill cost.** 4 consults aborted by client cancel are
  excluded from the sums.

### Salvage geometry (why 17 won and 9 lost)

- Losses: F ∈ {4,410 … 6,062} — the client's transcript-rewrite clusters sit
  BELOW the smallest captured checkpoint (first gate-ckpt ≈ 8-9.7k); full
  re-prefill anyway (518,409 tok paid). Upper-bound potential if the first
  capture moved to the shared system-prompt head end (turn-anchored capture,
  A42 trigger (d)): ≈ 460,904 tok ≈ 33 min/epoch.
- Wins: F ∈ {8,092 … 9,876} — the recurring divergence depths land just past
  the gate-captured checkpoints at pos_min ≈ 8,091 / 9,564 / 9,664 / 9,743 /
  9,843. The 9,664 position recurs across DIFFERENT conversations (states
  73,985–92,537 tok all rolling back to it) — the shared system-prompt head
  ends there, and the prompt-boundary gate captures it; that single checkpoint
  covers 7 of 17 wins.
- Win legs saved 8.1-9.8k tokens each (−35 to −45 s each at ~4.3 ms/tok,
  minus a 3-8 s restore); loss legs cost the restore for nothing (−3 to
  −8 s each).

### Operational observations

- 15 force-kills (children busy at swap, killed after the 10 s timeout) and
  14 client cancels — A61's measured drag pattern confirmed again at scale.
- Restore times: verbatim 0.5-8.8 s typical; salvage 2.8-8.8 s; outliers
  71.2 s (n=6108, cold NVMe right after load) and 8.8 s.
- Salvage win example: 167,297-tok state, incoming 20,688, F=8,118 → rollback
  to ckpt@8,091, tail 12,596 tok vs 20,688 full; six wrong-branch checkpoints
  (21k-167k) rejected and erased cleanly; stash integrity intact.

## Economics (final epoch)

- The dominant re-prefill cost on this rig is opencode's client-side
  re-serialization of the transcript between turns (recurring divergence
  clusters at F ≈ 4.4-6.1k uncovered, 8.1-9.9k covered). The convergence turns
  covered re-entries into (state load + short tail) and the verbatim path
  (unchanged) carries the bulk of the savings (1.78M of 1.93M tokens avoided).
- When no checkpoint covers F, salvage is load-cost-only (−3 to −8 s per
  event); the 4096 gate keeps shallow divergences out.
- Net effect measured across the full epoch: 12,929 s ≈ 215 min saved, 21.2%
  of wall — the largest single lever measured in the campaign so far.


## Residuals / follow-ups

1. **Positions peek (needs stash-format trailer):** uncovered interior
   divergences still pay the full-state load for nothing; a cheap
   ckpt-positions peek at scan time would refuse these before the load.
   Deferred (llama state-file header does not carry the appendix offset).
2. **Shallow usable ckpts (P ≪ F):** the gate keys on F, not on usable depth;
   the same trailer would let the gate refuse when `(P+1)·4.2 ms < load ms`.
3. **Live-deploy pending:** prod currently serves the pre-convergence build
   (build-a59); swapping needs a hot-swap in a quiet window (the a42conv soak
   has now completed with the convergence build serving the whole epoch — the
   deploy question reduces to making the prod build/launcher point at the
   merged prod tree).
4. **F ≈ 4.4-6.1k cluster uncovered by the 8192-grid:** final-epoch evidence
   quantifies the prize at ≈ 460,904 tok ≈ 33 min/epoch; trigger (d)
   (turn-anchored capture at the shared system-prompt head end) remains the
   lever and now has a measured ceiling.
5. **9B side-model stash asymmetry (NEW, measured):** 16/16 Qwen3.5-9B
   children ran with the kv-stash disabled (no consult, no dump) — same
   config gap as the A39 11-10 epoch side profile; enabling it would extend
   verbatim/salvage savings to side-model alternation.
6. Related rows queued for the same quiet-window restart: A60 (prune_dir
   no-op fix, ~118 GB reclaim) and A61 (stop-timeout = 120 / dump-on-exit;
   15 force-kills re-confirmed this epoch).

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
