# Campaign report: A44 re-bench, A55 soak eval + revert, A40/A61 deploy + alternation test — 2026-11-16

Scope: the 2026-11-15/16 window of campaign 1 — prod deploy of A60 + A40/A61,
A44 re-bench verdict, A55 `spec-draft-p-min` soak evaluation and revert, the
A60 orphan sweep, and the A40 fork-dump alternation test that exposed a new
defect (A62). All numbers from the prod server logs (host-side `ak_ai` logs)
and the bench harness CSVs/logs under `benchmark/`.

## Verdict summary

- **A44 b_shift: REJECTED (second empirical bench).** Correctness fixed
  (identity gate passes both legs, both epochs), performance gate failed:
  cold rep1 PP at a 104,474-token prompt measured −0.4%/−0.5% for b_shift ON
  across two epochs (gate wanted +3–7%); TG flat-to-slightly-worse.
- **A55 spec-draft-p-min 0.50: NO BENEFIT — reverted.** Tokens/round −9%
  (3.049 → 2.78), draft-token generation −70%, wall TG unchanged at matched
  context. INI lines removed on disk; restart to load the revert deferred to
  the next quiet window (prod was busy with a live client at revert time).
- **A60 kv-stash prune fix: deployed and verified live.** Init prune
  auto-trimmed ~101 GiB on first model load after the deploy; orphan sweep
  completed (see below).
- **A40 fork-dump trigger: DEFECT FOUND (new row A62).** The trigger does not
  fire on a qualifying same-preset alternation; fire-path logging is already
  visible, so the check is either not reached or silently exits.
- **A61 exit unwind: working as designed.** The per-epoch state saves are
  preset-switch exit dumps, not per-conversation dumps; one real 67,604-token
  conversation was preserved mid-session when a preset switch hit it.

## 1. Prod deploy (2026-11-15)

Merged into prod: `a60-prune-fix` (merge `fe8318a1d`) and `a40a61`
(A40 phase-1 same-preset fork-dump trigger + A61 exit unwind, merge
`0d8f07a7d`), zero conflicts. Rebuilt the prod build dir and restarted the
prod router (detached launch, parent WmiPrvSE) into epoch
`P40+3080-PROD-KVSTASH-20261115083914`.

Verified at boot / first model load:

- A55's `spec-draft-p-min 0.50` loaded (startup spec line shows `p_min=0.50`).
- A60 init prune: KV-stash dir free space jumped +100.9 GiB on the first model
  load — the previously silent no-op prune now evicts.
- Temp-0 smoke request returned the expected single token on the expected
  preset.
- VRAM back in the normal envelope (about 9.7 GiB on the 3080, 19.4 GiB on the
  P40 at the 105k preset; slightly under the 118k numbers because the smoke
  request bound the 105k preset).
- Rollback point recorded pre-merge in the host-side `prod_lastgood.txt`
  (pre-merge prod tip `c474f6dbb`).
- Retired the two merged branches (worktrees + branch build dirs removed).
  Fixed a bug in the retire tool on the way: its merged-check regex only
  stripped the `*` branch marker, not the `+` prefix git emits for
  worktree-checked-out branches, so genuinely merged branches failed the check.

## 2. A60 orphan sweep (2026-11-16)

Stash-state file format: each state is a three-file set sharing a stem —
`<stem>.kvstate` (the blob, 0.6–5 GiB), `<stem>.kvstate.tokens` (token
companion used for prefix matching on consult, ~0.1–0.4 MiB), and
`<stem>.kvstate.json` (sidecar, written last = completeness marker).

- Corrected the 2026-11-14 count arithmetic: the "13 companion orphans
  (29 .tokens vs 42 .kvstate)" were actually **13 token-less blob-orphans**.
  The A60 init prune evicted 12 of them (42 → 30 blobs) together with their
  companions.
- One blob-orphan remained: a 645.6 MiB interrupted dump (mtime the previous
  evening, no companions — unrestorable by construction). Deleted manually
  after safety checks (stale mtime, no dump in flight, single file);
  0.63 GiB reclaimed.
- Final state: exactly 29 complete three-file states, zero orphans of either
  kind.
- Remaining engineering gap (pinned by test, `test_prune_companion_files`):
  companion-only files are still neither counted against the byte cap nor
  removed by the pruner. Follow-up if the creep pattern reappears.

## 3. A44 re-bench (2026-11-16) — REJECTED

Two full campaign epochs (06:02 and 08:47), branch build of `a44-bshift`,
reps=2, identity gate temp-0, config `q6k118` (attn:16, 118k ctx, the
production config), bench on port 8901 with prod stopped (rig.lock held,
prod auto-restored and verified healthy after each epoch).

Cold rep1 PP (104,474-token prompt):

| epoch | b_shift OFF | b_shift ON | delta |
|---|---|---|---|
| 06:02 | 229.36 t/s | 228.35 t/s | −0.4% |
| 08:47 | 229.50 t/s | 228.43 t/s | −0.5% |

- Identity gate: identical temp-0 output on both legs in both epochs — the
  NaN class from the first bench is closed; the re-bench is a valid test.
- Performance gate (+3–7% PP): **not met** — measured is consistently
  slightly negative.
- TG: 9.36 vs 9.33 and 9.52 vs 8.99 t/s — within noise, never better.
- The `b_shift path taken` log line is still not visible at the harness
  verbosity (report defect stands from the first bench).
- Verdict: **REJECTED, no prod merge.** Third data point with this shape on
  this rig (A19 demotion, A23 fused-GDN, A44 b_shift): the fused/shifted
  kernel path is not the bottleneck. The harness restored prod cleanly and
  released the lock.

## 4. A55 soak evaluation (2026-11-16) — REVERTED

`spec-draft-p-min = 0.50` was added to all eight 27B blocks in the prod INI on
2026-11-14 and went live in the 02:03 epoch on 2026-11-15 (the 2026-11-14
09:33 epoch had loaded `p_min=0.00` because the INI edit postdated that server
start — data before 02:03 is baseline). Soak epochs evaluated:

| epoch | rounds | tokens/round | draft tokens/round | rate/pos |
|---|---|---|---|---|
| baseline (p_min 0.00) | 23,890 | 3.049 | 3.00 | (0.813, 0.671, 0.565) |
| soak 11-15 08:39 | 10,121 | 2.779 (−8.9%) | 0.887 (−70%) | (0.846, 0.647, 0.513) |
| soak 11-16 06:23 | 7,273 | 2.773 (−9.1%) | — | (0.839, 0.638, 0.499) |

- Tokens/round = (rounds + accepted tokens) / rounds from the cumulative
  spec statistics lines.
- The skip-lever engaged exactly as designed: ~70% of draft positions skipped.
- Wall TG (the fork's per-task late-generation tg prints at ≥100k context):
  soak 14.29 t/s (one task, 44 prints, at 100.8k ctx) vs baseline 13.3 /
  16.76 / 15.98 t/s (tasks at 104–143k ctx) — mid-spread, no improvement at
  smaller-or-equal context.
- Verdict: **no TG benefit, −9% accepted tokens/round — MTP draft generation
  is not the bottleneck** (same shape as A23/A44). Reverted: all 8
  `spec-draft-p-min` lines removed from the prod INI (backup kept alongside).
  The revert goes live at the next prod restart; p_min=0.50 was still live on
  the current epoch at revert time because prod was busy with a live client.

## 5. A40/A61 deploy soak + alternation test (2026-11-16) — defect A62

Soak observation (epochs 11-15 08:39, ~21 h, and 11-16 06:23):

- dump-on-STOP healthy: 75 states saved in the soak epoch (context-checkpoint
  + draft-state line pairs), 2 in the next epoch.
- No re-prefill storms: 24 prefills ≥50k tokens in the soak epoch vs 24 in the
  baseline epoch.
- Zero `fork` log hits — investigated below.

**Preset-switch discovery.** The 75 saves are **not** per-conversation dumps:
A61's dump-on-STOP runs at server exit / router preset switch (dump-all over
IDLE slots), not at per-conversation slot release. This was proven when the
deliberate alternation test (below) bound the 105k preset while a real
67,604-token conversation was resident on another preset: the router switched
presets, the A61 exit unwind saved the full 67k conversation (2.4 GiB state)
mid-session — exactly the designed behavior.

**Alternation test.** Synthetic same-preset fork against the 105k preset:
request A = 200 unique SEQ lines (~4.3k tokens with template) + 255 generated
tokens; request B = the first 150 SEQ lines + 50 divergent ALT lines (~75%
fork point), queued behind A and assigned at A's release. Conditions at B's
slot-assign, all verified from the log: A resident (4,370 tokens; release
keeps prompt tokens), A's state not on disk, kv-stash enabled (digest proven
by the exit dump seconds later), `cache_prompt` default-true, trigger logic
verified correct offline (it would dump for this geometry).

**Result: the trigger did not fire.** Zero fork-dump INF lines, no dump by the
trigger. Code inspection: the fire path already logs at visible SLT_INF (the
earlier "log promotion" assumption was wrong — nothing to promote); the gate
(`cache_prompt && !stateless`) passes for chat requests. So the check is
either not reached or silently exits. Recorded as **new backlog row A62**
(entry-level log line in `kv_stash_fork_dump_check`, rebuild, re-run the
synthetic test in a quiet window, then fix whatever the line reveals).

## 6. Artifacts

- Bench results/log CSVs and per-leg logs under `benchmark/`
  (`a1bench_results_20261116_*.csv`, `a1bench_*_a44_*.log`,
  `a44_campaign_run.log`).
- Alternation test: `benchmark/a40_forkdump_test.ps1` + outputs under
  `benchmark/a40_forkdump/`.
- Host-side server logs: `ak_ai` log dir, epochs
  `P40+3080-PROD-KVSTASH-20261115083914`, `...20261116062349`,
  `...20261116090708`.
- Backlog rows updated: A44 (rejected, re-bench verdict), A55 (revert
  applied), A60 (deploy + orphan sweep), A61/A40 (soak + test findings),
  A62 (new, in-progress).
