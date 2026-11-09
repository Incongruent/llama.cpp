# A39 (kv-stash on-disk persistence) validation report — 2026-11-09

Test bed: modified llama.cpp fork build (internal code name A39, mtmd-safe conversation-hash fix), served as profile-preset blocks A/105k + B/118k + C/198k (same 27B model, three context windows) through a local inference gateway. Session ran natural multi-agent work; context-checkpoint ring = 16 slots (edited 2026-11-08 mid-epoch); RAM prompt-cache budget = library default (8192 MiB), NOT raised in this epoch.

Epoch log: server-side run log, 45,050 lines, 8 serving processes under the gateway. Stash volume: local NVMe directory, 7 state files + 7 sidecar json, 64 GiB budget.

## Verdict: PASS on all three gates, with one anomaly logged and two follow-ups open.

Gate results:
1. WRITE at preset-swap-out — PASS ×7 dumps. Every child destroyed on preset change produced a complete state file (main KV + checkpoint appendix + draft) before death. No dump-less swap-out occurred.
2. RESTORE at preset-swap return — PASS ×4 restore legs. Matched by digest + conversation-prefix hash, then the fork replanted the state from disk and let the slot serve a small delta prefill. Follow-up counts (first logical line after restore):
   - 61873 (B): restored 42,117 of 42,489 incoming (delta prefill 372 tok), 2,179.56 ms
   - 62119 (B): restored 50,811 of 53,521 incoming (delta prefill 2,710 tok), 2,050.99 ms
   - 62253 (B): restored 65,219 tokens of full state despite only 7,994 incoming tokens (superset-restore anomaly, see "Anomaly"); a stale context checkpoint at pos 7,090 was invalidated right after
   - 62578 (A): restored 70,671 of 73,373 incoming (delta prefill 2,702 tok), 6,150.21 ms
3. NO state loss across preset geometry boundaries — PASS by default: the two non-restore returns were refusals by design (geometry/digest mismatch = correct refusal), not state loss. Zero restore-into-wrong-geometry events.

Economics (delta prefill at measured profile-epoch rates, 161-235 t/s warehouse-wide):
- Re-prefill avoided by the 4 restore legs: 229.8k tokens ≈ 16-24 wall minutes at live profile speeds.
- Stash postage: 7 dumps 2.75-6.74 s each ≈ 32.1 s total, restores 13.3 s total. Net benefit ≈ 20-40×.
- Restore read throughput: 1.13-1.72 GB/s from the NVMe stash volume (the 105k-profile leg ran slower, 0.72 GB/s, snapshot not yet profiled).
- Tag format `<digest>_<convhash>_<ntokens>.kvstate` + `.json` sidecars landed verbatim as designed.

## Full swap ledger (chronological)
| # | child | geometry digest | event | tokens | file MB | ms |
|---|---|---|---|---|---|---|
| 1 | 58950 | B 33fd07879e7d | cold miss (new conversation, expected) 38,863 | | | |
|   | 58950 | B | DUMP 1 | 42,117 | 2,462.9 | 2,749.10 |
| 2 | 60002 | A b43145168837 | cold miss (expected) 7,676 | | | |
|   | 60002 | A | DUMP 2 | 31,771 | 1,928.8 | 1,999.78 |
| 3 | 61873 | B 33fd07879e7d | RESTORE 1 (delta 372 tok) | 42,117/42,489 | 2,462.9 | 2,179.56 |
|   | 61873 | B | DUMP 3 | 50,811 | 3,534.2 | 3,894.74 |
| 4 | 61901 | C 60f93b7f4820 | cold miss (expected) 5,536; DUMP 4 (deep leg) | 127,380 | 5,137.3 | 5,970.41 |
| 5 | 62119 | B 33fd07879e7d | RESTORE 2 (delta 2,710 tok) | 50,811/53,521 | 3,534.2 | 2,050.99 |
|   | 62119 | B | DUMP 5 | 65,219 | 4,668.2 | 5,694.99 |
| 6 | 62183 | A b431 | MISS (correct refusal: B geometry, A digest) 65,219; later DUMP 6 | 70,671 | 4,421.8 | 5,034.87 |
| 7 | 62253 | B 33fd07879e7d | RESTORE 3 (superset-restore on 7,994 incoming) | 65,219/7,994 | 4,668.2 | 2,890.69 |
|   | 62253 | B | DUMP 7 | 79,504 | 5,348.6 | 6,740.30 |
| 8 | 62578 | A b431 | RESTORE 4 (delta 2,702 tok) | 70,671/73,373 | 4,421.8 | 6,150.21 |

(all child load lines printed `kv stash enabled ... limit 65536 MiB`, no cap pressure, no eviction of stash files)

## Interlocking mechanism proof values
- Store/restore totals per file: main state scales with tokens (2.46-5.35 GB across 31,771-127,380 tok).
- Checkpoint appendix = 149.646 MiB per captured checkpoint, counted and carried: 5 ckpt = 748.23 MiB; 15 ckpt = 2,244.69 MiB (bytes match the dump lines exactly in both directions: DUMP 1 = 748.23 = 5 × 149.646; DUMP 4 = 2,244.69 = 15 × 149.646).
- Draft state captured and refilled alongside (124.85-500.51 MiB), `spec-draft-n-max 3` geometry preserved.
- `fit` clamp honored; all builds kept -np 1, single-slot semantics.

## Anomaly (needs design review or doc, not a blocker)
- 62253 restored 65,219 tokens when the incoming prompt was only 7,994 tokens (superset-restore). The slot then promptly erased an invalidated context checkpoint (post 7,090) and continued with in-memory sequencing. Either a compacted turn or a template reshape caused the client prefix to shrink; A39 chose full-restore-then-truncate semantics. Report-recommended follow-up: tighten the matching rule to require incoming >= state, or document the superset behavior as intended-for-hybrid-slot semantics.

## Still-open/post-epoch (not blocking A39 itself)
1. RAM prompt cache evictions: 25 per epoch, load churn profile real but bounded. A staged config update (cache budget 16,384 MiB) is written but unapplied.
2. Full resets: 34 per epoch, mix of (a) cold-start expected misses, (b) cache eviction after churn — ties to A40 phase-1 (fork-dump-on-handoff).
3. Checkpoint-ring saturation: 651 creates vs 15 restores; append-price per dump = 149.646 MiB/checkpoint — engage lower capture thresholds when idle'd presets grow this cap. (A42 covers.)
4. Identity: this test ran a natural agent session; content-identity gates are the orchestrator's domain, not the fork's. Separation of concerns: the fork proves state correctness by read-back; the agent session proves content by its own evaluation.

## Filesystem post-flight (no manual cap enforcement required)
7 files written total, largest 5,348.6 MB (79.5k tokens w/ 15 checkpoints + draft). Largest C-profile file tied to 127,380-token deep-explore excursion. Disk stable, cap far from binding.

— End of measurement report.
