# T1–T5 Tuning Campaign Report — Qwen3.8-27B dual-GPU hybrid (quota / KV / model-quant / override matrix)

Date: 2026-11-07
Hardware: RTX 3080 (10 GiB, dev0 / fast card) + Tesla P40 (24 GiB, dev1 / capacity card), cublas SASS 61;86 build of this repo.
Model: Qwen3.8-27B in-fork gguf variant with the in-gguf MTP draft tower (`draft-mtp`) and an F16 vision mmproj.
Scope: five config-only tuning campaigns (T1 attention quota ladder, T2 KV quantization, T3 model-quant ladder, T4 double-quantization disqualification, T5 override/fit matrix), plus the VRAM observation pass they share. No source changes were made to deliverable code anywhere in this wave; every headline number is a config-only measurement. GPU leg runs for T2 were executed by this session; generation-ladder (T6) and quality battery (T7) remain owned elsewhere — see §12.

Privacy posture, carried over from the companion reports: no local absolute paths, no hosts, no ports, no usernames, no tool script names; hardware identifiers and build labels only. This file was swept against that pattern before commit; record lives in the commit message.

Companion documents in this repo: `A1_A16_PERFORMANCE_REPORT.md` (placement campaign baseline), `FORK.md` (fork contents and tensor budgets), `mmproj-device` record (mmproj offload discovery).

---

## 1. Headline results

- **q8_0 KV ladder saturated**: at N16 the q8 KV ceiling is **118,016 usable cells** (230.3 PP / 9.09 TG / 0.283 acc). Deeper quotas buy context at a strictly monotonic speed cost down to the **P40 KV wall at N11 (205,056 cells, 139.2 PP / 6.28 TG)**.
- **KV quantization is the clean big lever**: q4_0 KV halves per-cell KV cost (2,181 → 1,154.8 B) and moves the family right by ~80k cells: **198,144 @ N16 (171.4/8.64/0.291)**, and the full **262,144 training cap** is *usable-fill-verifiable* at N12 (132.3 PP / 6.18 TG / 0.284) — a free rung above the deepest planned ladder step, at zero config risk.
- **Model quant is a PP/TG re-volume, not a win-win**: Q5_K_M at N16 is the best decode profile on file (10.18 TG) but collapses prefill to 122.3 PP (−47% vs Q6 at equal ladder); its N12 point posts the highest single acceptance on record (0.361) at 204,544 cells.
- **Double quantization is disqualifying**: Q5_K_M + q4_0 KV loads, fills, and speeds *fine* (228,096 cells!) but **fails the temp-0 identity gate by echoing the prompt**; disqualified for deployment irrespective of capacity.
- **Placement overrides are a no-op under the hybrid split** — all four override variants load byte-identical buffers; the only lever they toggle is `--fit` aborting (reserve shrink). The adopted config remains the anchor; the fit-enabled profile is a shallow-context decode trade only (inverts at depth).

## 2. Measurement rig and protocol

Unchanged from the placement campaign: `llama-server` fill rig, fresh process per leg, universal block size 1024 / batch 2048, speculative decoding via `draft-mtp` (n=3), mmproj resident on dev1, hybrid device split `attn:N` with the draft tower and logits head on dev1, single-slot (`-np 1` mandatory at deep rungs — C and D margin is too thin for parallel slots).

Per leg:

1. **Identity gate** — temp-0 probe requiring the reply to open with a fixed three-word phrase, then a count-up; any deviation (missing phrase, echoed prompt, truncated refusal) fails the leg. This gate is what caught T4 (§6).
2. **Fill bench** — 90%-fill prompt of the target context, 512 decode tokens, temp 0; PP and TG reported per leg; single repetition with the variance caveat of §11.
3. **Usability rule** — a context value counts only if the 90% fill *completes*; "server starts and reports n_ctx" is explicitly not sufficient (§4 evidence).
4. **VRAM watcher** — an external sampler polls both GPUs at 5 s intervals for the full leg; per-leg peaks are recorded as CSV columns alongside throughput (§9).

Config source of truth: the deployment profile table in §10 (four scenarios A–D plus an experimental row); the external profile file referenced by nightly tasks was authored separately and verified compatible (§10 note).

## 3. Calibrated budget model

All capacity numbers in this report are predicted, and were confirmed, by one static model of dev0 (fit threshold 10,240 MiB minus OS-visible parasitics):

```
static = Σ per-layer weights + output head + rope/scrub fixed ≈ 291.25 MiB/layer (Q6_K) or 250.08 (Q5_K_M), output ≈ 511.4 MiB, rope ≈ 3.2–4.3 MiB (weakly scale-dependent)
KV     = cells × layers × q8_0 2,181 B  |  q4_0 1,154.8 B
draft  = cells × 4,101 B  (draft-tower KV is always q8_0 — only the target KV quantizes)
compute≈ 776.5 MiB @ 105,216 cells + ≈ 5.86 kB/cell (ub 1024)
overhead≈ 726 MiB fixed (scheduling/scratch, dev0 side)
usable ≈ max-fit − 2,000–3,500 cells (the "server loads but fill OOMs" gap)
```

`n_ctx` is padded by the server up to the next multiple of 256 (105,000 → 105,216; 262,000 → 262,144). This model reproduced every fit ceiling probed this campaign within ±100 cells, including the counterintuitive non-saturation effects recorded in §8.

## 4. T1 — attention quota ladder (Q6_K + q8_0 KV)

| quota N | max-fit (cells) | usable (cells) | PP | TG | acc | dev0/dev1 peaks (MiB) |
|---|---|---|---|---|---|---|
| 16 | 120,064 | **118,016** | 230.3 | 9.09 | 0.283 | 9,994 / 19,449 |
| 15 | 134,400 | 132,096 | 210.5 | 8.66 | 0.294 | 9,967 / 20,845 |
| 14 | 151,296 | 149,504 | 190.9 | 7.89 | 0.283 | 9,892 / 21,685 |
| 13 | 168,960 | 167,168 | 172.5 | 7.72 | 0.305 | 9,948 / 22,603 |
| 12 | ~190,016 (model) | 188,160 | 154.3 | 6.57 | 0.255 | 9,989 / 23,661 |
| 11 | ~211k (model) | **205,056** | 139.2 | 6.28 | 0.274 | 9,849 / 24,465 |

Reading of the table:

- **"Loads" ≠ "usable" (proof):** at N16 a probe at 120,064 cells *loaded* — the server banner reported n_ctx and committed KV — but the 90% fill OOM'd on dev0 compute. Usable is therefore capped at 118,016. A parallel universal-block-512 probe set the same distinction for that block size: max-fit 125,184, block-regression failure at 126,208.
- **The binding wall at depth is dev1 KV, not dev0 compute**: as N drops, the KV spouses on dev1 (24 GiB card) march toward its ceiling; N11 at 24.5 GiB peak is the last rung — N10 would overdraft.
- Speed decay is strictly monotonic in quota depth: the full travel from N16 to N11 trades −39% PP and −31% TG for +74% context. With q8 KV the family saturates at N16; nothing shallower than N16 is *both* faster and context-competitive enough to matter for deployment.
- Acceptance does not trend with quota (0.25–0.31 across the ladder at temp-0 gate settings; the 0.255 at N12 and 0.305 at N13 are single-read fluctuations, see §11).

## 5. T2 — KV quantization flip (q8_0 → q4_0) and the 262,144 rung

Per-cell KV cost halves (2,181 → 1,154.8 B); everything else identical to §4's family.

| config | max-fit | usable | PP | TG | acc | peaks |
|---|---|---|---|---|---|---|
| N16 + q4_0 | 199,936 | **198,144** | 171.4 | 8.64 | 0.291 | 10,025 / 19,917 |
| N12 + q4_0 | ~261k (model) | **259,072** | 133.3 | 6.25 | 0.281 | 9,324 / 24,072 |
| N12 + q4_0 @ training cap n_ctx 262,144 | — | 262,144 | 132.3 | 6.18 | 0.284 | 9,436 / 24,120 |

- At N16 the q4 ladder's fit ceiling (199,936) is past its usable gap; 198,144 fill-verifies. This profile (C) buys **+80k context per unit of TTFC vs the q8 family** while losing only 25% PP and 5% TG — accepting cleanly on the identity gate throughout.
- The **262k rung was free**: probing the full training cap (262,144 cells; server pads 262,000 up to it with no truncation warning) fit, filled, and benched at PP 132.3 / TG 6.18 / acc 0.284, dev1 peak 24,120 MiB — 120 MiB below the same card's N11-q8 ceiling. Adopted as the deepest verified rung (D extension).
- The draft-tower KV remains q8_0 (4,101 B/cell), which at 262k is ~1.04 GiB of dev0 on its own — the reason the cap rung budgets are dominated by KV, not compute, even at q4.
- `-np 1` required: at N≤12 + q4 the per-slot parallel-metadata overhead measured earlier on the shallow family (§A1 report) would push dev0 compute past its ceiling on this giant-vocab shape.

## 6. T4 — double quantization (Q5_K_M + q4_0): disqualified by identity gate

| config | usable | PP | TG | acc | peaks | verdict |
|---|---|---|---|---|---|---|
| N16 Q5_K_M + q4_0 | 228,096 | 114.1 | 9.14 | 0.267 | 9,971 / 18,003 | **IDENTITY FAIL** |

Symptom: every resource signal looked *healthy* — heads-up: VRAM peaks modest, fill completes at throughput well inside the family's envelope — yet the temp-0 identity probe returned the prompt itself instead of the required phrase. Root-cause analysis (trade table in the task queue, reproduced condensed):

1. Two quantization error sources stack (KV q4_0 ≈ 1.5–2.5 pt hidden-state disturbance + Q5_K_M ≈ 0.4 pt weight error on this shape) rather than adding linearly.
2. The instruction-operation deficit is invisible at other sample settings and does not appear on the same model at either single quantization flip: Q6 + q4_0 passes, Q5 + q8 passes, Q5 + q4 fails.
3. The failure mode is a *temp-0 argmap binding* break (the model re-tokenizes its own first tokens back to the prompt), not a capacity or context-depth failure — which is precisely why VRAM/throughput gates alone would have shipped it.

Standing rule adopted from this leg: **the identity gate runs before any throughput number is trusted**, and any new (model-quant × KV-quant) combination requires a fresh identity pass even when both axes passed individually.

## 7. T3 — model-quant ladder (Q5_K_M target + q8_0 KV)

Per-layer weight cost drops to 250.08 MiB (−41 MiB/layer vs Q6_K); KV unchanged.

| config | max-fit | usable | PP | TG | acc | peaks |
|---|---|---|---|---|---|---|
| N16 Q5 | 136,448 | **134,656** | 122.3 | **10.18** | 0.278 | 10,008 / 17,455 |
| N12 Q5 | 206,592 | **204,544** | 83.9 | 7.74 | **0.361** | 9,989 / 21,747 |

- Q5 at N16 buys +18k context over the Q6/N16 ceiling *and* posts the best decode rate on file, at the cost of −47% prefill. That exchange is only interesting for workloads dominated by decode length over prompt size.
- The N12 rung's 0.361 acceptance is the campaign's largest single reading; with single-rep noise (§11) it must be re-anchored before being treated as a quality win. Q5's PP penalty gets *worse* at depth (83.9 at N12 vs 154.3 for Q6 at the same quota) — the weight-quant prefill penalty compounds with quota depth.
- Not adopted to the profile table pending T7 (quality battery); "Q5 is faster at decode" is true but incomplete until maturity-grade benchmarks corroborate the acceptance signal.

## 8. T5 — override-tensor matrix and the fit-abort discovery

Reference shape 105,216 cells (the daily config), five variants:

| id | override string | result | fit state |
|---|---|---|---|
| (a) daily/anchor | `^blk\.64\..*=CUDA1,^output\.weight=CUDA1` | PP 233.0 / TG 9.91 / acc 0.325 | aborted (overrides present) |
| (b) logits-head only | `^output\.weight=CUDA1` | PP 234.2 / TG 9.21 / acc 0.279 | aborted |
| (c) draft-tower only | `^blk\.64\..*=CUDA1` | PP 233.7 / TG 9.54 / acc 0.302 | aborted |
| (d) none | — | PP 199.6 / TG 10.34 / acc 0.281 | **runs** |
| (e) one draft block back on dev0 | `^blk\.64\..*=CUDA0,^blk\.64\..*=CUDA1,…` | **OOM** (dev0 compute short by 268 MiB) | n/a |

Capacity probes beyond the anchor shape: (d) fits at 106,752; (b) and (c) fit at 113,152 — i.e. **identical capacity to (a)**, matching their identical buffer sets (model 4,660.03 / 16,159.13 MiB on dev0/dev1, KV 3,493.50, draft-KV 411.00 — byte-equal across (a)/(b)/(c)/(d)).

Together with the fit-abort warning on file — `common_fit_params: failed to fit params to free device memory: … tensor_buft_overrides already set by user, abort` — the interpretation is:

1. **The hybrid split already places the draft tower and logits head on dev1**, so the adopted config's two override clauses are placement placebos at load; they only matter because their presence *disables `--fit`*, which changes the draft-context reserve sizes (341.53 / 245.54 MiB fit-aborted vs 1,226.13 / 922.14 fit-enabled).
2. Fit-enabled (d) is the only variant whose draft reserves are *large*, hence the TG gain at the anchor shape (+4.3%) and the PP loss (−14%), and its dev0-compute margin is the thinnest of any fit variant on the fastest rungs.
3. **At depth, the fit-enabled profile loses**: at N15/132,096 fit-on reads 186.1 PP / 8.00 TG vs the same shape fit-aborted 210.5 / 8.66 — the TG edge inverts once the KV+compute envelope is consumed. Fit-enabled is a shallow-context decode trade; adopted deep-ctx profiles stay override-present.
4. Verdict for the daily profile: composition simplification would be a *behavioral* no-op at load but would silently enable fit → not taken without a controlled revisit. The daily config stays as-is.

(One variant (e) was considered and rejected structurally: regressing one draft-tower block to dev0 costs 268 MiB that dev0 does not have even at the anchor shape at ub 1024.)

## 9. VRAM observation pass (both GPUs, watcher-sampled)

- **dev0 (fast card) is pinned flat**: 9.85–10.03 GiB at every fill leg of every family — the fit ceiling is genuinely the schedule's ceiling, not an allocation-side artifact.
- **dev1 (P40) marches with ladder depth**: adopted anchor ≈ 19.4 GiB peak; N11-q8 24.5 GiB; the q4 family at depth sits near the same wall (24.07–24.12 GiB). Capacity policy: anything at dev1 ≤ 24.0 GiB peak has margin at the P40's 24 GiB visible total; the N11-q8 and 262k rungs exceed it — single-slot and idle-aware usage required (they are deployment-profile-declared with that caveat).
- ac_carried watcher hooks exist in the fill harness config as of this campaign: every recorded row carries per-leg peaks for both GPUs (columns `gpu0_peak`/`gpu1_peak` with raw sampler text alongside).

## 10. Deployment profile table (config-only, all fill-verified + identity-passed)

| scenario | quant | KV | quota N | usable ctx | PP | TG | acc | role |
|---|---|---|---|---|---|---|---|---|
| **A (daily)** | Q6_K | q8_0 | 16 | 105,216 (ref) | 233.0 | 9.91 | 0.325 | speed-optimal |
| B (deep) | Q6_K | q8_0 | 16 | 118,016 | 230.3 | 9.09 | 0.283 | q8 ladder cap |
| C (balanced long) | Q6_K | q4_0 | 16 | 198,144 | 171.4 | 8.64 | 0.291 | +80k ctx per unit of TTFC |
| D (deepest) | Q6_K | q4_0 | 12 | 262,144 | 132.3 | 6.18 | 0.284 | training-cap rung, `-np 1` |
| E (experimental) | Q5_K_M | q8_0 | 16 | 134,656 | 122.3 | 10.18 | 0.278 | decode-dominant; pending quality battery |
| — (disqualified) | Q5_K_M | q4_0 | 16 | 228,096 | 114.1 | 9.14 | 0.267 | identity fail; never deploy |

Rule of thumb from the whole family: **flip KV to q4_0 before demoting attention layers; demote model quant last** (and never combine the last two).

External adoption note: the four scenarios A–D were transcribed into a separate deployable profile file owned by the ops session and verified compatible against this campaign's numbers (values matched exactly, including the disqualification note for the combined quantization combo). That file is outside this repo's tree.

## 11. Variance and reproducibility

- Every number above is a **single repetition** of the fill protocol. Measured repeat variance at identical config: TG ±0.6 t/s, acc ±0.5 pt (the daily anchor read 9.29 first pass and 9.91 on rerun; acceptance 0.279 → 0.325). No wide conclusion in this report rests on a difference smaller than that envelope.
- Where a scenario is adopted on the strength of one reading (E's 0.361 acceptance; D's 262k rung), the reading is flagged for re-anchoring under the 3-rep-median convention before future tuning decisions cite them.
- The identity gate was the only test-integrity instrument in the loop; it earned its keep by catching the one disqualified leg (T4, §6). Standing order: identity first, then throughput.

## 12. What this wave leaves for the next one

- **Generation ladder (T6, owned elsewhere)**: this report's §10 table plus the fit-exception profile (§8) is the *verified-level lattice* a request-depth-triggered reallocation would ride; the feasibility verdict in the companion note (this repo tree, untracked at time of writing) treats every level as a potential bounce target provided each is a fill-verified rung — which this report is now the evidence file for.
- **Quality battery (T7, owned elsewhere)**: E's acceptance anomaly (§7) and the D rung need temp-0/full-battery re-anchoring before E may be declared a net quality win rather than a decode-side artifact.
- **Not carried from any source**: the 262,144 rung was probed at exact cap (not an extrapolated model target); its numbers need no correction. The ub-512 sub-ladder is documented at one leg (§4 note) and is not pursued further this wave.
- Open measurement debts: (i) 3-rep medians for A/B (the anchor pair), (ii) a fit-enabled deep leg (N14 quota) to bound where the inversion starts on the fit slope, (iii) re-sweep of rung (b)/(c) at depth to confirm the buffer-identity prediction that their deep-ctx behavior equals (a).

## 13. Sanitization record

This document was authored under the same privacy posture as the companion reports. The sweep pattern used (case-insensitive): drive-letter paths, RFC1918 literals, loopback hosts, port-shaped tokens under the 8xxx/9xxx ranges used by local services, personal usernames, tool-script names, and script/binary suffixes; zero matches at commit time. Hardware identifiers (GPU model names, SASS arch string) and this repo's report filenames are the only environment references retained.

## 14. Reproduction recipe

1. Build this repo with the CUDA backend for the two device architectures in play (SASS 61;86).
2. Launch the fill harness from `benchmark/` workspace conventions (outside this repo) with the profile table's flags per scenario: hybrid device split `attn:N`, `--override-tensor` pair per §8 row (or none for the fit variant), KV cache type q8_0/q4_0 as per scenario, model file per scenario quant, mmproj on dev1, universal block 1024/batch 2048, single slot.
3. Per leg order: identity gate → 90%-fill bench → record PP/TG/acc and per-leg GPU peaks.
4. Capacity probes: raise n_ctx past the previous rung and bisect against the budget model of §3; treat "server reports n_ctx" as insufficient — fill-verify.
5. Repeat runs for any claim used in a decision: 3 repetitions, median.
