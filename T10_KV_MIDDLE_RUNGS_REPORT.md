# T10 Report — KV Middle Rungs (q5_0 / q4_1) and the q4_1 Max-Fit Hunt

Completed: 2026-11-08. Backlog row T10 closed. Scope directives honored for this item:
no f32, no iq4, no bf16 KV legs (user-confirmed; iq4_nl skipped), KV space restricted to
q5_0 and q4_1, with the deep max-fit hunt run on q4_1 alone.

## 0. TL;DR

1. The original T10 premise was wrong: `q6_K` / `q5_K` are not valid KV cache types in
   llama.cpp; the server rejects them at argument parse time with the allowed list
   `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`. The deck was re-scoped to
   q5_0 (−35% bytes vs q8) + q4_1 (−41%).
2. The KV byte model is now verified to the byte at every informative probe: dev0 KV per
   token equals (full-attention-layer quota) x (per-layer bytes), with the compute slope
   re-confirmed at 0.06 percent accuracy.
3. The KV precision ladder is monotonic and predicts fits within ~1 percent. No surprise
   middle rung exists: q4_0 keeps the deep-context crown (259k+, and 290k with YaRN),
   q8 stays the quality pick below ~180k, and the new useful rung is q4_1 at ~249-256k.
4. Max fit with q4_1 alone: ~256,000 tokens at the 13-attention-layer split (first EXIT at
   257,024); ~185,344 at the 16-layer speed split; the 11-layer split technically loads the
   full 262,144 training cap but with only ~10 MiB of dev0 margin — a fill-time razor, not
   recommended and not benched.
5. A new INI scenario (q4.1 at 249k context, benched twice, identity PASS both times) is
   now part of the hot-swap preset board.

## 1. Premise correction

The T10 backlog entry (written before the T9 correction propagated) assumed q6_K/q5_K KV.
Actual server behavior: instant argument-parse failure for any of q4_K / q5_K / q6_K, with
the accepted types enumerated in the usage line. This agrees with the T9 finding that the
KV byte costs per element are NOT monotone across families (q8_0 is 1.0625 B/elem, not
~2.06): the middle rungs by real byte cost are q5_0/q5_1 (-35 percent vs q8) and q4_1
(-41 percent), not q6_K/q5_K. Backlog text updated accordingly at close-out.

## 2. KV byte model — verified ladder (per full-attention layer, K+V, 1024 elems/side)

| KV type | theory B/layer | measured check |
|---|---|---|
| f32 | 8,192 | n/a (out of scope by directive) |
| f16 | 4,096 | exact, two probes (T9) |
| q8_0 | 2,176 | 2,181.4 incl. padding (T1 era) |
| q6_K | 1,680 | NOT a valid KV type |
| q5_0 / q5_1 | 1,408 | 1,408.6-1,409.5 at N16/N13 (this item) |
| q4_1 | 1,280 | 1,280.0-1,281.4 at N16/N13/N11 (three independent sizes) |
| q4_0 | 1,154.8 | measured (T2 era) |
| iq4_nl | ~1,152 | skipped by user directive (same class as q4_0) |

Additional confirmation: between the two N16 first-round-EXIT points the dev0 layout slope
came out 26,640 B/token at the 16-layer split vs theory 26,625 (KV 1,280 x 16 + compute
slope 6,145) — 0.06 percent error. Predictive fit placement is now trusted to ~1 percent
against the arithmetic model, with the caveat below about probe machinery margins.

## 3. Fit-ceiling bisection (q4_1 alone — the requested hunt)

Method: chained single-probe runs (fit + slot entry + SMI snapshot, identity-checked on
READY states), one knob moved at a time; EXITs decode the exact over-budget amount from the
failed allocation size in the log.

| split | READY (highest) | EXIT (lowest) | verdict |
|---|---|---|---|
| attn:13 | 254,592 (slot 254,720) and 255,808 (slot 256,000) | 257,024 | max fit ~256,000 — 1,024 tokens under the 262,144 cap |
| attn:16 | 183,552 and 185,216 (slot 185,344) | 186,880 | max fit ~185,344 |
| attn:11 | 262,000 (slot 262,144 = full cap) | — | loads the cap but ~10 MiB dev0 margin + failed-then-recovered intermediate alloc — fill razor, not benched |

Machinery caveat that shaped the method: the first-round compute reserve makes EXIT
margins in the probe path, and the "failed alloc then trimmed re-init" recovery triggers
near the boundary sometimes and not others; extrapolating from EXIT deltas alone gave
boundaries sharply below known-good datapoints, so the ceilings were pinned by
straight bisection, not extrapolation.

q5_0 status after the scope narrowing to q4_1: fit >= 168,192 at the 16-layer split
(probe READY, margin left) and left unclosed by design; its 11-layer probe was unclean
(failed mid-load compute-meta alloc, razor-adjacent) and was not chased.

## 4. Bench results (Q6_K weights, adopted override, ub 1024 / b 2048, mmproj on CUDA1)

All legs identity-gated (temp 0 echo) PASS before scoring; peaks are SMI sighting maxima
during the whole leg. Format: PP / TG t/s / MTP-acc, fill tokens at 90 percent of context.

| leg | ctx | fill | PP | TG | acc | dev0/dev1 peaks | note |
|---|---|---|---|---|---|---|---|
| q5_0 165k | 165,120 | 145,651 | 221.8 | 8.47 | 0.267 | 9,850 / 19,577 | rep2 in a prior aborted run: 221.0 / 8.92 / 0.283 |
| q5_0 219k | 219,136 | 193,267 | 156.0 | 7.56 | 0.336 | 9,852 / 22,907 | |
| q4_1 177k | 177,152 | 156,025 | 216.5 | 9.31 | 0.31 | 9,923 / 19,647 | earlier rep: 215.6 / 10.15 / 0.346 |
| q4_1 249k | 249,088 | 220,563 | 137.4 | 6.62 | 0.246 | 9,934 / 23,287 | earlier rep: 137.8 / 7.17 / 0.291 |
| q4_1 238k | 238,016 | 210,008 | 129.4 | 5.70 | 0.235 | 9,093 / 24,280 | 11-layer split leg |
| q4_1 253k (max rung) | 252,928 | 223,949 | 136.6 | 7.66 | 0.336 | 9,947 / 23,347 | identity PASS |
| q4_1 182k (max rung) | 182,272 | 160,501 | 181.1 | 9.39 | 0.291 | 9,907 / 19,677 | identity PASS |

Reading of the middle rungs against the incumbents:

- q5_0 165k sits in the same TG range as q8's 134-167k rungs with more context per tier;
  it is the "same speed, more room" middle point.
- q4_1 253k trades ~13 percent TG vs q4_0's 259k rung for the deeper-fidelity anchor and
  a max-fit ~3,000 tokens higher than any verified q4_0-family config (259k + YaRN
  extrapolation aside, which is a research scenario).
- q4_1 177k is the standout sealed-deal config: double-digit TG (9.3-10.2) with acc at or
  above the q8 anchors at 30-45 percent more context than the q8 speed rung.

## 5. Variance and evidence-integrity notes

1. Same-config repeat drift measured this item: TG +0.45-0.84 t/s between two runs of the
   identical leg (165k, 177k, 249k), same direction family as f16-era observations, but the
   largest seen yet. Standing rule upgraded in practice: doc-grade claims need the 3-rep
   protocol; single runs are trend-grade only.
2. Two bench pipelines were aborted mid-flight for unrelated machine-side needs and
   re-run cleanly after zombie/VRAM checks (one real abort left a short in-flight fill;
   zero llamas survived, verified before each retry). The aborted-run leg that had
   completed paid itself back as a free second rep. No cross-session interference
   signature (no foreign server, no driver events at run time) was seen in any T10 leg.
3. The 11-layer q5_0 probe joined the known "unclean" class: failed intermediate alloc
   followed by successful re-init (matches the pattern seen on other family probes). Any
   future leg that prints that pattern AND ends near sub-300-MiB margins gets the same
   treatment: recorded as razor, not benched without an explicit decision.

## 6. Verdicts

- Depth: q4_0 remains the deep champion (259k production-rung, 290k research ring via
  YaRN). The middle rungs do not beat it at depth; they were never expected to.
- Fidelity-anchored middle: q4_1 at 249k (scenario-ready, benched twice) and 253k
  (max-rung). Suggested use: long-horizon agentic work where q4_0's budget is acceptable
  but a slightly tighter KV is worth ~5-8 percent TG.
- Speed-with-room middle: q4_1 177k / q5_0 165k; both at or above q8-class TG with
  meaningful context gains.
- Reference ground truth: unchanged by this item (f16 deck, T9).

## 7. Deliverables and artifacts

- New INI scenario named after its head anchor (249k, q4.1 KV, 13-attention-layer split,
  ctx key 249000 -> slot 249,088): preset board now includes it as scenario F, with
  comment guardrails (verified ceiling ~256k — do not raise ctx with these keys).
- Landmine fix discovered during scenario placement: the D scenario block carried a
  16-attention-layer split that contradicted its own measured config and header comment
  (N12). That block could not have loaded at 259k (dev0 over budget by ~840 MiB). Fixed
  to the measured 12-attention-layer split.
- Backlog T10 closed with the premise correction and results inline; dashboard file
  (context-vs-speed chart) regenerated for this item and its byte table corrected to the
  verified q4_1 = 1,280 figure.
- Evidence: fit-probe logs for the q4.1 bisection series (244-262k region and 178-196k
  region), six bench runs across four CSVs dated 2026-11-08, and the aborted-run logs
  retained for the variance note.

## 8. Open items

- Single-rep drift now peaks at 0.84 t/s; promote the 3-rep protocol for any future
  claim-based comparison, not just doc-grade ones.
- q5_0's 16-layer fit ceiling is known only as "> 168,192"; closing it would mirror the
  q4_1 bisection (~15 minutes of probe time) but has no INI impact today.
- The 11-layer split at the full training cap remains a curiosity: loads, fills may not.
  Bench it only if a razor-filling exercise is ever wanted (not recommended).
- Speed/quality separation for the middle rungs (KL-style grading vs the f16 reference)
  belongs to the standing quality battery item and inherits its open status.
