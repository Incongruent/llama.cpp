# T9 report — f16 KV: premise correction, corrected cost model, and the equal-context pair verdict

Date: 2026-11-08. Part of the T-series tuning campaign (see `T1_T5_TUNING_REPORT.md`,
`YARN_EXTRAPOLATION_REPORT.md`). Config-only wave; no deliverable source changes. Privacy
posture unchanged: no local absolute paths, no hosts, no ports, no tool names beyond those
already documented in companion reports.

## 1. What this leg was planned to be, and what it actually taught

T9 was queued on a wrong arithmetic premise: that ~8-bit KV (~`q8_0`, 2.0625 B/elem) is
slightly *cheaper* than f16 KV (2.0 B/elem), making f16 a near-free fidelity upgrade. The
first probe batch falsified that instantly: five deep probes (Q6@N16, N13, N11; Q5@N16, N12)
all died with KV-cache allocation failures whose requested buffer sizes decoded to an
**exact 65,536 B/cell** at N16 — 4,096 B per attention layer. The arithmetic recovery:

- `q8_0` is (32 data bytes + 2 scale bytes) per 32 elements = **1.0625 B/elem**, not 2.0625.
- f16 is 2.0 B/elem — so f16 KV is **+88.2% bytes vs q8_0**, the heaviest KV option under f32.
- Measured q8_0 in the parent campaign (2,181 B/cell/layer) reconciles as 2 × 1,024 elems ×
  1.0625 = 2,176 B + ~5 B block padding — this also fixes the parent model's KV constants.

## 2. Corrected KV byte ladder (per attention layer, K+V)

Derived from the two exact f16/q8_0 measurement points plus gguf block sizes (each side is
1,024 elements/layer — consistent with the model's 8 kv-heads × 128 head-dim):

| KV type | B/layer | vs q8_0 | measured? |
|---|---|---|---|
| f32 | 8,192 | +277% | (model only) |
| **f16** | **4,096** | **+88%** | **exact: 65,536 B/cell at N16 (twice)** |
| q8_0 | 2,176 | 0% (baseline) | 2,181 (parent campaign) |
| q6_K | 1,680 | −23% | (theory; T10 probe target) |
| q5_K | 1,408 | −35% | (theory; T10 probe target) |
| q4_0 | 1,155 | −47% | 1,154.8 (T2 measured) |
| q4_K | 1,152 | −47% | (theory) |

Consequence: **f16's usable context is ~43–55% of q8_0 at every quota**, so a deep f16 deck
is geometrically impossible — the T9 deck was re-scoped on the spot to the equal-context
pair benchmarks at shallow rungs (which is the piece with actual scientific value: a
lossless-KV anchor for the T7 KL pass).

## 3. Fit ceiling experiments

- Five deep probes at q8-equivalent rungs (+2.5–3% offsets): all failed at the predicted
  inflated KV sizes (alloc requests 7,696 / 8,801 / 9,152 / 8,752 / 9,924 MiB on the fast
  card vs a ~10,240 MiB cap) — the corrected model was confirmed *by its own failures*.
- Reality range (both READY, both with exact KV confirmation):
  - Q6_K + f16 + N16 @ 50,944 cells: fast-card KV 3,184.00 MiB, compute reserve 343.53 MiB.
  - Q5_K_M + f16 + N16 @ 55,296 cells: fast-card KV 3,456.00 MiB, compute reserve 352.03 MiB.
- The two same-N16 compute points (343.5 @ ~51k, and the parent campaign's 776.5 @ ~105k,
  both f16-independent) give a two-point compute slope ≈ **0.008 MiB/cell at N16** — slightly
  heftier than the parent report's N16-fitted estimate; predicted fit ceilings ≈ 66k (Q6)
  and ≈ 75k (Q5) cells.

## 4. Equal-context pairs (the verdict)

Fill rig (90% fill, spec draft-mtp n=3, ub 1024 / b 2048, mmproj to slow card, fit-aborted
umbrella, single-rep):

| pair | fill cells | KV f16 (fast-card peak) | KV q8_0 (fast-card peak) | delta |
|---|---|---|---|---|
| Q6_K @ 62,080 slot (54,965 filled) | .249 / **9.44** TG / 240.4 PP | .245 / 9.21 / 244.1 | TG +2.5%, PP −1.5%, **+1,656 MiB** |
| Q5_K_M @ 71,680 slot (63,046) | .259 / **10.16** / 231.6 | .260 / 9.92 / 239.7 | TG +2.4%, PP −3.4%, **+1,898 MiB** |

- Identity temp-0 gate: **PASS on all four legs** (clean count-ups, no echo) — worth noting
  because KV fidelity now has a measured "does not break behavior" point adjacent to the
  T4 double-quantization failure.
- Acceptance is indistinguishable between KV f16/q8_0 in both pairs (−0.001..+0.004).
- TG deltas (+0.23 / +0.24) sit inside the ±0.6 t/s single-rep variance band — directionally
  consistent (lossless KV slightly faster decode) but **do not quote as wins**; 3-rep
  before any claim.
- PP cost −1.5..−3.4% is likewise small; the decisive fact is the byte bill: ~1.9× KV bytes
  for at-best-noise-level TG.

## 5. Verdict + handoffs

- **f16 KV = reference deck, not a production shape.** Production KV stays `q8_0` (shallow/
  mid) / `q4_0` (deep), per the parent report's ladder. The preset board was **not** touched.
- **T7 gets its KV-side anchor**: run the two f16 legs' fills as the lossless reference at
  ~62k/71k — the KL ladder can now grade q8_0 → q6_K → q5_K → q4_0 against a zero-error
  KV point at the same shape.
- **T10 queued**: the actually-useful middle rungs (q6_K KV ≈ −23% bytes, q5_K ≈ −35%),
  at N16/N13/N11 with the corrected byte model verified at the first probe — same
  discipline (identity gate before scoring, exact-KV confirmation like §3).
