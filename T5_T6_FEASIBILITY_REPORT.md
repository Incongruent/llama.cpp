# T5/T6 Feasibility Report — override-tensor variants (T5) & ctx-driven layer reallocation (T6)

Date: 2026-11-07
Scope: T5 (`--mmproj-device` CUDA1 + override-tensor permutations for draft tower and logits
head) and T6 (dynamic reallocation of attention layers/weights by request context size) —
feasibility against the current fork codebase, approach, implementation/testing plan,
tradeoffs, and verdicts. This document covers analysis only; the backlog rows T5/T6 own the
execution status. GPU leg runs for T2 remain owned by a separate session; nothing here
spawns anything.

All code references are repo-relative. Privacy posture unchanged: no local absolute paths,
no hosts, no ports, no tool names beyond the harness/binaries already documented in
companion reports.

---

## Part I — T5: override-tensor variants with mmproj on CUDA1

### I.1 Codebase facts (read directly from source)

1. `-ot/--override-tensor` parses `pattern=buffer-type` pairs into
   `params.tensor_buft_overrides` (common/arg.cpp:2704). Matching is by tensor-name regex at
   tensor-load time — it works on any tensor, including the in-gguf MTP draft tower blocks
   (they are ordinary `blk.N` entries with `N >= n_layer`).
2. `--spec-draft-override-tensor (-otd)` (common/arg.cpp:4131) targets an *external draft
   model's* tensors. With Vk's in-gguf MTP tower (draft-mtp), the main `-ot` catches the
   draft blocks; `-otd` is expected to be a no-op here (verify once, then drop).
3. Hybrid split impl (src/llama-model.cpp:1760-1832): `attn:N` assigns the first N
   full-attention layers to dev0 by a load-time quota; everything else — recurrent layers,
   the MTP draft tower, the logits head — defaults to dev1. Precedence notes:
   - hybrid split ignores `--tensor-split` when active;
   - the logits head is assigned via `get_layer_buft_list(n_layer_all)` → dev1 by default
     under hybrid;
   - `LLAMA_OUTPUT_PIN` (A13) explicitly pins the logits head to the last CUDA device, with
     the documented caveat that it *overrides* an explicit `output.weight` override-tensor
     (last placement wins);
   - `LLAMA_MTP_PIN` (A5) pins draft-tower layers *and their KV* to the output device.
4. Measured tensor sizes on this model (FORK.md suite + HYBRID doc): one attention block
   291.25 MiB weights + 218.35 MiB KV per ctx-105k cell ≈ 2.18 kB/cell-layer (q8_0 KV);
   `output.weight` ≈ 511 MiB; draft tower ≈ 480 MiB; mmproj worst-case ≈ 1,158 MiB.
5. Campaign evidence already on file: QA A1dig measured "attn:13–15 with output kept on
   CUDA0 → all OOM" — the logits head alone is the larger dev0 hog. A5 measured draft-tower
   placement as *speed-neutral* (draft-run TG/acceptance identical both modes at the 9B
   shape); A13 measured output pin as *harmful* (−14% TG) on the 9B `-ts` shape but
   *capacity-liberating* at 27B (A1dig both-moved configs). The clean per-member speed/ctx
   attribution at the 27B hybrid shape — which T5 adds — has not been measured.

### I.2 Feasibility verdict

**HIGH** — config-only. Every member of the experiment space is a command-line switch or a
models.ini value in an already-shipped flag. Zero source changes, zero build risk. The only
"code" work is extending the existing bench config schema, and the harness already accepts
per-config `hyb/ovr/mmp/kvc/m` fields.

One genuine hypothesis to test, which changes the *interpretation* of the adopted prod
config: since the hybrid split already places the logits head and draft tower on dev1 by
default, the adopted config's `^output\.weight=CUDA1` clause may be a **placebo** (the
override matches a tensor that would land on CUDA1 anyway). Variant legs will confirm —
if confirmed, prod config simplifies and one fewer override-pattern failure mode exists.

### I.3 Approach: two-axis measurement

Axis A — **speed isolation** (same geometry, move one member): run all variants at the
common reference shape that is feasible for the *worst* variant, 90%-fill bench, compare
PP/TG/acc. Differences between variants are then attributable to placement alone, not to
ctx drift or quota changes.

Axis B — **capacity isolation** (same quota, placement maximizes fit): for each variant,
derive its max-usable ctx from the calibrated budget model, one confirm leg each. This
yields "how much ctx does this placement buy".

Variant matrix (mmproj on CUDA1 fixed throughout):

| id | draft tower | logits head | -ot content | purpose |
|----|-------------|-------------|-------------|---------|
| v-adopted | CUDA1 | CUDA1 | `^blk\.64\..*=CUDA1,^output\.weight=CUDA1` | anchor (current prod) |
| v-no-out | CUDA1 | default | `^blk\.64\..*=CUDA1` only | placebo test — should tie v-adopted |
| v-out0 | default | CUDA0 | (no override; hybrid default) | rebalance capacity→dev0? (predicted infeasible at deep quotas) |
| v-both0 | CUDA0 | CUDA0 | none | the max-capacity dev0 variant — the ctx ceiling of the family |
| v-otd | CUDA1 | CUDA1 | `-otd` probe (one fit-only leg) | confirm `-otd` inertness with draft-mtp |
| v-pin | env-only variant of v-adopted (`LLAMA_MTP_PIN=1` / `LLAMA_OUTPUT_PIN=1`) | | env switches | in-tree explicit pins for tooling comparability |

Interleave quota points at the promising variants only (e.g. deep quotas for v-adopted
already exist from T1; re-anchor one quota for v-no-out to prove equivalence).

### I.4 Predicted outcome (to be falsified empirically)

- v-no-out == v-adopted at every measured point (placebo confirmed) → simplify prod config.
- Draft tower placement: speed-neutral (A5 precedent) → capacity-only lever; on dev0 it is
  a bad trade (−480 MiB ≈ −14.5k cells of ctx for no speed gain).
- Logits head placement: the only member with a real speed bias. Direction of the bias at
  27B hybrid is genuinely open (9B precedent says head-on-fast is *slower*; A1dig says
  both-moved beats baseline, but that number is confounded by quota change). If head-on-dev0
  proves faster at equal TG, prod may want the head back on dev0 *when ctx budget allows* —
  i.e. T5 defines a second ctx-vs-speed dial orthogonal to the T1 quota dial.

### I.5 Implementation plan (half a day to a day of GPU legs)

1. Harness: no changes needed (per-config `ovr`/`mmp` fields already parse). Add the two
   env pins via `-ExtraArgs`-style env injection only if v-pin legs are wanted.
2. Axis A legs at the reference shape (5 legs ≈ warmup+identity+fill each).
3. Axis B: budget-model targets per variant, one fit probe + one usable fill leg each
   (reuse the calibrated model from T1; KV 2.18 kB/cell/layer, compute reserve
   ≈ 5.86 kB/cell, fixed overhead ≈ 726 MiB — boundaries land within ±1 leg).
4. Deliverables: CSV rows (existing harness), backlog T5 row updated with the verdict table,
   and a diff-style note if prod config simplifies (models.ini change requires explicit
   approval — running the sweep does not).

### I.6 Testing plan

- Temp-0 identity probe identical across all variants (existing convention).
- VRAM peaks both GPUs recorded automatically by the watcher (T1-added).
- Negative control: one leg with mmproj default-resident (no `--mmproj-device`) at the
  reference shape, to reconfirm the mmproj contributes ~1.1 GiB of dev1 footprint and no
  measurable speed effect at identity/fill level.
- Equivalence test for the placebo hypothesis: v-no-out vs v-adopted must tie within noise
  on TG/PP/acc AND produce identical VRAM peak on dev0 (that is the cleanest signal).

### I.7 Tradeoffs / caveats — T5

- The `-ot` pattern list is order-sensitive as a *list of overrides*; keep patterns
  disjoint to avoid ambiguity (distinct prefixes here, so no overlap).
- `LLAMA_OUTPUT_PIN` overrides `-ot` for the head (last wins) — never set both for the same
  tensor to conflicting devices.
- mmproj on CUDA1 consumes ~1.1 GiB of dev1 — at deep T1/T2 points dev1 gets tight; T5's
  usability legs should inherit that pressure and prefer Ran deep points with the mmproj
  parked (it is parked throughout).
- Projections of `output.weight` size scale with vocab; the ≈511 MiB figure is this model's
  shape — do not transplant OOM lessons to a different vocab without re-deriving.

### I.8 Pros / cons — T5

Pros: pure config; A/B on identical geometry resolves the confound in the A1dig data; may
simplify the prod config (one hypothesis kill); defines the ladder lattice members for T6
(which placements exist at all ctx levels); no source changes to deliverable code.

Cons: GPU-hours for ~7-8 legs; the speed-bias finding may turn out "no exploitable bias at
27B" (the honest outcome, still worth having); placebo result materially weakens the
"override" mental model in companion docs and requires doc reconciliation.

---

## Part II — T6: dynamic reallocation by request context size

### II.1 Statement re-examined against the codebase

The user's ask: shift attention layers/weights (and budget) between dev0/dev1 according to
request context size — more capacity when a request approaches the ctx max of the current
config, back to speed when contexts compact/idle.

Code-reality check:

1. **No live re-layering exists or is close to existing.** Device assignment
   (`get_layer_buft_list`, src/llama-model.cpp:1806-1832) is computed at model-load time,
   the KV cache buffers are allocated at context init, and nothing in this fork (or upstream)
   reallocates them mid-process. Any "realloc" therefore equals a context rebuild — in
   server terms, a process bounce.
2. **The server already ships the two primitives a bounce needs:**
   - `GET /slots` (server-context.cpp:5803, wired in server.cpp:213) exposes per-slot state
     incl. prompt/processed token counts — the usage feed for a trigger;
   - `POST /slots/{id}?action=save|restore|erase` (server-context.cpp:5864-5870,
     README.md:1249/1269) persists and restores a *sequence-local* state file, with a
     dedicated unit-test file exercising save/restore/erase flows. Sequence-local means the
     dump scales with usage, not with the full cache (a 30k-token session ≈ 60-70 MB at
     q8_0 KV ≈ 2.18 kB/cell×30k ≈ 65 MB — seconds of NVMe IO, not minutes).
3. **The prod stack already owns the one thing that can do a bounce safely**: the
   hot-swapper outside this repo restarts the server against models.ini entries
   (verified behavior during the earlier stack restore) — T6's natural home is a sidecar
   state machine there, not a fork patch.

### II.2 Feasibility verdict

**Conditional-High** — engineering is well-supported by existing primitives; two POC gates
must pass before it is "rational to implement":

- **Gate 1 — state bridge survives the config change**: save state on config A, restore on
  config B with different `n_ctx` and, crucially, with **draft-mtp speculative decoding
  enabled**. If the draft state does not round-trip, fall back to prompt re-send (the client
  re-prices the session, KV rebuild via prefill — always available, it is literally what the
  bench harness does every leg).
- **Gate 2 — economics**: a bounce costs `load ≈ 2.5-4 min + (state-restore IO ≈ seconds |
  prompt re-send prefill ≈ usage / pp-speed)`. The trigger must only fire when the
  alternative is *worse* (request impossible → 400, or a full compaction cycle). If users'
  prompt-size distribution makes breaches rare, the feature should be disabled in prod.

### II.3 Approach: ladder state machine attached to the swapper

Lattice — the configs the machine may take — comes directly from measured data:

| level | quota | KV | usable ctx (fill-verified) | role |
|-------|-------|----|-----------------------------|------|
| L0 (fast) | 16 | q8_0 | ~118k | default (speed-optimal) |
| L1 | 12 | q8_0 | ~188k | deep-responder |
| L2 (deepest) | 11 | q8_0 | ~205k | worst-case absorber (P40 binding wall) |
| L0q (extension, after T2) | 16 | q4_0 | ~to be measured; predicted ≈ double of L0 | faster-EQ at double ctx |
| pending T2/T5 results | q4_0 combos + placement variants | | | the lattice grows only from verified legs |

Rules (single-slot prod; the machine multiplies `usage×n_slots` for future parallel use):

- **Down-shift** (capacity): when an incoming request's expected total tokens
  (`expected = prompt_tokens + n_predict_budget + safety pad ≈ +4k`) exceeds the current
  level's usable ctx by the hysteresis margin → save slot state → bounce one level deeper
  that still fits → restore (fallback: re-send).
- **Up-shift** (speed reclaim): when recorded peak usage across all slots has decayed below
  target-level usable ctx for `T_idle` consecutive samples (default: one hour), or right
  after an explicit compaction/finalization event → bounce up. One bounce at a time, with a
  cooldown to prevent flapping.
- **No-nag rule**: never bounce for a request that *might* get long; only for one that
  *did* breach. Overshoot handling: on hard breach with no deeper level available, answer
  the client with the documented error (today's failure is silent kill/refill).

### II.4 Implementation plan (outside this repo; ~1-2 weeks part-time)

- **P0 — instrument (no behavior change)**: sidecar process (same Python3 toolchain as the
  swapper) polls `GET /slots` + tails usage from responses the prod adapter already emits;
  persist a rolling per-slot "predicted depth" series. Ship it dark, observe breach
  frequency for a week. *If zero breaches observed → declare "not rational today" and stop.*
- **P1 — POC in bench harness**: script runs: bring up L0 config → long-fill request that
  breaches ~118k → sidecar triggers bounce to L1 with seq-state save → server up at L1 →
  restore → continue generation → compare to a non-bounced reference at L1 (idle + temp-0
  identity + finish_reason + timing parity). Gate 1 decided here.
- **P2 — swapper integration behind a feature flag** (`ladder=1` in the swapper config):
  the machine owns switching between verified models.ini level entries only; kill switch =
  flag off → revert to current static config; levels missing → treat as static.
- **P3 — prod, quiet rollout**: L0↔L1 only, conservative cooldowns, every bounce logged
  with full accounting (breach size, bounce latency, restore vs re-send path chosen).

No llama.cpp source changes in any phase. If Gate 1 fails *and* re-send economics prove
bad, the documented (but not endorsed) fallback would be an in-fork Voluntary-context
restore extension — out of scope until measured necessary.

### II.5 Testing plan

- P1 POC: (a) same-prompt same-seed generation parity across a bounced session vs a direct
  deeper-config run (only valid without sampling changes if MTP acceptance is stable
  across restore — else compare finish_reason + content-equivalence tooling);
  (b) breach → bounce → resume latency measured end-to-end; (c) restore-failure fault
  injection → automatic re-send fallback; (d) flapping stress (alternate long/short
  prompts) → cooldown proven by log; (e) config-restore contract: after flag-off, prod
  returns to current static config exactly.
- Acceptance: breach handling that used to be impossible now succeeds; a week-long dark
  P0 with realistic prod traffic producing zero unplanned downtime attributable to T6.

### II.6 Tradeoffs / caveats — T6

- **Bounce is not free**: load ≈ 2.5-4 min always; plus restore-IO or re-send prefill.
  Under a burst of breaching requests this serializes badly → cooldown + one-at-a-time are
  load-bearing.
- **Sequence state vs draft state**: untested with draft-mtp (Gate 1). Rough estimate says
  the seq dump is usage-proportional, but the spec-decode buffers may not all round-trip.
- **State-file authenticity**: save/restore writes server-side files (path is server cwd or
  filename param); lifecycle management (which dir, cleanup, disk space at usage-proportional
  sizes) is the sidecar's job.
- **Lattice integrity**: every level the machine can take must be a *fill-verified* config —
  inherit T1's lesson that "loads" ≠ "usable" on all four legs, cross-checked per level.
- **Reasoning UX**: a bounce is invisible at the API level except for latency; streaming
  clients mid-generation are the edge case — prod adapter currently treats generation as
  discrete requests, which fits the model.
- **Accuracy caveat**: predicted depth from `usage.prompt_tokens` per request is an
  approximation of autocumulative chat growth; the slot cache makes effective depth grow
  as sessions continue — the +4k pad is the crude cover; refine with per-slot cached
  counters from `/slots` if P0 shows the pad is insufficient.

### II.7 Pros / cons — T6

Pros: converts an impossible class of requests (prompt > static ctx) into a slow-but-served
one; reclaims speed automatically when pressure passes; builds entirely on shipped endpoints
and the existing swapper; the lattice writes down the ops knowledge accumulated in T1/T2/T5;
zero fork-diff protects the hard "no source change" invariant of this tuning wave.

Cons: real complexity in a prod-critical path (the swapper) unless kept behind a flag;
bounce latency is user-visible with no in-flight generation today; the value depends on a
traffic property (breach frequency) that must be measured first (P0) — if it is ~0, the
task correctly terminates; state-restore across spec decoding is unproven until Gate 1.

---

## Joint summary

- Dependency: T5 and T2 both feed T6's lattice (verified levels only). T5 also resolves the
  prod-config composition question (placebo clause). Order: T5 can run whenever the GPUs
  are free of the T2 session's legs; T6 starts at P0 (instrumentation, no GPU) immediately
  — the two are independent until P2.
- Zero-source-change invariant holds for both tasks as specified. The only plausible fork
  change (a future ctx-restore enhancement) is explicitly deferred and gated.
- Verdicts: T5 = HIGH feasibility, ~7-8 config-only legs, one decision (simplify prod
  config or not). T6 = conditional-High, decided by two POC gates and a week of dark P0
  telemetry; worst realistic outcome is a cheap, well-instrumented "not rational yet".

Open questions to close during execution:
1. v-no-out vs v-adopted equivalence (T5 Axis A legs) — placebo or not.
2. Logits-head speed bias at equal TG, 27B shape (9B precedent says adverse; falsify).
3. Seq-state restore with draft-mtp (T6 Gate 1).
4. Breach frequency in prod traffic (T6 P0) — the stop/go number.
