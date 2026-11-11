# Day-one numbers for a disk-backed KV stash across context-slot reloads (one real workday, one hybrid model)

*Same plain language, but written for people who work on the server. No fluff, numbers follow. Read time ~6 minutes.*

## The context

One shared GPU server, one model, one slot (`-np 1`). The model is hybrid: 16
full-attention layers plus 48 recurrent GDN layers. KV cache is quantized. The
server hosts several context profiles (105k, 118k, 198k tokens) and agent work
cycles through them all day. When the active profile changes, the server tears
down the slot and loads a new one.

## What we treated as the problem

Every slot teardown killed the KV cache for the conversation that was living in
it. The next request on that profile did what every llama-server does honestly:
the longest common prefix against the reused slot came out tiny, and the server
re-processed the whole history. At this rig's measured prefill speed
(161-235 tokens/s), that was 4 to 9 minutes per return on a 40-70k-token
conversation.

Two server-side memory layers tried to soften it and each had a ceiling:
- The in-RAM prompt cache held full states, but five conversations cycling all
  day outgrew its 8 GiB budget. Its FIFO eviction then dropped the fat entry
  (the big conversation) exactly when that conversation was about to come back.
- The context checkpoint ring (in-slot recurrent-state snapshots) can rewind a
  *returning* conversation without a full re-process - when it covers the fork
  point. When the conversation had moved past the newest checkpoint, the server
  correctly refused to fake the recurrent state and printed the familiar
  message: forcing full prompt re-processing because the cache data for the
  hybrid/recurrent layers does not reach that far back (the same behavior the
  ongoing SWA/hybrid prefix-reuse discussion is about). That cost one full
  re-process each time it happened.

So the gap to close was across teardowns: the conversation state had nowhere to
live that survived a slot kill.

## The change we tested

A server-side "stash" around slot teardown, in three lines:

1. On slot teardown (profile/model swap), serialize the whole live state with
   the standard sequence-state save path: main KV cache, the checkpoint-ring
   appendix (149.6 MiB per captured checkpoint), and the speculative-decode
   draft state. The file goes to an NVMe directory, keyed as
   `<geometry-digest>_<conversation-prefix-hash>_<n_tokens>.kvstate`.
2. On slot load, match by geometry digest + conversation prefix hash. A match
   replants the state from disk and the slot serves only a small delta prefill
   (the suffix). A geometry mismatch refuses loudly and stays cold - we did not
   want state bent across incompatible cache formats.
3. Read-back is the only success criterion. No "written 2.4 GB" toast counts;
   every restore is judged by what the slot actually served.

## Day-one results (45,050 server log lines, 8 slot loads)

Real work, not a bench: several agents passed tasks back and forth, profiles
changed 7 times, the same conversations kept returning.

| Event | Count | Size (MB) | Time |
|---|---|---|---|
| Save on teardown | 7 | 1,929-5,349 | 2.0-6.7 s each |
| Restore on return | 4 | 2,463-4,668 | 2.1-6.2 s each |
| Served delta prefill after restore | 3 normal + 1 anomaly | - | 372 / 2,710 / 2,702 tokens |
| Full re-processes avoided | 4 | - | ~4-9 min each at live speeds |

Digests did their job: of the 8 slot loads, 4 matched and restored, and the
other 4 were correct refusals (different geometry) or genuine cold starts
(new conversations). Zero wrong-geometry restores, zero lost conversations.

Numbers that matter most:
- Total re-processing avoided: ~230k tokens ≈ 16-24 wall minutes.
- Total stash postage: ~45 s (saves + reads).

That is a 20-40× net gain on the operation it fixes.

## One honest anomaly

One restore loaded a 65,219-token state for an incoming prompt of 7,994 tokens
(the conversation had been compacted or reshaped between turns). The slot took
the full state, dropped invalidated checkpoint entries, and kept serving.
Nothing broke. But the matching rule currently allows restoring *more* than the
request needs. Either tighten it to require `incoming >= state`, or document
superset-restore as intended slot semantics for compressed client history.

## Also honest: the gap that remains

Disk persistence covers teardown. It does not cover what we call the
"ring-gap" rewind - the hybrid/recurrent case above, 21 forced full
re-processes in the same epoch against 12 successful checkpoint rewinds. The
fix on our list: when a rewind lands past the last checkpoint, rebuild only the
recurrent layers over the missing span from the nearest checkpoint, instead of
dropping everything the attention layers still remember. That one is a much
bigger speedup per line of code than the stash, because it never touches disk.

## Limits, plainly

- One rig, one model family, one day. Your numbers will move.
- The largest saved state (127,380 tokens) was written but never read back in
  this run; restore cost at that size is extrapolated, not measured.
- All prefill rates are this rig's own; compare rates, not absolute seconds.

## If it is useful

The approach is plain: standard state save/load API, keyed by digest +
prefix-hash, plus band-aids we already had (checkpoint ring, draft state). Most
of the value comes from two decisions: write *before* the state dies, and let
exact read-back, not intents, define success. Happy to share the matching-rule
and teardown details if anyone wants to try the same shape on their server.
