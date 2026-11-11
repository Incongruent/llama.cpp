#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// KV-cache stash (fork A39): on-disk persistence of slot server states so a
// conversation survives llama-router model swaps, sleeping states and process
// restarts. Staged at sleep/exit and restored into fresh slots when an
// incoming prompt extends (or is a verified prefix of) a saved token stream.
// server-context.cpp performs the llama state serialization; this module owns
// tags, file naming, matching and directory hygiene.
//
// file naming: kvst_<digest>_<conv_hash>_<n_tokens>.kvstate (+ .json sidecar,
// + .tokens companion holding the raw text-token stream)
// sidecar json: {"digest", "conv_hash", "n_tokens", "model_name", "saved_at"}
// Any mismatch (tokenizer, KV geometry, context size...) makes the restore
// attempt fail cleanly and fall back to a full re-prefill.

namespace kstash {

struct state_tag {
    std::string digest;    // model + KV geometry identity, hex
    std::string conv_hash; // FNV-1a of the token stream, hex
    int64_t     n_tokens   = 0;
};

uint64_t hash_tokens(const llama_token * data, size_t n_tokens);

// fork A39.1: true when the raw token stream carries media placeholder
// (LLAMA_TOKEN_NULL) entries, i.e. the conversation includes image/audio
// chunks. The stash conversation key is the TEXT-token stream (placeholders
// filtered), so two conversations with identical text but different media
// would be indistinguishable and could restore each other's KV (image
// positions carrying the other conversation's media content). Such streams
// are hard-refused by the stash on both the dump and the consult side -
// multimodal use is unsupported by the stash by design.
bool stream_has_media(const llama_token * data, size_t n_tokens);

std::string hex64(uint64_t v);

std::string geometry_digest(const std::vector<std::string> & parts);

std::string state_base_name(const state_tag & t);

bool write_sidecar(const std::string & state_path, const state_tag & t, const std::string & model_name);

bool read_sidecar(const std::string & sidecar_path, state_tag & t, std::string & model_name);

// raw llama_token stream companion (<state>.kvstate.tokens); written before
// the sidecar so a sidecar implies both files exist
bool write_token_file(const std::string & path, const llama_token * data, size_t n_tokens);

// read + size-check a companion token file (expect_n = sidecar n_tokens)
bool read_token_file(const std::string & path, std::vector<llama_token> & out, size_t expect_n);

// first token index at which the two streams differ over the first n tokens
// (returns n when identical); single O(n) scan, used on the refusal path only
size_t first_mismatch_index(const llama_token * a, const llama_token * b, size_t n);

// incremental FNV-1a over token ids, so prefixes can be hashed cheaply
struct prefix_hasher {
    prefix_hasher(const llama_token * data, size_t n_tokens) : data_(data), n_tokens_(n_tokens) {}

    uint64_t prefix_hash(size_t n) const;

    const llama_token * data() const { return data_; }
    size_t size() const { return n_tokens_; }

private:
    const llama_token * data_;
    size_t              n_tokens_;
};

struct candidate {
    std::string state_path;
    state_tag   tag;
};

// pick the best saved state for an incoming token stream (best = largest
// verified restored prefix, ranked by n_tokens across candidates with a
// matching digest). candidates whose saved stream fits inside the incoming
// tokens are verified by hashing the first n saved incoming tokens; longer
// candidates (a saved state that also covers a model generation tail) are
// accepted only when the incoming stream is an exact token prefix of the
// saved stream, verified against the .tokens companion. candidates without
// a companion token file are never restored when longer than the incoming
// stream. when nothing verifies, note (if given) describes why, including
// (fork A59 phase 1) the first-mismatch token index of the longest
// prefix-checked or hash-refused candidate plus a salvage verdict: index 0
// marks token-0 divergence (unsalvageable by any disk state - the fix is
// client-side token-stream stability), index > 0 marks interior divergence
// (salvageable by checkpoint restore, A59 phase 2).
bool find_best_state(const std::string & dir, const std::string & digest,
                     const prefix_hasher & hasher, size_t n_incoming, candidate & out,
                     std::string * note = nullptr);

// delete the oldest states (by mtime) until the stash fits limit_bytes;
// a negative limit disables pruning; keep_path is never deleted
void prune_dir(const std::string & dir, int64_t limit_bytes, const std::string & keep_path);

// create the directory (and parents) if missing; returns false on error
bool ensure_dir(const std::string & dir);

} // namespace kstash
