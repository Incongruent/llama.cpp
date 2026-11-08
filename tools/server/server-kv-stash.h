#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// KV-cache stash (fork A39): on-disk persistence of slot server states so a
// conversation survives llama-router model swaps, sleeping states and process
// restarts. Staged at sleep/exit and restored into fresh slots when an
// incoming prompt extends a saved token prefix. server-context.cpp performs
// the llama state serialization; this module owns tags, file naming, matching
// and directory hygiene.
//
// file naming: kvst_<digest>_<conv_hash>_<n_tokens>.kvstate (+ .json sidecar)
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

std::string hex64(uint64_t v);

std::string geometry_digest(const std::vector<std::string> & parts);

std::string state_base_name(const state_tag & t);

bool write_sidecar(const std::string & state_path, const state_tag & t, const std::string & model_name);

bool read_sidecar(const std::string & sidecar_path, state_tag & t, std::string & model_name);

// incremental FNV-1a over token ids, so prefixes can be hashed cheaply
struct prefix_hasher {
    prefix_hasher(const llama_token * data, size_t n_tokens) : data(data), n_tokens(n_tokens) {}

    uint64_t prefix_hash(size_t n) const;

private:
    const llama_token * data;
    size_t              n_tokens;
};

struct candidate {
    std::string state_path;
    state_tag   tag;
};

// pick the best saved state for an incoming token stream (best = largest
// restored prefix, ranked by n_tokens across candidates with a matching
// digest). candidates whose saved stream fits inside the incoming tokens are
// exact-hash-verified before selection; longer candidates (a saved state that
// also covers a model generation tail) depend on the server trimming the
// covered prefix afterwards, which is always safe.
bool find_best_state(const std::string & dir, const std::string & digest,
                     const prefix_hasher & hasher, size_t n_incoming, candidate & out);

// delete the oldest states (by mtime) until the stash fits limit_bytes;
// a negative limit disables pruning; keep_path is never deleted
void prune_dir(const std::string & dir, int64_t limit_bytes, const std::string & keep_path);

// create the directory (and parents) if missing; returns false on error
bool ensure_dir(const std::string & dir);

} // namespace kstash
