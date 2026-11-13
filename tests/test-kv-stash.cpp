// fork A39/A59: offline behavior tests for the KV-stash matcher
// (tools/server/server-kv-stash.cpp) - no model, no GPU, temp-dir fixtures.
// The A59 phase-1 assertions cover the enriched refusal note: first-mismatch
// token index of the longest prefix-checked / hash-refused candidate plus the
// salvage verdict (token-0 = unsalvageable by any disk state, interior =
// salvageable). The matching rules themselves (accept paths, refusals, digest
// filter) are regression-guarded here too.
//
// fork A59 phase 2: interior divergence is no longer a pure refusal. when no
// candidate fully verifies, the matcher now returns the candidate carrying the
// longest verified shared prefix, flagged salvage=true with
// salvage_prefix = that length (token-0 divergence stays unsalvageable,
// salvage_prefix 0). these tests pin that contract: salvage is returned for
// interior divergence, max-verified-prefix wins across candidates, and a full
// (superset-prefix) match still beats any salvage.

#include "server-kv-stash.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static int g_checks   = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    g_checks++; \
    if (!(cond)) { \
        g_failures++; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

// CHECK_CONTAINS(haystack, "needle")
#define CHECK_CONTAINS(hay, needle) do { \
    g_checks++; \
    if ((hay).find(needle) == std::string::npos) { \
        g_failures++; \
        std::fprintf(stderr, "FAIL %s:%d: note missing '%s'\nnote: %s\n", \
                __FILE__, __LINE__, (needle), (hay).c_str()); \
    } \
} while (0)

#define CHECK_NOT_CONTAINS(hay, needle) do { \
    g_checks++; \
    if ((hay).find(needle) != std::string::npos) { \
        g_failures++; \
        std::fprintf(stderr, "FAIL %s:%d: note unexpectedly contains '%s'\nnote: %s\n", \
                __FILE__, __LINE__, (needle), (hay).c_str()); \
    } \
} while (0)

static const std::string DIGEST = "33fd07879e7d";

// write one saved state (token companion + sidecar; sidecar last, so a
// sidecar implies both files exist - mirrors the production dump order)
static void make_state(const fs::path & dir, const std::string & digest,
                       const std::vector<llama_token> & toks, bool with_tokens = true) {
    kstash::state_tag tag;
    tag.digest    = digest;
    tag.conv_hash = kstash::hex64(kstash::hash_tokens(toks.data(), toks.size()));
    tag.n_tokens  = (int64_t) toks.size();
    const std::string base = (dir / kstash::state_base_name(tag)).string();
    if (with_tokens) {
        CHECK(kstash::write_token_file(base + ".tokens", toks.data(), toks.size()));
    }
    CHECK(kstash::write_sidecar(base, tag, "test-model"));
}

struct consult_result {
    bool                  found  = false;
    kstash::candidate     cand;
    std::string           note;
};

static consult_result consult(const fs::path & dir, const std::vector<llama_token> & incoming, std::string * note_out = nullptr) {
    consult_result r;
    kstash::prefix_hasher hasher(incoming.data(), incoming.size());
    r.found = kstash::find_best_state(dir.string(), DIGEST, hasher, incoming.size(), r.cand, &r.note);
    if (note_out != nullptr) {
        *note_out = r.note;
    }
    return r;
}

static void test_first_mismatch_index() {
    const std::vector<llama_token> a      = {1, 2, 3, 4, 5};
    const std::vector<llama_token> b_same = {1, 2, 3, 4, 5};
    const std::vector<llama_token> b_zero = {0, 2, 3, 4, 5};
    const std::vector<llama_token> b_mid  = {1, 2, 9, 4, 5};
    CHECK(kstash::first_mismatch_index(a.data(), b_same.data(), 5) == 5); // identical -> n
    CHECK(kstash::first_mismatch_index(a.data(), b_zero.data(), 5) == 0);
    CHECK(kstash::first_mismatch_index(a.data(), b_mid.data(), 5) == 2);
    CHECK(kstash::first_mismatch_index(a.data(), b_mid.data(), 2) == 2);  // divergence inside the first n
    CHECK(kstash::first_mismatch_index(a.data(), b_same.data(), 0) == 0); // empty scan -> 0
}

// fork A39.1: media detection drives the stash hard-refusal on both sides.
// The collision class (same text, different image) is closed because BOTH
// streams carry media placeholders -> BOTH are excluded (dump + consult),
// so neither can ever restore the other's KV.
static void test_stream_has_media() {
    const std::vector<llama_token> text_only  = {101, 102, 103};
    const std::vector<llama_token> empty      = {};
    const std::vector<llama_token> img_one    = {101, 102, LLAMA_TOKEN_NULL, 103};                   // 1 placeholder
    const std::vector<llama_token> img_two    = {101, 102, LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL, 103}; // 2 placeholders
    const std::vector<llama_token> img_twoimg = {101, LLAMA_TOKEN_NULL, 102, LLAMA_TOKEN_NULL, 103}; // 2 images

    CHECK(!kstash::stream_has_media(text_only.data(), text_only.size())); // text-only: stashing proceeds as before
    CHECK(!kstash::stream_has_media(empty.data(), empty.size()));         // empty: false (guard order: empty check first)
    CHECK(kstash::stream_has_media(img_one.data(), img_one.size()));
    CHECK(kstash::stream_has_media(img_two.data(), img_two.size()));
    CHECK(kstash::stream_has_media(img_twoimg.data(), img_twoimg.size()));

    // same text, different images: text-token streams are identical...
    {
        std::vector<llama_token> t_a, t_b;
        for (llama_token t : img_one)    if (t != LLAMA_TOKEN_NULL) t_a.push_back(t);
        for (llama_token t : img_twoimg) if (t != LLAMA_TOKEN_NULL) t_b.push_back(t);
        CHECK(t_a == t_b);
        CHECK(kstash::hash_tokens(t_a.data(), t_a.size()) == kstash::hash_tokens(t_b.data(), t_b.size())); // ...conv_hash collides
    }
    // ...but both raw streams carry media -> both hard-refused -> no restore possible
    CHECK(kstash::stream_has_media(img_one.data(), img_one.size()));
    CHECK(kstash::stream_has_media(img_twoimg.data(), img_twoimg.size()));
}

static void test_accept_superset_prefix(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4, 5, 6, 7, 8});
    consult_result r = consult(dir, {1, 2, 3, 4});
    CHECK(r.found);
    CHECK(r.cand.tag.n_tokens == 8);
    CHECK(r.note.empty());
}

static void test_accept_shorter_fit(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4});
    consult_result r = consult(dir, {1, 2, 3, 4, 5, 6});
    CHECK(r.found);
    CHECK(r.cand.tag.n_tokens == 4);
}

// fork A59 phase 2: interior divergence (saved longer, verified prefix then
// diverge) is salvaged, not refused - the longest verified prefix [0, F) is
// returned flagged salvage so the caller restores it and re-prefills the tail.
static void test_salvage_superset_interior(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4, 5, 6, 7, 8});
    consult_result r = consult(dir, {1, 2, 3, 9, 4});
    CHECK(r.found);
    CHECK(r.cand.salvage);
    CHECK(r.cand.salvage_prefix == 3);   // verified prefix {1, 2, 3}, diverge at 3
    CHECK(r.cand.tag.n_tokens == 8);
    CHECK_CONTAINS(r.note, "salvaging verified 3-token prefix");
    CHECK_CONTAINS(r.note, "re-prefill from token 3");
}

static void test_refuse_superset_token0(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4, 5, 6, 7, 8});
    consult_result r = consult(dir, {0, 2, 3, 4});
    CHECK(!r.found);
    CHECK_CONTAINS(r.note, "first mismatch at token 0");
    CHECK_CONTAINS(r.note, "token-0 divergence");
    CHECK_CONTAINS(r.note, "unsalvageable");
}

static void test_refuse_superset_no_companion(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4, 5, 6, 7, 8}, /*with_tokens=*/false);
    consult_result r = consult(dir, {1, 2, 3, 4});
    CHECK(!r.found);
    CHECK_CONTAINS(r.note, "1 longer state(s) refused");
    CHECK_CONTAINS(r.note, "no prefix check reached");
    CHECK_NOT_CONTAINS(r.note, "first mismatch");
}

static void test_refuse_superset_companion_hash_mismatch(const fs::path & dir) {
    // token file and sidecar hash disagree (torn dump): refuse, no index
    const std::vector<llama_token> toks = {1, 2, 3, 4, 5, 6, 7, 8};
    kstash::state_tag tag;
    tag.digest    = DIGEST;
    tag.conv_hash = kstash::hex64(kstash::hash_tokens(toks.data(), toks.size()) ^ 0xdeadbeef);
    tag.n_tokens  = (int64_t) toks.size();
    const std::string base = (dir / kstash::state_base_name(tag)).string();
    CHECK(kstash::write_token_file(base + ".tokens", toks.data(), toks.size()));
    CHECK(kstash::write_sidecar(base, tag, "test-model"));

    consult_result r = consult(dir, {1, 2, 3, 4});
    CHECK(!r.found);
    CHECK_CONTAINS(r.note, "no prefix check reached");
    CHECK_NOT_CONTAINS(r.note, "first mismatch");
}

// fork A59 phase 2: interior divergence on the hash-fit path (saved shorter,
// prefix hash refused) is likewise salvaged with the verified prefix [0, F).
static void test_salvage_hashfit_interior(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4});
    consult_result r = consult(dir, {1, 2, 9, 4, 5, 6});
    CHECK(r.found);
    CHECK(r.cand.salvage);
    CHECK(r.cand.salvage_prefix == 2);   // verified prefix {1, 2}, diverge at 2
    CHECK(r.cand.tag.n_tokens == 4);
    CHECK_CONTAINS(r.note, "salvaging verified 2-token prefix");
    CHECK_CONTAINS(r.note, "re-prefill from token 2");
}

static void test_refuse_hashfit_no_companion(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4}, /*with_tokens=*/false);
    consult_result r = consult(dir, {1, 2, 9, 4, 5, 6});
    CHECK(!r.found);
    CHECK_CONTAINS(r.note, "1 shorter state(s) hash-refused");
    CHECK_CONTAINS(r.note, "no token companion to scan");
    CHECK_NOT_CONTAINS(r.note, "first mismatch");
}

// fork A59 phase 2: the salvage is the candidate with the LONGEST verified
// shared prefix, not the longest state. here the 8-token state diverges at
// token 0 (F=0) while the 5-token state diverges at token 2 (F=2) -> the
// 5-token state wins.
static void test_salvage_max_prefix_wins(const fs::path & dir) {
    make_state(dir, DIGEST, {9, 2, 3, 4, 5, 6, 7, 8}); // 8 tok, token-0 divergence (F=0)
    make_state(dir, DIGEST, {1, 2, 9, 4, 5});          // 5 tok, interior at 2 (F=2)
    consult_result r = consult(dir, {1, 2, 3, 4, 5, 6, 7});
    CHECK(r.found);
    CHECK(r.cand.salvage);
    CHECK(r.cand.salvage_prefix == 2);
    CHECK(r.cand.tag.n_tokens == 5); // the 5-token state, not the longer 8-token one
}

// fork A59 phase 2: a full (superset-prefix) match always wins over a salvage,
// even when a longer state carries a salvageable interior divergence.
static void test_full_match_beats_salvage(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 9, 4, 5, 6, 7, 8}); // 8 tok, interior at 2 (F=2)
    make_state(dir, DIGEST, {1, 2, 3, 4, 5});          // 5 tok, exact superset-prefix fit
    consult_result r = consult(dir, {1, 2, 3, 4, 5, 6, 7});
    CHECK(r.found);
    CHECK(!r.cand.salvage);
    CHECK(r.cand.tag.n_tokens == 5);
    CHECK(r.note.empty());
}

// the longest candidate refuses, the loop must continue to the next one
static void test_continue_after_refusal(const fs::path & dir) {
    make_state(dir, DIGEST, {9, 1, 2, 3, 4, 5, 6, 7}); // 8 tok, token-0 divergence
    make_state(dir, DIGEST, {1, 2, 3, 4});             // 4 tok, exact prefix fit
    consult_result r = consult(dir, {1, 2, 3, 4, 5, 6, 7});
    CHECK(r.found);
    CHECK(r.cand.tag.n_tokens == 4);
}

static void test_digest_filter(const fs::path & dir) {
    make_state(dir, "deadbeefdead", {1, 2, 3, 4, 5, 6, 7, 8});
    consult_result r = consult(dir, {1, 2, 3});
    CHECK(!r.found);
    CHECK(r.note.empty()); // no same-digest candidates -> caller says "new conversation"
}

static void test_empty_and_missing_dir(const fs::path & dir) {
    fs::create_directories(dir);
    consult_result r0 = consult(dir, {1, 2, 3});
    CHECK(!r0.found);
    CHECK(r0.note.empty());

    consult_result r1 = consult(dir / "does_not_exist", {1, 2, 3});
    CHECK(!r1.found);
    CHECK(r1.note.empty());
}

static void test_note_nullptr(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4, 5, 6, 7, 8});
    const std::vector<llama_token> incoming = {0, 2, 3, 4};
    kstash::prefix_hasher hasher(incoming.data(), incoming.size());
    kstash::candidate cand;
    CHECK(!kstash::find_best_state(dir.string(), DIGEST, hasher, 4, cand, /*note=*/nullptr));
}

// fork A60: prune_dir regression tests. the name filter previously compared
// the last 7 chars ("kvstate") against the 8-char ".kvstate" suffix and never
// matched, so prune was a silent no-op (nothing ever deleted, live dir grew
// unbounded). these tests pin: byte-cap eviction oldest-first, companion
// (.json/.tokens) removal with their state, keep_path protection, and
// companion-only files (orphans) being neither counted nor removed.

static bool file_exists(const fs::path & p) {
    std::error_code ec;
    return fs::exists(p, ec) && !ec;
}

static void write_raw(const fs::path & path, size_t nbytes) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    CHECK((bool) ofs);
    const std::string blob(nbytes, 'x');
    ofs.write(blob.data(), (std::streamsize) blob.size());
    CHECK((bool) ofs);
    // ensure distinct mtimes: prune evicts oldest-first by mtime
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

// one fake state = .kvstate blob of `nbytes` + companions at the production
// paths (<state>.kvstate.json / <state>.kvstate.tokens)
static void make_raw_state(const fs::path & dir, const char * stem, size_t nbytes) {
    write_raw((dir / (std::string(stem) + ".kvstate")).string(), nbytes);
    write_raw((dir / (std::string(stem) + ".kvstate.json")).string(), 8);
    write_raw((dir / (std::string(stem) + ".kvstate.tokens")).string(), 8);
}

static void test_prune_oldest_to_cap(const fs::path & dir) {
    make_raw_state(dir, "kvst_aaaaaa_h1_10", 10);
    make_raw_state(dir, "kvst_aaaaaa_h2_20", 20);
    make_raw_state(dir, "kvst_aaaaaa_h3_30", 30);

    // under the cap: nothing is deleted
    kstash::prune_dir(dir.string(), 100, "");
    CHECK(file_exists(dir / "kvst_aaaaaa_h1_10.kvstate"));
    CHECK(file_exists(dir / "kvst_aaaaaa_h3_30.kvstate"));

    // total 60, cap 31: the two oldest states go (with companions), newest stays
    kstash::prune_dir(dir.string(), 31, "");
    CHECK(!file_exists(dir / "kvst_aaaaaa_h1_10.kvstate"));
    CHECK(!file_exists(dir / "kvst_aaaaaa_h1_10.kvstate.json"));
    CHECK(!file_exists(dir / "kvst_aaaaaa_h1_10.kvstate.tokens"));
    CHECK(!file_exists(dir / "kvst_aaaaaa_h2_20.kvstate"));
    CHECK(!file_exists(dir / "kvst_aaaaaa_h2_20.kvstate.json"));
    CHECK(!file_exists(dir / "kvst_aaaaaa_h2_20.kvstate.tokens"));
    CHECK(file_exists(dir / "kvst_aaaaaa_h3_30.kvstate"));
    CHECK(file_exists(dir / "kvst_aaaaaa_h3_30.kvstate.json"));
    CHECK(file_exists(dir / "kvst_aaaaaa_h3_30.kvstate.tokens"));
}

static void test_prune_keep_path(const fs::path & dir) {
    make_raw_state(dir, "kvst_bbbbbb_h4_40", 40);
    make_raw_state(dir, "kvst_bbbbbb_h5_50", 50);
    // total 90, cap 60, keep the OLDEST (40): the newer 50 must go instead
    kstash::prune_dir(dir.string(), 60, (dir / "kvst_bbbbbb_h4_40.kvstate").string());
    CHECK(file_exists(dir / "kvst_bbbbbb_h4_40.kvstate"));
    CHECK(file_exists(dir / "kvst_bbbbbb_h4_40.kvstate.json"));
    CHECK(file_exists(dir / "kvst_bbbbbb_h4_40.kvstate.tokens"));
    CHECK(!file_exists(dir / "kvst_bbbbbb_h5_50.kvstate"));
    CHECK(!file_exists(dir / "kvst_bbbbbb_h5_50.kvstate.json"));
    CHECK(!file_exists(dir / "kvst_bbbbbb_h5_50.kvstate.tokens"));
}

static void test_prune_companion_files(const fs::path & dir) {
    // companion-named files WITHOUT their .kvstate (orphans) are neither
    // counted toward the cap nor removed by prune_dir
    write_raw(dir / "kvst_cccccc_h6_5.kvstate.json", 8);
    write_raw(dir / "kvst_cccccc_h6_5.kvstate.tokens", 8);
    make_raw_state(dir, "kvst_cccccc_h7_5", 5);

    // cap 1: the 5-byte state cannot fit, it goes; orphans survive
    kstash::prune_dir(dir.string(), 1, "");
    CHECK(!file_exists(dir / "kvst_cccccc_h7_5.kvstate"));
    CHECK(!file_exists(dir / "kvst_cccccc_h7_5.kvstate.json"));
    CHECK(!file_exists(dir / "kvst_cccccc_h7_5.kvstate.tokens"));
    CHECK(file_exists(dir / "kvst_cccccc_h6_5.kvstate.json"));
    CHECK(file_exists(dir / "kvst_cccccc_h6_5.kvstate.tokens"));
}

// fork A40 phase-1: same-preset alternation fork-dump trigger
static void test_check_fork_dump() {
    std::vector<llama_token> tail(4096);
    for (size_t i = 0; i < tail.size(); ++i) {
        tail[i] = (llama_token) (i % 1000);
    }

    // token-0 divergence (different conversation): dump
    {
        std::vector<llama_token> in(100, (llama_token) 5000);
        kstash::fork_check fc = kstash::check_fork_dump(tail.data(), tail.size(), in.data(), in.size());
        CHECK(fc.tail == 4096);
        CHECK(fc.fork == 0);
        CHECK(fc.should_dump);
    }

    // interior early fork: dump
    {
        std::vector<llama_token> in(tail.begin(), tail.begin() + 500);
        in.push_back((llama_token) 5000);
        kstash::fork_check fc = kstash::check_fork_dump(tail.data(), tail.size(), in.data(), in.size());
        CHECK(fc.fork == 500);
        CHECK(fc.should_dump); // 500 < 0.9 * 4096
    }

    // exact tail growth (incoming == tail): the same conversation growing, no dump
    {
        kstash::fork_check fc = kstash::check_fork_dump(tail.data(), tail.size(), tail.data(), tail.size());
        CHECK(fc.fork == 4096);
        CHECK(fc.fork == fc.tail);
        CHECK(!fc.should_dump);
    }

    // deep continuation (fork >= 90% of tail): no dump; exact 90% is not "below"
    // (tail 4090 so 0.9 * tail is an exact integer boundary: 3681)
    {
        std::vector<llama_token> in(tail.begin(), tail.begin() + 3681);
        in.push_back((llama_token) 5000);
        kstash::fork_check fc = kstash::check_fork_dump(tail.data(), 4090, in.data(), in.size());
        CHECK(fc.fork == 3681);
        CHECK(fc.fork * 10 == fc.tail * 9); // exactly 90% of tail
        CHECK(!fc.should_dump);
    }
    {
        std::vector<llama_token> in(tail.begin(), tail.begin() + 3680); // one below the gate
        in.push_back((llama_token) 5000);
        kstash::fork_check fc = kstash::check_fork_dump(tail.data(), 4090, in.data(), in.size());
        CHECK(fc.fork == 3680);
        CHECK(fc.should_dump);
    }

    // incoming shorter than tail, still a deep continuation: no dump
    {
        std::vector<llama_token> in(tail.begin(), tail.begin() + 4000);
        kstash::fork_check fc = kstash::check_fork_dump(tail.data(), tail.size(), in.data(), in.size());
        CHECK(fc.fork == 4000);
        CHECK(!fc.should_dump);
    }

    // short resident tail (below the 2048-token gate): never dump
    {
        std::vector<llama_token> in(8, (llama_token) 5000);
        kstash::fork_check fc = kstash::check_fork_dump(tail.data(), 100, in.data(), in.size());
        CHECK(fc.tail == 100);
        CHECK(fc.fork == 0);
        CHECK(!fc.should_dump);
    }

    // degenerate inputs: null/empty sides never dump
    {
        kstash::fork_check fc = kstash::check_fork_dump(nullptr, 0, nullptr, 0);
        CHECK(!fc.should_dump);
        fc = kstash::check_fork_dump(tail.data(), 0, tail.data(), tail.size());
        CHECK(!fc.should_dump);
        fc = kstash::check_fork_dump(tail.data(), tail.size(), tail.data(), 0);
        CHECK(!fc.should_dump);
    }
}

 int main() {
    const fs::path root = fs::temp_directory_path() / "kvst_test_a59";
    std::error_code ec;
    fs::remove_all(root, ec);
    CHECK(fs::create_directories(root));

    auto run = [&](const char * name, void (* fn)(const fs::path &)) {
        const fs::path dir = root / name;
        fs::create_directories(dir);
        fn(dir);
    };

    test_first_mismatch_index();
    test_stream_has_media();
    test_check_fork_dump();
    run("accept_superset",      test_accept_superset_prefix);
    run("accept_shorter",       test_accept_shorter_fit);
    run("salvage_sup_int",      test_salvage_superset_interior);
    run("refuse_sup_t0",        test_refuse_superset_token0);
    run("refuse_sup_notok",     test_refuse_superset_no_companion);
    run("refuse_sup_badhash",   test_refuse_superset_companion_hash_mismatch);
    run("salvage_hash_int",     test_salvage_hashfit_interior);
    run("refuse_hash_notok",    test_refuse_hashfit_no_companion);
    run("salvage_max_prefix",   test_salvage_max_prefix_wins);
    run("fullmatch_beats_salv", test_full_match_beats_salvage);
    run("continue_after",       test_continue_after_refusal);
    run("digest_filter",        test_digest_filter);
    run("empty_missing",        test_empty_and_missing_dir);
    run("note_null",            test_note_nullptr);
    run("prune_oldest",         test_prune_oldest_to_cap);
    run("prune_keep",           test_prune_keep_path);
    run("prune_orphans",        test_prune_companion_files);

    fs::remove_all(root, ec);

    std::printf("test-kv-stash: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
