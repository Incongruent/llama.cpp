// fork A39/A59: offline behavior tests for the KV-stash matcher
// (tools/server/server-kv-stash.cpp) - no model, no GPU, temp-dir fixtures.
// The A59 phase-1 assertions cover the enriched refusal note: first-mismatch
// token index of the longest prefix-checked / hash-refused candidate plus the
// salvage verdict (token-0 = unsalvageable by any disk state, interior =
// salvageable by checkpoint restore in phase 2). The matching rules themselves
// (accept paths, refusals, digest filter) are regression-guarded here too.

#include "server-kv-stash.h"

#include <cstdio>
#include <filesystem>
#include <string>
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

static void test_refuse_superset_interior(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4, 5, 6, 7, 8});
    consult_result r = consult(dir, {1, 2, 3, 9, 4});
    CHECK(!r.found);
    CHECK_CONTAINS(r.note, "1 longer state(s) refused");
    CHECK_CONTAINS(r.note, "first mismatch at token 3");
    CHECK_CONTAINS(r.note, "interior divergence at token 3");
    CHECK_CONTAINS(r.note, "salvageable by checkpoint restore");
    CHECK_CONTAINS(r.note, "kvst_"); // candidate name is carried in the note
    CHECK_NOT_CONTAINS(r.note, "unsalvageable");
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

static void test_refuse_hashfit_interior(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4});
    consult_result r = consult(dir, {1, 2, 9, 4, 5, 6});
    CHECK(!r.found);
    CHECK_CONTAINS(r.note, "1 shorter state(s) hash-refused");
    CHECK_CONTAINS(r.note, "first mismatch at token 2");
    CHECK_CONTAINS(r.note, "interior divergence at token 2");
    CHECK_CONTAINS(r.note, "salvageable by checkpoint restore");
}

static void test_refuse_hashfit_no_companion(const fs::path & dir) {
    make_state(dir, DIGEST, {1, 2, 3, 4}, /*with_tokens=*/false);
    consult_result r = consult(dir, {1, 2, 9, 4, 5, 6});
    CHECK(!r.found);
    CHECK_CONTAINS(r.note, "1 shorter state(s) hash-refused");
    CHECK_CONTAINS(r.note, "no token companion to scan");
    CHECK_NOT_CONTAINS(r.note, "first mismatch");
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
    run("accept_superset",      test_accept_superset_prefix);
    run("accept_shorter",       test_accept_shorter_fit);
    run("refuse_sup_int",       test_refuse_superset_interior);
    run("refuse_sup_t0",        test_refuse_superset_token0);
    run("refuse_sup_notok",     test_refuse_superset_no_companion);
    run("refuse_sup_badhash",   test_refuse_superset_companion_hash_mismatch);
    run("refuse_hash_int",      test_refuse_hashfit_interior);
    run("refuse_hash_notok",    test_refuse_hashfit_no_companion);
    run("continue_after",       test_continue_after_refusal);
    run("digest_filter",        test_digest_filter);
    run("empty_missing",        test_empty_and_missing_dir);
    run("note_null",            test_note_nullptr);

    fs::remove_all(root, ec);

    std::printf("test-kv-stash: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
