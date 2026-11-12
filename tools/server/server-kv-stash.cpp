#include "server-kv-stash.h"

#include "common.h"
#include "json.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace kstash {

namespace fs = std::filesystem;

static constexpr uint64_t fnva_init  = 14695981039346656037ull;
static constexpr uint64_t fnva_prime = 1099511628211ull;

uint64_t hash_tokens(const llama_token * data, size_t n_tokens) {
    uint64_t h = fnva_init;
    const size_t nbytes = n_tokens * sizeof(llama_token);
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(data);
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= bytes[i];
        h *= fnva_prime;
    }
    return h;
}

uint64_t prefix_hasher::prefix_hash(size_t n) const {
    return hash_tokens(data_, n < n_tokens_ ? n : n_tokens_);
}

bool stream_has_media(const llama_token * data, size_t n_tokens) {
    for (size_t i = 0; i < n_tokens; ++i) {
        if (data[i] == LLAMA_TOKEN_NULL) {
            return true;
        }
    }
    return false;
}

static uint64_t fnva_str(uint64_t h, const std::string & s) {
    for (uint8_t c : s) {
        h ^= c;
        h *= fnva_prime;
    }
    h ^= 0xff;
    h *= fnva_prime;
    return h;
}

std::string hex64(uint64_t v) {
    std::stringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << v;
    return ss.str();
}

std::string geometry_digest(const std::vector<std::string> & parts) {
    uint64_t h = fnva_init;
    for (const auto & p : parts) {
        h = fnva_str(h, p);
    }
    return hex64(h).substr(0, 12);
}

std::string state_base_name(const state_tag & t) {
    std::stringstream ss;
    ss << "kvst_" << t.digest << "_" << t.conv_hash << "_" << t.n_tokens << ".kvstate";
    return ss.str();
}

bool write_sidecar(const std::string & state_path, const state_tag & t, const std::string & model_name) {
    try {
        common_json sidecar = {
            { "digest",     t.digest },
            { "conv_hash",  t.conv_hash },
            { "n_tokens",   t.n_tokens },
            { "model_name", model_name },
            { "saved_at",   (int64_t) std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count() },
        };
        fs::path sidecar_path = fs::path(state_path);
        sidecar_path += ".json";
        std::ofstream ofs(sidecar_path, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            return false;
        }
        ofs << sidecar.dump_safe(4);
        return !!ofs;
    } catch (const std::exception &) {
        return false;
    }
}

static bool parse_sidecar_json(const std::string & contents, state_tag & t, std::string & model_name) {
    try {
        if (contents.size() > 4096) {
            return false;
        }
        const common_json data = common_json::parse(contents);
        if (!data.is_object()) {
            return false;
        }
        if (!data.contains("digest") || !data.at("digest").is_string() ||
            !data.contains("conv_hash") || !data.at("conv_hash").is_string() ||
            !data.contains("n_tokens") || !data.at("n_tokens").is_number_integer()) {
            return false;
        }
        if (data.at("n_tokens").get<long long>() <= 0) {
            return false;
        }
        t.digest     = data.at("digest").get<std::string>();
        t.conv_hash  = data.at("conv_hash").get<std::string>();
        t.n_tokens   = data.at("n_tokens").get<long long>();
        model_name   = data.contains("model_name") && data.at("model_name").is_string() ? data.at("model_name").get<std::string>() : "";
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool read_sidecar(const std::string & sidecar_path, state_tag & t, std::string & model_name) {
    try {
        std::ifstream ifs(fs::path(sidecar_path), std::ios::binary);
        if (!ifs) {
            return false;
        }
        std::stringstream ss;
        ss << ifs.rdbuf();
        return parse_sidecar_json(ss.str(), t, model_name);
    } catch (const std::exception &) {
        return false;
    }
}

bool write_token_file(const std::string & path, const llama_token * data, size_t n_tokens) {
    try {
        std::ofstream ofs(fs::path(path), std::ios::binary | std::ios::trunc);
        if (!ofs) {
            return false;
        }
        ofs.write(reinterpret_cast<const char *>(data), (std::streamsize) (n_tokens * sizeof(llama_token)));
        return !!ofs;
    } catch (const std::exception &) {
        return false;
    }
}

bool read_token_file(const std::string & path, std::vector<llama_token> & out, size_t expect_n) {
    try {
        std::error_code ec;
        const auto sz = fs::file_size(fs::path(path), ec);
        if (ec || sz % sizeof(llama_token) != 0) {
            return false;
        }
        if (sz / sizeof(llama_token) != expect_n) {
            return false;
        }
        std::ifstream ifs(fs::path(path), std::ios::binary);
        if (!ifs) {
            return false;
        }
        out.resize(expect_n);
        ifs.read(reinterpret_cast<char *>(out.data()), (std::streamsize) sz);
        return (size_t) ifs.gcount() == sz;
    } catch (const std::exception &) {
        return false;
    }
}

size_t first_mismatch_index(const llama_token * a, const llama_token * b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return i;
        }
    }
    return n;
}

bool find_best_state(const std::string & dir, const std::string & digest,
                     const prefix_hasher & hasher, size_t n_incoming, candidate & out,
                     std::string * note) {
    std::error_code ec;
    if (!fs::exists(fs::path(dir), ec) || ec) {
        return false;
    }

    std::vector<candidate> cands;

    for (fs::directory_iterator it(fs::path(dir), fs::directory_options::skip_permission_denied, ec), end;
            it != end && !ec; it.increment(ec)) {
        if (ec || !it->is_regular_file(ec) || ec) {
            continue;
        }

        std::string name = it->path().filename().string();
        if (name.size() < 10 || name.compare(name.size() - 5, 5, ".json") != 0) {
            continue;
        }

        state_tag t;
        std::string model_name;
        if (!read_sidecar(it->path().string(), t, model_name)) {
            continue;
        }
        if (t.digest != digest || t.n_tokens <= 0) {
            continue;
        }

        candidate cur;
        cur.state_path = it->path().string().substr(0, it->path().string().size() - 5); // drop ".json"
        cur.tag = std::move(t);

        cands.push_back(std::move(cur));
    }

    if (cands.empty()) {
        return false;
    }

    // longest first
    std::sort(cands.begin(), cands.end(), [](const candidate & a, const candidate & b) {
        return a.tag.n_tokens > b.tag.n_tokens;
    });

    size_t n_refused_superset = 0;
    size_t n_refused_hashfit  = 0;

    // fork A59 phase-1 (instrumentation, no matching-rule change): capture
    // the first-mismatch index for the longest candidate that reaches the
    // token comparison in each refusal class, so the refusal note can
    // classify the divergence (index 0 = client-side re-serialization,
    // unsalvageable by any disk state; index > 0 = interior divergence,
    // salvageable by checkpoint restore in phase 2)
    struct refusal_mm {
        bool        have    = false; // a candidate of this class reached the compare
        bool        scanned = false; // a mismatch index was computed
        size_t      idx     = 0;
        int64_t     n_saved = 0;
        std::string name;
    };
    refusal_mm mm_sup;  // longest saved-longer candidate that reached the prefix check
    refusal_mm mm_hash; // longest saved-fits candidate whose prefix hash refused

    // fork A59 phase 2: track the refused candidate carrying the longest
    // verified shared prefix with the incoming stream (interior divergence).
    // returned as a salvage candidate when no candidate fully verifies, so the
    // caller restores that prefix and re-prefills the tail instead of the whole
    // prompt. only prefixes of length > 0 are salvageable (token-0 divergence
    // is unsalvageable by any disk state).
    candidate best_salvage;
    int64_t   best_salvage_F = 0;
    const auto consider_salvage = [&best_salvage, &best_salvage_F](const candidate & c, size_t fmm) {
        if ((int64_t) fmm > best_salvage_F) {
            best_salvage_F = (int64_t) fmm;
            best_salvage   = c;
        }
    };

    for (const auto & c : cands) {
        const size_t n = (size_t) c.tag.n_tokens;
        if (n <= n_incoming) {
            // saved stream fits inside the incoming prompt: exact-verify the
            // shared prefix (hash of the first n saved incoming tokens)
            if (hex64(hasher.prefix_hash(n)) == c.tag.conv_hash) {
                out = c;
                return true;
            }
            n_refused_hashfit++;
            if (!mm_hash.have) {
                mm_hash.have    = true;
                mm_hash.n_saved = (int64_t) n;
                mm_hash.name    = fs::path(c.state_path).filename().string();
                std::vector<llama_token> saved;
                if (read_token_file(c.state_path + ".tokens", saved, n)) {
                    mm_hash.idx     = first_mismatch_index(saved.data(), hasher.data(), n);
                    mm_hash.scanned = true;
                    consider_salvage(c, mm_hash.idx);
                }
            }
            continue;
        }

        // saved stream is longer than the incoming prompt: only a verified
        // token prefix may be restored (client-side compaction / retried turn).
        // the .tokens companion is the only source of the saved stream's
        // tokens, so a missing companion means "cannot verify" = refuse.
        std::vector<llama_token> saved;
        if (!read_token_file(c.state_path + ".tokens", saved, n)) {
            n_refused_superset++;
            continue;
        }
        if (hex64(hash_tokens(saved.data(), saved.size())) != c.tag.conv_hash) {
            n_refused_superset++;
            continue;
        }
        if (n_incoming == 0 || std::memcmp(saved.data(), hasher.data(), n_incoming * sizeof(llama_token)) != 0) {
            n_refused_superset++;
            if (n_incoming > 0) {
                // single O(n) scan over the compare just performed; the
                // accept path above stays a plain memcmp
                const size_t fmm = first_mismatch_index(saved.data(), hasher.data(), n_incoming);
                if (!mm_sup.have) {
                    mm_sup.idx     = fmm;
                    mm_sup.have    = true;
                    mm_sup.scanned = true;
                    mm_sup.n_saved = (int64_t) n;
                    mm_sup.name    = fs::path(c.state_path).filename().string();
                }
                // fork A59 phase 2: every saved-longer refusal carries a
                // verified prefix [0, fmm); keep the longest across candidates
                consider_salvage(c, fmm);
            }
            continue;
        }
        out = c;
        return true;
    }

    // fork A59 phase 2: no candidate fully verified, but one carries a
    // verified shared prefix of length > 0 (interior divergence). return it as
    // a salvage candidate so the caller restores that prefix and re-prefills
    // the tail instead of the whole prompt.
    if (best_salvage_F > 0) {
        out = best_salvage;
        out.salvage        = true;
        out.salvage_prefix = best_salvage_F;
        if (note != nullptr) {
            *note = string_format("interior divergence: no full match, salvaging verified %lld-token prefix (tail re-prefill via checkpoint rollback)",
                    (long long) best_salvage_F, (long long) best_salvage_F);
        }
        return true;
    }

    if (note != nullptr && (n_refused_superset > 0 || n_refused_hashfit > 0)) {
        auto verdict = [](size_t idx) -> std::string {
            return idx == 0
                ? std::string("verdict: token-0 divergence - unsalvageable by disk stash (client-side token-stream re-serialization)")
                : string_format("verdict: interior divergence at token %zu - salvageable (A59 phase 2)", idx);
        };
        std::string s;
        if (n_refused_superset > 0) {
            s += string_format("%zu longer state(s) refused", n_refused_superset);
            if (mm_sup.scanned) {
                s += string_format(": first mismatch at token %zu of longest prefix-checked candidate (%lld tokens, %s); %s",
                        mm_sup.idx, (long long) mm_sup.n_saved, mm_sup.name.c_str(), verdict(mm_sup.idx).c_str());
            } else {
                s += " (no prefix check reached: missing/corrupt token companion or companion hash mismatch)";
            }
        }
        if (n_refused_hashfit > 0) {
            if (!s.empty()) {
                s += "; ";
            }
            s += string_format("%zu shorter state(s) hash-refused", n_refused_hashfit);
            if (mm_hash.scanned) {
                s += string_format(": first mismatch at token %zu of longest hash-refused candidate (%lld tokens, %s); %s",
                        mm_hash.idx, (long long) mm_hash.n_saved, mm_hash.name.c_str(), verdict(mm_hash.idx).c_str());
            } else {
                s += " (no token companion to scan for the mismatch index)";
            }
        }
        *note = s;
    }
    return false;
}

void prune_dir(const std::string & dir, int64_t limit_bytes, const std::string & keep_path) {
    if (limit_bytes <= 0) {
        return;
    }

    std::error_code ec;
    if (!fs::exists(fs::path(dir), ec) || ec) {
        return;
    }

    struct item {
        fs::path path;
        int64_t  mtime;
        int64_t  size;
    };

    std::vector<item> states;
    int64_t total = 0;

    for (fs::directory_iterator it(fs::path(dir), fs::directory_options::skip_permission_denied, ec), end;
            it != end && !ec; it.increment(ec)) {
        if (ec || !it->is_regular_file(ec) || ec) {
            continue;
        }

        std::string name = it->path().filename().string();
        if (name.size() < 7 || name.compare(name.size() - 7, 7, ".kvstate") != 0) {
            continue;
        }

        item cur;
        cur.path = it->path();
        cur.mtime = std::chrono::duration_cast<std::chrono::seconds>(
                it->last_write_time(ec).time_since_epoch()).count();
        cur.size = (int64_t) it->file_size(ec);
        if (ec) {
            continue;
        }

        states.push_back(cur);
        total += cur.size;
    }

    // oldest first
    std::sort(states.begin(), states.end(), [](const item & a, const item & b) {
        return a.mtime < b.mtime;
    });

    for (const auto & s : states) {
        if (total <= limit_bytes) {
            break;
        }

        std::error_code rm_err;
        bool is_keep = false;
        if (!keep_path.empty()) {
            is_keep = fs::equivalent(s.path, fs::path(keep_path), rm_err) && !rm_err;
        }
        if (is_keep) {
            continue;
        }

        fs::remove(fs::path(s.path.string() + ".json"), rm_err);
        fs::remove(fs::path(s.path.string() + ".tokens"), rm_err);
        fs::remove(s.path, rm_err);
        total -= s.size;
    }
}

bool ensure_dir(const std::string & dir) {
    std::error_code ec;
    fs::create_directories(fs::path(dir), ec);
    return !ec;
}

} // namespace kstash
