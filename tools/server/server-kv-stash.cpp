#include "server-kv-stash.h"

#include "json.h"

#include <algorithm>
#include <chrono>
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
    return hash_tokens(data, n < n_tokens ? n : n_tokens);
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

bool find_best_state(const std::string & dir, const std::string & digest,
                     const prefix_hasher & hasher, size_t n_incoming, candidate & out) {
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

    // candidates fully inside the incoming stream are exact-verified:
    // the saved token hash must equal the hash of the first n saved tokens
    for (const auto & c : cands) {
        const size_t n = (size_t) c.tag.n_tokens;
        if (n > n_incoming) {
            continue;
        }
        if (hex64(hasher.prefix_hash(n)) == c.tag.conv_hash) {
            out = c;
            return true;
        }
    }

    // no exact match: fall back to the longest candidate (the server trims
    // whatever the incoming prompt does not cover)
    for (const auto & c : cands) {
        if ((size_t) c.tag.n_tokens > n_incoming) {
            out = c;
            return true;
        }
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
