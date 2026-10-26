// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// kvib_bench (homelab): what parking a large conversation in another machine's RAM costs, through the engine's own
// --conversation-save code (ConversationDisk over a kvib:// store, or over a local folder for the baseline).
//   kvib_bench --store kvib://HOST:PORT | DIR [--gib 4] [--tokens 200000] [--gdn-mib 64] [--reps 3]
//              [--raw-gib 0 (= --gib)] [--verify 1] [--block-bytes 3264] [--block-reads 20000] [--seed 1] [--async 0]
// (STRATA_CONV_IO_THREADS sets the copy threads; --async 1 adds the background park: the time push() holds the caller -
// the serve thread's stall - the writer's wall time, and a claim of an image mid-write; --diff-dir DIR, a debug aid,
// writes the conversation to the store and to DIR and compares the two files byte for byte, then stops; --prefetch 1
// adds the restore prefetch: the serve thread's time per restore when an admission is held back for its read (the
// polls plus the take), and when a hint came L ms ahead, for L at 0, 25, 50, 75, 100 and 125 % of the read's time)
// Prints key=value lines: the save (park) and load (restore) wall time and bytes/s of a synthetic conversation, a raw
// sequential stream write and read (--verify 0: the raw read does not compare, so it is the transport plus one copy out
// of the staging buffer), this client's single-thread memcpy rate (the copy every path makes), and (kvib only) one-block
// synchronous read latency.  Every loaded conversation byte is compared.
#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_park_queue.hpp"
#include "strata/core/conversation_prefetch.hpp"
#include "strata/core/kvib.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;

namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

struct Options {
    std::string store;
    double gib = 4;
    int64_t tokens = 200000;
    size_t gdn_mib = 64;
    int reps = 3;
    double raw_gib = 0;
    size_t block_bytes = 3264;   // one K8V4 cache block (816 B a cell, 4 cells, 2 KV heads)
    size_t block_reads = 20000;
    uint64_t seed = 1;
    bool verify = true;
    bool async = false;   // also the background park: the caller's stall, the writer's time, a claim mid-write
    bool prefetch = false;   // also the restore prefetch: the serve thread's part of a restore, and the lead it needs
    std::string diff_dir;  // debug: the conversation also written to this folder, the two files compared in 64 KiB reads
};

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--store") o.store = v;
        else if (a == "--gib") o.gib = std::atof(v.c_str());
        else if (a == "--tokens") o.tokens = std::atoll(v.c_str());
        else if (a == "--gdn-mib") o.gdn_mib = (size_t) std::atoll(v.c_str());
        else if (a == "--reps") o.reps = std::atoi(v.c_str());
        else if (a == "--raw-gib") o.raw_gib = std::atof(v.c_str());
        else if (a == "--block-bytes") o.block_bytes = (size_t) std::atoll(v.c_str());
        else if (a == "--block-reads") o.block_reads = (size_t) std::atoll(v.c_str());
        else if (a == "--seed") o.seed = (uint64_t) std::atoll(v.c_str());
        else if (a == "--verify") o.verify = std::atoi(v.c_str()) != 0;
        else if (a == "--async") o.async = std::atoi(v.c_str()) != 0;
        else if (a == "--prefetch") o.prefetch = std::atoi(v.c_str()) != 0;
        else if (a == "--diff-dir") o.diff_dir = v;
        else return false;
    }
    if (argc % 2 == 0) return false;
    if (o.raw_gib <= 0) o.raw_gib = o.gib;
    return !o.store.empty() && o.gib > 0 && o.tokens > 1 && o.reps > 0 && o.block_bytes > 0;
}

void fill(ConversationBuffer& b, size_t n, std::mt19937_64& rng) {
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t k, size_t) {
        size_t i = 0;
        for (; i + 8 <= k; i += 8) {
            const uint64_t x = rng();
            std::memcpy(p + i, &x, 8);
        }
        for (; i < k; ++i) p[i] = (uint8_t) rng();
        return true;
    });
}
void fill(std::vector<uint8_t>& b, size_t n, std::mt19937_64& rng) {
    b.resize(n);
    for (auto& x : b) x = (uint8_t) rng();
}

// 12 streamed layers and the draft layer, hybrid K/V, the payload split between K and V
SavedConversation synthetic(const Options& o) {
    std::mt19937_64 rng(o.seed);
    SavedConversation s;
    s.geometry[0] = 1;
    s.live.ids.resize((size_t) o.tokens);
    for (auto& id : s.live.ids) id = (int32_t) (rng() % 150000);
    fill(s.live.gdn, o.gdn_mib << 20, rng);
    fill(s.live.tails, 4096, rng);
    ConversationCheckpoint half;
    half.ids.assign(s.live.ids.begin(), s.live.ids.begin() + o.tokens / 2);
    fill(half.gdn, o.gdn_mib << 20, rng);
    s.checkpoints.push_back(std::move(half));
    const int layers = 13;
    const size_t per = (size_t) (o.gib * (double) (uint64_t{1} << 30)) / (size_t) (layers * 2);
    s.kv.resize(layers);
    for (auto& kv : s.kv) {
        kv.format = strata::kernels::kKvHybrid;
        kv.cells = o.tokens;
        kv.heads = 2;
        kv.head_dim = 512;
        kv.page_size = 4;
        fill(kv.k, per, rng);
        fill(kv.v, per, rng);
        fill(kv.k_scale, per / 64, rng);
        fill(kv.v_scale, per / 64, rng);
    }
    return s;
}

size_t payload(const SavedConversation& s) {
    size_t n = s.live.gdn.size() + s.live.tails.size();
    for (const auto& c : s.checkpoints) n += c.gdn.size();
    for (const auto& kv : s.kv) n += kv.k.size() + kv.v.size() + kv.k_scale.size() + kv.v_scale.size();
    return n;
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.live.ids != b.live.ids || a.live.gdn != b.live.gdn || a.live.tails != b.live.tails ||
        a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (a.checkpoints[i].ids != b.checkpoints[i].ids || a.checkpoints[i].gdn != b.checkpoints[i].gdn) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto &x = a.kv[i], &y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || !(x.k == y.k) || !(x.v == y.v) ||
            !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale))
            return false;
    }
    return true;
}

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[(size_t) std::min<double>((double) v.size() - 1, p * (double) (v.size() - 1))];
}

void stats(const char* key, const std::vector<double>& s, size_t bytes) {
    double best = 1e30, sum = 0;
    for (const double x : s) {
        best = std::min(best, x);
        sum += x;
    }
    const double mean = sum / (double) s.size();
    std::printf("%s_bytes=%zu %s_s_best=%.3f %s_s_mean=%.3f %s_GBps_best=%.2f %s_GBps_mean=%.2f\n", key, bytes, key, best,
                key, mean, key, (double) bytes / best / 1e9, key, (double) bytes / mean / 1e9);
}

int fail(const std::string& why) {
    std::fprintf(stderr, "kvib_bench: %s\n", why.c_str());
    return 1;
}
} // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse(argc, argv, o)) {
        std::fprintf(stderr, "usage: see the header of tools/kvib/kvib_bench.cpp\n");
        return 2;
    }
    const bool remote = conversation_folder_is_remote(o.store);
    std::shared_ptr<kvib::Folder> kf;
    std::shared_ptr<ConversationFolder> folder;
    if (remote) {
        kf = kvib::folder(o.store.substr(std::string(kvib::kScheme).size()));
        folder = kf;
    } else {
        folder = conversation_folder_local(o.store);
    }
    std::string err;
    if (!folder->prepare(err)) return fail(err);
    std::printf("store=%s transport=%s\n", o.store.c_str(), remote ? kf->transport().c_str() : "file");

    // the engine's path: ConversationDisk::save (park) and ::load (restore)
    const auto t_build = Clock::now();
    const SavedConversation image = synthetic(o);
    const size_t bytes = payload(image);
    std::fprintf(stderr, "kvib_bench: built a %.2f GiB conversation of %lld tokens in %.1f s\n",
                 (double) bytes / (1u << 30), (long long) o.tokens, since(t_build));
    ConversationDisk disk;
    ConversationDiskOptions d;
    d.dir = o.store;
    d.folder = folder;
    d.fingerprint = "kvib_bench";
    d.budget = (size_t) -1 / 2;
    if (!disk.open(d, err)) return fail(err);
    std::vector<double> save_s, load_s;
    size_t written = 0;
    if (!o.diff_dir.empty()) {
        auto local = std::shared_ptr<ConversationFolder>(conversation_folder_local(o.diff_dir));
        ConversationDiskOptions l = d;
        l.folder = local;
        ConversationDisk ld;
        if (!ld.open(l, err) || !ld.save(image, written, err) || !disk.save(image, written, err))
            return fail("diff: " + err);
        std::string name;
        for (const auto& f : local->list()) name = f.name;
        auto a = local->open_read(name), b = folder->open_read(name);
        if (!a || !b) return fail("diff: cannot open " + name);
        std::vector<uint8_t> x(65536), y(65536);
        uint64_t at = 0, diffs = 0, first = 0;
        for (;;) {
            const size_t n = a->read(x.data(), x.size()), m = b->read(y.data(), y.size());
            if (n != m) { std::printf("diff: lengths differ at %llu\n", (unsigned long long) at); break; }
            if (n == 0) break;
            for (size_t i = 0; i < n; ++i)
                if (x[i] != y[i]) {
                    if (diffs++ == 0) first = at + i;
                    if (diffs <= 16)
                        std::printf("diff: at %llu folder %02x store %02x\n", (unsigned long long) (at + i), x[i], y[i]);
                }
            at += n;
        }
        for (size_t l = 0; l < image.kv.size(); ++l) {   // the last bytes of each layer's V, as the image holds them
            uint8_t t[2] = {};
            image.kv[l].v.read(t, image.kv[l].v.size() - 2, 2);
            std::printf("diff: image layer %zu v tail %02x %02x\n", l, t[0], t[1]);
        }
        std::printf("diff: %llu bytes compared, %llu differ, first at %llu\n", (unsigned long long) at,
                    (unsigned long long) diffs, (unsigned long long) first);
        return 0;
    }
    for (int r = 0; r < o.reps; ++r) {
        const auto t = Clock::now();
        if (!disk.save(image, written, err)) return fail("save: " + err);
        save_s.push_back(since(t));
        const auto m = disk.best(image.live.ids, {}, image.cvec);
        if (m.tokens == 0) return fail("the saved conversation is not found");
        // a prompt longer than the conversation finds it (the last token starts the next window)
        SavedConversation back;
        const auto t2 = Clock::now();
        if (!disk.load(m.index, back, err)) return fail("load: " + err);
        load_s.push_back(since(t2));
        if (!same(image, back)) return fail("the loaded conversation differs from the saved one");
    }
    std::printf("conversation_tokens=%lld file_bytes=%zu identical=1 io_threads=%d\n", (long long) o.tokens, written,
                conversation_io_threads());
    stats("park", save_s, written);
    stats("restore", load_s, written);
    if (o.async) {   // the background park, through a second connection as the engine makes it
        auto wf = disk.open_writer_folder(err);
        if (!wf) return fail("writer folder: " + err);
        ConversationParkQueue q(disk.options(), wf, 2, (size_t) -1 / 2);
        std::vector<double> push_ms, bg_s, claim_ms;
        for (int r = 0; r < o.reps; ++r) {
            SavedConversation copy = image;   // not timed: the capture's host image
            double waited = 0;
            auto t = Clock::now();
            q.push(std::move(copy), waited);
            push_ms.push_back(since(t) * 1e3);
            q.drain();
            bool ok = false;
            for (auto& d : q.done()) {
                ok = d.ok;
                bg_s.push_back(d.write_ms / 1e3);
                if (d.ok) disk.adopt(d.file);
                else return fail("background write: " + d.error);
            }
            if (!ok) return fail("no background write finished");
            // the next turn arrives while its park is being written: the image comes back
            copy = image;
            const uint64_t id = q.push(std::move(copy), waited);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            t = Clock::now();
            auto back = q.claim(id);
            claim_ms.push_back(since(t) * 1e3);
            if (back && !same(image, *back)) return fail("a claimed image differs");
            q.drain();
            for (auto& d : q.done()) if (d.ok) disk.adopt(d.file);
        }
        auto best = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
        auto mean = [](const std::vector<double>& v) {
            double x = 0;
            for (double y : v) x += y;
            return x / (double) v.size();
        };
        std::printf("async_push_ms_best=%.3f async_push_ms_mean=%.3f async_write_s_best=%.3f async_write_s_mean=%.3f "
                    "async_write_GBps_best=%.2f claim_mid_write_ms_best=%.1f claim_mid_write_ms_mean=%.1f\n",
                    best(push_ms), mean(push_ms), best(bg_s), mean(bg_s), (double) written / best(bg_s) / 1e9,
                    best(claim_ms), mean(claim_ms));
    }

    if (o.prefetch) {   // the restore prefetch, through a loader connection of its own as the engine makes it
        ConversationPrefetch::Options po;
        po.disk = disk.options();
        po.folder = disk.open_writer_folder(err);
        if (!po.folder) return fail("loader folder: " + err);
        po.locate = [&](const ConversationPrefetch::Hint& h) { return disk.locate(h.ids, {}, h.cvec); };
        po.max_bytes = (size_t) -1 / 2;
        ConversationPrefetch pf(std::move(po));
        if (!disk.save(image, written, err)) return fail("save: " + err);
        std::vector<int32_t> prompt = image.live.ids;
        prompt.push_back(7);
        const auto m = disk.best(prompt, {}, image.cvec);
        if (m.tokens == 0) return fail("prefetch: the saved conversation is not found");
        const auto target = disk.target(m.index, m);
        std::string text;
        for (size_t i = 0; i < prompt.size(); ++i) text += (i ? "," : "") + std::to_string(prompt[i]);
        auto ms = [](Clock::time_point t) { return since(t) * 1e3; };
        {   // the serve thread's match of a held admission: the line parsed (as parse_bgen) and the index searched
            ConversationPrefetch::Hint h;
            std::string why;
            auto t = Clock::now();
            if (!ConversationPrefetch::parse(text, h, why)) return fail("prefetch: parse: " + why);
            const double parse_ms = ms(t);
            t = Clock::now();
            const auto bm = disk.best(h.ids, {}, true);
            std::printf("prefetch_match_parse_ms=%.2f prefetch_match_best_ms=%.2f line_bytes=%zu matched=%lld\n", parse_ms,
                        ms(t), text.size(), (long long) bm.tokens);
        }
        // held admission: the serve thread polls settled() between decode steps (1 ms apart here), then takes
        std::vector<double> held_wall, held_serve, read_ms;
        for (int r = 0; r < o.reps; ++r) {
            const auto t0 = Clock::now();
            double serve = 0;
            auto t = Clock::now();
            const uint64_t ticket = pf.load(target);
            serve += ms(t);
            for (;;) {
                t = Clock::now();
                const bool done = pf.settled(ticket);
                serve += ms(t);
                if (done) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            held_wall.push_back(ms(t0));
            double waited = 0;
            std::string note;
            t = Clock::now();
            auto img = pf.take(target, waited, note);
            serve += ms(t);
            if (!img) return fail("prefetch: the held read was not taken: " + note);
            if (r == 0 && !same(image, *img)) return fail("prefetch: the read-ahead image differs");
            held_serve.push_back(serve);
            t = Clock::now();
            img.reset();   // (in the engine the image's K/V goes on into the RAM cache: not a serve-thread free)
            read_ms.push_back(ms(t));
        }
        auto best = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
        std::printf("prefetch_held_wall_ms_best=%.1f prefetch_held_serve_ms_best=%.3f prefetch_held_serve_ms_worst=%.3f "
                    "image_free_ms_best=%.1f\n", best(held_wall), best(held_serve),
                    *std::max_element(held_serve.begin(), held_serve.end()), best(read_ms));
        // a hint L ms ahead of the admission: what the take still blocks the serve thread
        const double full = best(held_wall);
        for (const double frac : {0.0, 0.25, 0.5, 0.75, 1.0, 1.25}) {
            std::vector<double> blocked, hint_ms;
            for (int r = 0; r < o.reps; ++r) {
                auto t = Clock::now();
                pf.hint(text);
                hint_ms.push_back(ms(t));
                std::this_thread::sleep_for(std::chrono::microseconds((int64_t) (frac * full * 1000)));
                double waited = 0;
                std::string note;
                t = Clock::now();
                auto img = pf.take(target, waited, note);
                const double b = ms(t);
                if (!img) {   // not started yet (the hint is still parsed and matched): the serve thread reads it
                    SavedConversation back;
                    t = Clock::now();
                    if (!disk.load(disk.best(prompt, {}, image.cvec).index, back, err)) return fail("load: " + err);
                    blocked.push_back(b + ms(t));
                } else {
                    blocked.push_back(b);
                }
                img.reset();
                while (pf.stats().resolved < pf.stats().hints) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                pf.drop(target.name);   // a hint matched after the take: its read is not wanted
                while (pf.stats().jobs != 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            std::printf("prefetch_lead_pct=%.0f lead_ms=%.0f serve_blocked_ms_best=%.1f serve_blocked_ms_mean=%.1f "
                        "hint_call_ms=%.3f\n", frac * 100, frac * full, best(blocked),
                        [&] { double x = 0; for (double y : blocked) x += y; return x / (double) blocked.size(); }(),
                        best(hint_ms));
        }
        const auto st = pf.stats();
        std::printf("prefetch_reads=%llu prefetch_read_ms_mean=%.1f\n", (unsigned long long) st.loads,
                    st.loads ? st.load_ms / (double) st.loads : 0.0);
    }

    // the transport alone: one stream written and read sequentially, 16 MiB at a time
    const size_t chunk = size_t{16} << 20, chunks = std::max<size_t>(1, (size_t) (o.raw_gib * 1024 / 16));
    std::vector<uint8_t> buf, in(chunk);
    std::mt19937_64 rng(o.seed + 1);
    fill(buf, chunk, rng);
    std::vector<double> w_s, r_s;
    for (int r = 0; r < o.reps; ++r) {
        auto t = Clock::now();
        auto w = folder->open_write("raw.bin.tmp");
        if (!w) return fail("raw: cannot open for writing");
        for (size_t c = 0; c < chunks; ++c) {
            std::memcpy(buf.data(), &c, sizeof c);
            if (!w->write(buf.data(), chunk)) return fail("raw: write failed");
        }
        if (!w->finish()) return fail("raw: finish failed");
        w.reset();
        if (!folder->rename("raw.bin.tmp", "raw.bin")) return fail("raw: commit failed");
        w_s.push_back(since(t));
        t = Clock::now();
        auto rd = folder->open_read("raw.bin");
        if (!rd) return fail("raw: cannot open for reading");
        for (size_t c = 0; c < chunks; ++c) {
            if (rd->read(in.data(), chunk) != chunk) return fail("raw: short read");
            size_t got = 0;
            std::memcpy(&got, in.data(), sizeof got);
            if (got != c || (o.verify && std::memcmp(in.data() + 8, buf.data() + 8, chunk - 8) != 0))
                return fail("raw: bytes differ");
        }
        r_s.push_back(since(t));
    }
    stats("raw_write", w_s, chunks * chunk);
    stats("raw_read", r_s, chunks * chunk);
    {   // the copy every path makes once (staging buffer <-> the caller's memory), on this client, one thread
        std::vector<uint8_t> a(chunk), b(chunk);
        std::memset(a.data(), 1, chunk);
        std::memset(b.data(), 2, chunk);
        const auto t = Clock::now();
        for (size_t c = 0; c < chunks; ++c) std::memcpy((c & 1) ? a.data() : b.data(), (c & 1) ? b.data() : a.data(), chunk);
        std::printf("memcpy_GBps=%.2f\n", (double) (chunks * chunk) / since(t) / 1e9);
    }

    // one cache block at a time: what a fetch on demand would cost
    if (remote && o.block_reads > 0) {
        std::vector<uint8_t> blk(o.block_bytes);
        std::vector<double> us;
        us.reserve(o.block_reads);
        const uint64_t n_blocks = (uint64_t) chunks * chunk / o.block_bytes;
        std::mt19937_64 pick(o.seed + 2);
        const auto t = Clock::now();
        for (size_t i = 0; i < o.block_reads; ++i) {
            const uint64_t at = pick() % n_blocks * o.block_bytes;
            const auto t1 = Clock::now();
            if (!kf->read_at("raw.bin", at, blk.data(), blk.size(), err)) return fail("block read: " + err);
            us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t1).count());
        }
        const double total = since(t);
        std::printf("block_bytes=%zu block_reads=%zu block_us_p50=%.1f block_us_p99=%.1f blocks_per_s=%.0f "
                    "block_MBps=%.1f\n", o.block_bytes, o.block_reads, pct(us, 0.5), pct(us, 0.99),
                    (double) o.block_reads / total, (double) o.block_reads * (double) o.block_bytes / total / 1e6);
    }
    folder->remove("raw.bin");
    uint64_t used = 0, cap = 0;
    if (remote && kf->usage(used, cap, err))
        std::printf("arena_used_bytes=%llu arena_cap_bytes=%llu\n", (unsigned long long) used, (unsigned long long) cap);
    return 0;
}
