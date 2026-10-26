// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// homelab kvib on a device: a layer's K/V and indexer state - every format, resident and streamed (the host-pool copy
// a streamed layer keeps) - captured, parked in a kvib store through ConversationDisk, overwritten in place (a slot
// handed to another conversation), then loaded back and restored, is byte for byte the state it was: the bytes the
// attention reads are the same, so a parked-and-restored conversation decodes exactly as one that never left.  Over
// the in-process store and over loopback TCP.  A flipped byte in the store is caught by the file's checksum.
#include "conversation_fixture.hpp"
#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_park_queue.hpp"
#include "strata/core/kvib.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>

using namespace conversation_fixture;
namespace kvib = strata::core::kvib;

namespace {
void park_and_restore(int fmt, int mode, std::shared_ptr<ConversationFolder> folder, const std::string& via) {
    const std::string tag = via + " fmt " + std::to_string(fmt) + " mode " + std::to_string(mode);
    auto ok = [&](bool v, const std::string& what) { check(v, (tag + ": " + what).c_str()); };
    std::string err;
    ConversationDiskOptions o;
    o.folder = folder;
    o.fingerprint = "conversation_kvib_parity";
    o.budget = size_t{256} << 20;
    ConversationDisk disk;
    ok(disk.open(o, err), "the store opens: " + err);
    for (int64_t upto : {1, 5, 64, 96}) {
        const bool index = mode != 2;
        Fixture f(fmt, mode);
        f.fill(13);
        gpu_check(strata::gpu::device_sync());
        SavedConversation image;
        image.geometry[0] = fmt;
        image.live.ids.resize((size_t) upto);
        for (int64_t i = 0; i < upto; ++i) image.live.ids[(size_t) i] = (int32_t) (1000 + i * 7 + fmt);
        image.live.gdn.assign(4096, (uint8_t) (fmt + mode));
        image.kv.resize(1);
        ok(conversation_kv_save(image.kv[0], f.state, f.g, upto, index, err), "capture: " + err);
        size_t written = 0;
        ok(disk.save(image, written, err) && written > 0, "park in the store: " + err);
        // the slot goes to another conversation
        f.fill(177);
        gpu_check(strata::gpu::device_sync());
        ConversationKv other;
        ok(conversation_kv_save(other, f.state, f.g, upto, index, err) && !equal(other, image.kv[0]),
           "the slot now holds other bytes");
        // the parked conversation's next turn: found, loaded, restored
        std::vector<int32_t> prompt = image.live.ids;
        prompt.push_back(42);
        const auto m = disk.best(prompt, {}, image.cvec);
        ok(m.tokens == upto && m.live, "the next turn finds the whole conversation");
        SavedConversation back;
        ok(disk.load(m.index, back, err), "load: " + err);
        ok(back.kv.size() == 1 && equal(back.kv[0], image.kv[0]), "the loaded K/V equals the captured K/V");
        ok(conversation_kv_restore(back.kv[0], f.state, f.g, upto, index, err), "restore: " + err);
        gpu_check(strata::gpu::device_sync());
        ConversationKv now;
        ok(conversation_kv_save(now, f.state, f.g, upto, index, err) && equal(now, image.kv[0]),
           "the restored state reads back byte for byte");
        uint64_t fp = 0;
        ok(conversation_kv_verify(image.kv[0], f.state, f.g, upto, index, fp, err),
           "the authoritative and resident bytes verify: " + err);
        if (mode == 1) {
            std::vector<int32_t> table((size_t) f.state.n_pages);
            gpu_check(strata::gpu::copy(table.data(), f.state.page_table, table.size() * 4));
            ok(std::all_of(table.begin(), table.end(), [](int32_t x) { return x == -1; }),
               "a streamed layer's stale VRAM pages are invalidated (the next step fetches the restored bytes)");
        }
    }
    // a flipped byte in the stored file: the load refuses it, the state is untouched
    std::string name;
    for (const auto& file : folder->list()) name = file.name;
    ok(!name.empty(), "the store holds the file");
    std::vector<uint8_t> bytes;
    {
        auto r = folder->open_read(name);
        uint8_t buf[65536];
        for (size_t k; r && (k = r->read(buf, sizeof buf)) > 0;) bytes.insert(bytes.end(), buf, buf + k);
    }
    bytes[bytes.size() - 100] ^= 0x10;
    {
        auto w = folder->open_write(name + ".tmp");
        ok(w && w->write(bytes.data(), bytes.size()) && w->finish(), "rewrite it damaged");
    }
    ok(folder->rename(name + ".tmp", name), "commit the damaged file");
    ConversationDisk again;
    ok(again.open(o, err) && again.size() >= 1, "a fresh index reads its header");
    SavedConversation x;
    bool refused = false;
    for (size_t i = again.size(); i-- > 0;) refused = !again.load(i, x, err) || refused;
    ok(refused && err.find("checksum") != std::string::npos, "the damage is caught: " + err);
}
// homelab background park: the capture handed to the writer thread, the slot overwritten, then the next turn - its image
// taken back from the writer (queued or mid-write), or the file the writer committed - restores byte for byte
void park_async(int fmt, int mode, kvib::Server& server, const kvib::Settings& cs) {
    const std::string tag = "async fmt " + std::to_string(fmt) + " mode " + std::to_string(mode);
    auto ok = [&](bool v, const std::string& what) { check(v, (tag + ": " + what).c_str()); };
    std::string err;
    std::shared_ptr<ConversationFolder> main(kvib::folder_local(server, cs));
    ConversationDiskOptions o;
    o.folder = main;
    o.folder_factory = [&]() -> std::shared_ptr<ConversationFolder> {
        std::shared_ptr<ConversationFolder> f(kvib::folder_local(server, cs));
        return f;
    };
    o.fingerprint = "conversation_kvib_parity_async";
    o.budget = size_t{256} << 20;
    ConversationDisk disk;
    ok(disk.open(o, err), "the store opens: " + err);
    auto wf = disk.open_writer_folder(err);
    ok(wf != nullptr, "the writer's connection: " + err);
    ConversationParkQueue q(disk.options(), wf, 2, size_t{1} << 30);
    const bool index = mode != 2;
    for (int how = 0; how < 2; ++how) {
        const int64_t upto = how == 0 ? 64 : 96;
        Fixture f(fmt, mode);
        f.fill(21 + how);
        gpu_check(strata::gpu::device_sync());
        SavedConversation image;
        image.geometry[0] = fmt;
        image.live.ids.resize((size_t) upto);
        for (int64_t i = 0; i < upto; ++i) image.live.ids[(size_t) i] = (int32_t) (3000 + i * 5 + fmt + how);
        image.live.gdn.assign(4096, (uint8_t) (fmt + mode + how));
        image.kv.resize(1);
        ok(conversation_kv_save(image.kv[0], f.state, f.g, upto, index, err), "capture: " + err);
        const ConversationKv captured = image.kv[0];
        double waited = 0;
        if (how == 0) {   // another conversation's long write ahead of it, so this one is still queued at its next turn
            SavedConversation filler;
            filler.geometry[0] = fmt;
            filler.live.ids.assign(50, 77);
            filler.live.gdn.assign(size_t{192} << 20, 3);
            q.push(std::move(filler), waited);
        }
        const uint64_t id = q.push(std::move(image), waited);
        f.fill(199);   // the slot goes to another conversation while the image is queued or being written
        gpu_check(strata::gpu::device_sync());
        std::vector<int32_t> prompt;
        for (int64_t i = 0; i < upto; ++i) prompt.push_back((int32_t) (3000 + i * 5 + fmt + how));
        prompt.push_back(42);
        SavedConversation back;
        if (how == 0) {   // the next turn arrives at once: the image itself
            const auto m = q.best(prompt, {}, true);   // a small image may already be written (then m.id is 0)
            ok(m.id == 0 || (m.id == id && m.live && m.tokens == upto), "the next turn matches the queued image");
            auto img = m.id == id ? q.claim(id) : std::nullopt;
            if (!img) {   // the write finished first: it is a file now
                for (auto& d : q.done()) if (d.ok) disk.adopt(d.file);
                const auto dm = disk.best(prompt, {}, true);
                ok(dm.tokens == upto && disk.load(dm.index, back, err), "or its file loads: " + err);
            } else {
                back = std::move(*img);
            }
        } else {          // later: the file the writer committed
            q.drain();
            size_t adopted = 0;
            for (auto& d : q.done()) if (d.ok) { disk.adopt(d.file); ++adopted; }
            ok(adopted >= 1, "the background write landed");
            const auto dm = disk.best(prompt, {}, true);
            ok(dm.tokens == upto && dm.live && disk.load(dm.index, back, err), "the next turn loads its file: " + err);
        }
        ok(back.kv.size() == 1 && equal(back.kv[0], captured), "the K/V it restores equals the captured K/V");
        ok(conversation_kv_restore(back.kv[0], f.state, f.g, upto, index, err), "restore: " + err);
        gpu_check(strata::gpu::device_sync());
        ConversationKv now;
        ok(conversation_kv_save(now, f.state, f.g, upto, index, err) && equal(now, captured),
           "the restored state reads back byte for byte");
    }
}
} // namespace

int main() {
    void* probe = strata::gpu::alloc_device(1);   // no usable GPU: skipped
    if (probe == nullptr) return 77;
    strata::gpu::free(probe);
    kvib::ServerOptions so;
    so.arena_bytes = uint64_t{512} << 20;
    so.extent_bytes = uint64_t{1} << 20;
    so.transport = "tcp";
    kvib::Settings cs;
    cs.stage_mib = 1;
    cs.stage_depth = 3;
    cs.tcp_streams = 2;
    cs.transport = "tcp";
    std::string err;
    kvib::Server local(so);
    check(local.start(false, err), "an in-process store starts");
    const std::string ep = "127.0.0.1:" + std::to_string(20000 + (int) (::getpid() % 20000));
    so.listen = {ep};
    kvib::Server tcp(so);
    check(tcp.start(true, err), "a loopback TCP store starts");
    const std::array<std::pair<int, int>, 7> cases = {{{kKvF16, 0}, {kKvF16, 1}, {kKvInt8, 0}, {kKvInt8, 1},
                                                      {kKvQ4, 0}, {kKvQ4, 1}, {3, 0}}};
    for (const auto& [fmt, mode] : cases) {
        std::shared_ptr<ConversationFolder> a(kvib::folder_local(local, cs));
        park_and_restore(fmt, mode, a, "local");
        std::shared_ptr<ConversationFolder> b(kvib::folder(ep, cs));
        park_and_restore(fmt, mode, b, "tcp");
        park_async(fmt, mode, local, cs);
    }
    std::printf("conversation_kvib_parity: %d checks passed\n", checks);
    return 0;
}
