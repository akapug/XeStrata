// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// --serve's parked conversations on disk (--conversation-save): a conversation leaving the RAM cache, and every
// one held at QUIT, is written to a folder and survives a restart.  Host only; the device restore stays
// conversation_snapshot_restore's.  A file is a computation cache, never the chat's record: any file that does not
// match (another model or engine, a damaged one) is ignored or deleted, and the prompt is read the ordinary way.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_folder.hpp"

#include <chrono>
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace strata::core {

// Whether this build has c-blosc2 (CMake found libblosc2): needed to write compressed files and to read them.
bool conversation_disk_can_compress();

// What a saved conversation's state depends on beyond the geometry: canonical path, size and modification time of
// a file, or of every regular file directly in a folder.  "-" when the path is empty, "?" when it cannot be read.
std::string conversation_file_identity(const std::string& path);

struct ConversationDiskOptions {
    std::filesystem::path dir;     // a folder, or kvib://HOST:PORT[,...] (homelab: other machines' RAM, kvib.hpp)
    std::shared_ptr<ConversationFolder> folder;   // when set, used instead of `dir` (tests)
    std::string fingerprint;       // the model, engine and settings the files must have been written with
    size_t budget = 0;             // bytes of .xsc files the folder may hold
    std::chrono::seconds max_age{0};  // since the file was last written or restored
    size_t checkpoints = 2;        // checkpoints written per conversation: deepest, chain root, then newest use
    bool compress = false;         // the floating-point parts through c-blosc2 (needs conversation_disk_can_compress)
    size_t stage_parts = 0;        // a layer split's later GPUs: each state is followed by theirs (the fingerprint
                                   // names the split, so a file is read only by an engine split the same way)
    // homelab: a new connection to the same folder, for a writer on another thread (tests: another in-process store
    // client); unset: `dir` opened again
    std::function<std::shared_ptr<ConversationFolder>()> folder_factory;
};

class ConversationDisk {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
    };

    bool enabled() const { return enabled_; }
    // Creates the folder (0700) or connects to the store, removes stale .tmp files, expires old files and indexes the
    // usable ones.
    bool open(ConversationDiskOptions options, std::string& error);
    size_t size() const { return entries_.size(); }
    size_t folder_bytes() const { return folder_bytes_; }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // ties prefer the most recently written or restored file
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            const int64_t n = conversation_prefix(e.live, prompt, images);
            if (n > best.tokens) best = {i, n, true};
            for (const auto& c : e.checkpoints) {
                const int64_t k = conversation_prefix(c, prompt, images);
                if (k > best.tokens) best = {i, k, false};
            }
        }
        return best;
    }
    // The payload bytes a load allocates (for the RAM admission before it).
    size_t load_bytes(size_t index) const { return entries_.at(index).payload; }
    // Reads and checks the whole file.  A file that fails is deleted (false, error says why); the device is untouched.
    bool load(size_t index, SavedConversation& image, std::string& error);

    // homelab restore prefetch (conversation_prefetch.hpp): a file as another thread names it.  `serial` changes
    // whenever the index takes the file again (a newer write under the same name), so an image read before that is
    // never taken for the current one.
    struct Target {
        std::string name;
        uint64_t serial = 0;
        size_t payload = 0;
        int64_t tokens = 0;   // of the prompt it was matched against
        bool live = false;
    };
    Target target(size_t index, const Match& m) const {
        const Entry& e = entries_.at(index);
        return {e.name, e.serial, e.payload, m.tokens, m.live};
    }
    // best() then target(), from any thread: the index is read under its lock (this object's thread changes it under
    // the same lock).  nullopt: no file holds a start of the prompt.
    template<class Token>
    std::optional<Target> locate(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                                 bool cvec) const {
        std::lock_guard<std::mutex> lock(index_m_);
        const Match m = best(prompt, images, cvec);
        if (m.tokens <= 0) return std::nullopt;
        return target(m.index, m);
    }
    // Reads and checks file `name` through `folder` without the index (any thread, its own folder connection; a set
    // `cancel` stops it early).  load() is read_file plus deleting a file that fails.
    static bool read_file(const ConversationDiskOptions& options, ConversationFolder& folder, const std::string& name,
                          SavedConversation& image, std::string& error, const std::atomic<bool>* cancel = nullptr);
    // The file was used: its age starts again, and it becomes the most recent.
    void touch(size_t index);
    // Writes `image` (its checkpoints cut to the configured count), then deletes the files it supersedes and the
    // oldest ones over the budget.  A conversation whose uncompressed size exceeds the budget is not written.
    // `written`: file bytes.  write_file then adopt.
    bool save(const SavedConversation& image, size_t& written, std::string& error);

    // homelab background park: a save in two halves.  write_file writes the file and commits it under its final name;
    // it touches only `folder` and `options`, never the index, so it runs on another thread with its own folder
    // connection (open_writer_folder) while this object serves the request path.  adopt (this object's thread) then
    // indexes the file, deletes the ones it supersedes and expires.  A set `cancel` stops the write early (false,
    // "cancelled"; nothing is committed).
    struct Written {
        std::string name;
        size_t bytes = 0, payload = 0;
        ConversationCheckpoint live;                     // ids and image keys only
        std::vector<ConversationCheckpoint> checkpoints; // the kept ones, ids and image keys only
        std::vector<ConversationCheckpoint> chain;       // every checkpoint the image had, ids and image keys (#342)
        bool cvec = true;
    };
    static bool write_file(const ConversationDiskOptions& options, ConversationFolder& folder,
                           const SavedConversation& image, Written& out, std::string& error,
                           const std::atomic<bool>* cancel = nullptr);
    void adopt(const Written& file);
    const ConversationDiskOptions& options() const { return options_; }
    // A second connection to this folder, prepared (nullptr: error says why).
    std::shared_ptr<ConversationFolder> open_writer_folder(std::string& error) const;
    // Deletes files older than max_age, then the oldest while the folder is over its budget.  Returns how many.
    size_t expire();

private:
    struct Entry {
        std::string name;                         // the file's name in the folder
        ConversationCheckpoint live;              // ids and image keys only
        std::vector<ConversationCheckpoint> checkpoints;
        bool cvec = true;
        size_t payload = 0;
        uint64_t serial = 0;
    };
    bool remove_file(const std::string& name);
    bool enabled_ = false;
    mutable std::mutex index_m_;   // entries_ changes under it (this object's thread); locate() reads under it
    uint64_t next_serial_ = 1;
    ConversationDiskOptions options_;
    std::shared_ptr<ConversationFolder> folder_;
    std::vector<Entry> entries_;   // least recently used first
    size_t folder_bytes_ = 0;
};

} // namespace strata::core
