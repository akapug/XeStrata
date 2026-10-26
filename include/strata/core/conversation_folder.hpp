// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Where --conversation-save keeps its .xsc files: a local folder, or (homelab) a kvib:// store in other machines' RAM
// (kvib.hpp).  ConversationDisk writes and reads the same bytes through either: a folder lists, opens, renames,
// removes and touches files, and a file is a FolderStream.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct FolderFile {
    std::string name;                        // a file name in the folder, no directory
    std::filesystem::file_time_type time;    // last written or touched
    uint64_t bytes = 0;
};

/// One open file: sequential reads or writes, and the seek a writer needs to put the header back.
class FolderStream {
public:
    virtual ~FolderStream() = default;
    /// Up to `n` bytes; fewer only at the end of the file or on an error (then failed()).
    virtual size_t read(void* p, size_t n) = 0;
    virtual bool write(const void* p, size_t n) = 0;
    virtual bool seek(uint64_t offset) = 0;
    /// The position, -1 on an error.
    virtual int64_t tell() = 0;
    virtual bool failed() const = 0;
    /// A writer's end: flushed and made as durable as the folder can (fsync, or every remote write completed).  A
    /// stream destroyed without it is abandoned (its file is incomplete; the caller removes it).
    virtual bool finish() = 0;
};

class ConversationFolder {
public:
    virtual ~ConversationFolder() = default;
    /// Creates or connects; false (error says why) leaves the folder unusable.
    virtual bool prepare(std::string& error) = 0;
    /// Every regular file, oldest first.
    virtual std::vector<FolderFile> list() = 0;
    virtual std::unique_ptr<FolderStream> open_read(const std::string& name) = 0;
    /// A new, empty file for writing (an existing one of that name is replaced).
    virtual std::unique_ptr<FolderStream> open_write(const std::string& name) = 0;
    virtual bool rename(const std::string& from, const std::string& to) = 0;
    virtual bool remove(const std::string& name) = 0;
    virtual void touch(const std::string& name) = 0;
    virtual std::string where() const = 0;   ///< for messages
};

/// homelab kvib: the CPU side of moving a conversation file - the copies between a stream's staging buffers (or a file)
/// and the conversation's memory, split over a few threads, and the file checksum on a thread of its own beside them
/// (conversation_disk.cpp).  The bytes and the checksum are the same whatever the count.
struct ConversationIoSettings {
    static constexpr int kThreadsDefault = 4;
    int threads = kThreadsDefault;   ///< STRATA_CONV_IO_THREADS (1-64): 1 is the single-threaded path
};
/// STRATA_CONV_IO_THREADS over the default; a bad value is refused by name (false, error).
bool conversation_io_settings_from_env(ConversationIoSettings& s, std::string& error);
/// The process's copy threads (ConversationDisk::open sets them from the setting).
void conversation_io_set_threads(int threads);
int conversation_io_threads();
/// Copies `n` bytes; large copies are split over the copy threads (the caller is one of them).
void conversation_io_copy(void* dst, const void* src, size_t n);
/// Runs fn(i) for i in [0, n) over the copy threads and the caller, and returns once all are done.
void conversation_io_parallel(size_t n, const std::function<void(size_t)>& fn);
/// The calling thread may run on every CPU the process had at its start.  A thread inherits its creator's pin, and
/// the serve loop pins its own thread to one core: a helper thread it starts calls this first.
void conversation_io_widen_thread();

/// A local folder (`dir`).
std::unique_ptr<ConversationFolder> conversation_folder_local(const std::filesystem::path& dir);
/// `spec` = a folder path, or kvib://HOST:PORT[,HOST:PORT...] (kvib.hpp).
std::unique_ptr<ConversationFolder> conversation_folder_open(const std::string& spec);
bool conversation_folder_is_remote(const std::string& spec);

} // namespace strata::core
