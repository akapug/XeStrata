// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// kvib (homelab): parked conversations in another machine's RAM, over RDMA or TCP.
//
// A daemon (`strata-kvib-store`, tools/kvib) keeps named objects in RAM: each a list of extents, mapped and registered
// for RDMA when they are allocated and released when the object goes, under a cap on the whole arena.  The engine's
// --conversation-save folder becomes `kvib://HOST:PORT[,HOST:PORT...]` (conversation_folder.hpp): the same .xsc files,
// written and read through FILE* streams whose bytes go to and from the daemons.
//
// Control: one TCP connection a daemon (the wire below).  Data, by STRATA_KVIB_TRANSPORT:
//   rdma - the client RDMA-WRITEs and RDMA-READs the daemon's extents through a registered staging ring (an RC queue
//          pair through rdma_cm, RoCE or InfiniBand).  libibverbs / librdmacm are opened with dlopen only then: the
//          engine has no runtime dependency on them.
//   tcp  - WRITE / READ requests carry the bytes, over STRATA_KVIB_TCP_STREAMS parallel connections.
//   auto - rdma, else tcp (said once on stderr).
// A write is published atomically: a file is written under its .tmp name and rename() commits it under the final one,
// replacing what that name held.  Nothing is persistent: a daemon that restarts holds nothing, and the engine reads
// those prompts again (the folder is a computation cache, never the chat's record).
//
// One user thread a folder (ConversationDisk is single-threaded), one open stream at a time.
#pragma once

#include "strata/core/conversation_folder.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core::kvib {

inline constexpr const char* kScheme = "kvib://";
inline constexpr uint32_t kProtocol = 1;

/// The client's settings, each with its default (STRATA_KVIB_*).
struct Settings {
    std::string transport = "auto";        ///< STRATA_KVIB_TRANSPORT: auto | rdma | tcp
    int rdma_port_offset = 1;              ///< STRATA_KVIB_RDMA_PORT_OFFSET: the daemon's rdma_cm port = control + this
    size_t stage_mib = 16;                 ///< STRATA_KVIB_STAGE_MIB: one staging buffer
    size_t stage_depth = 4;                ///< STRATA_KVIB_STAGE_DEPTH: staging buffers in flight
    int tcp_streams = 4;                   ///< STRATA_KVIB_TCP_STREAMS: parallel data connections a daemon (tcp)
    int timeout_ms = 10000;                ///< STRATA_KVIB_TIMEOUT_MS: a control or data operation
    int reconnect_ms = 5000;               ///< STRATA_KVIB_RECONNECT_MS: a lost daemon is dialled again at most this often
    /// STRATA_KVIB_RECONNECT_MAX_MS: each failed dial doubles the wait before the next, up to this (a host that stays
    /// down costs the caller one connect timeout this often, not every reconnect_ms)
    int reconnect_max_ms = 300000;
    int connect_timeout_ms = 2000;         ///< STRATA_KVIB_CONNECT_TIMEOUT_MS: a dial (TCP connect, rdma_cm steps)
    std::string libibverbs = "libibverbs.so.1";   ///< STRATA_KVIB_LIBIBVERBS
    std::string librdmacm = "librdmacm.so.1";     ///< STRATA_KVIB_LIBRDMACM
};
/// The environment over the defaults; a bad value is refused by name (false, error).
bool settings_from_env(Settings& s, std::string& error);

struct ObjectInfo {
    std::string name;
    uint64_t bytes = 0;
    uint64_t age_ns = 0;   ///< since last written or touched, by the daemon's clock
};

class Server;

/// A folder over one or more daemons: ConversationFolder plus what the bench and tests need.
class Folder : public ConversationFolder {
public:
    /// Which data transport the connection settled on ("rdma" or "tcp"; "local" in process), after prepare().
    virtual std::string transport() const = 0;
    /// One synchronous read of `n` bytes at `offset` of a committed object (a block fetch through the data path).
    virtual bool read_at(const std::string& name, uint64_t offset, void* dst, size_t n, std::string& error) = 0;
    /// The daemons' arena: bytes used and the cap, summed.
    virtual bool usage(uint64_t& used, uint64_t& cap, std::string& error) = 0;
};

/// `hosts` = HOST:PORT[,HOST:PORT...] (what follows kvib://).  Settings from the environment.
std::unique_ptr<Folder> folder(const std::string& hosts);
std::unique_ptr<Folder> folder(const std::string& hosts, const Settings& s);
/// A folder over a server in this process: the same wire and store, no sockets (the tests' fake transport).
std::unique_ptr<Folder> folder_local(Server& server, const Settings& s);

struct ServerOptions {
    std::vector<std::string> listen = {"127.0.0.1:7471"};   ///< control endpoints (IP:PORT); RDMA on port + offset
    uint64_t arena_bytes = uint64_t{64} << 30;             ///< --arena-gib (64)
    uint64_t extent_bytes = uint64_t{64} << 20;            ///< --extent-mib (64)
    /// --prealloc-gib (0): extents mapped and registered at the start.  Freed extents always stay mapped and registered
    /// for the next object (registration pins and zeroes the pages: the slow part of a first write), so the daemon
    /// holds its high-water mark until it stops; this only moves that cost to the start.
    uint64_t prealloc_bytes = 0;
    int rdma_port_offset = 1;                              ///< --rdma-port-offset
    std::string transport = "auto";                        ///< --transport auto | rdma | tcp
    std::string libibverbs = "libibverbs.so.1";
    std::string librdmacm = "librdmacm.so.1";
    bool verbose = false;
};

struct ServerStats {
    uint64_t used = 0;     ///< bytes in objects and open writes
    uint64_t mapped = 0;   ///< bytes mapped (and registered): used plus the free extents kept for reuse
    uint64_t cap = 0, objects = 0, handles = 0, rdma_devices = 0;
};

class Server {
public:
    explicit Server(ServerOptions options);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    /// Opens the RDMA devices (transport auto/rdma) and, when `listen` is true, the control sockets and the
    /// rdma_cm listeners with their threads.  A server for folder_local() starts with listen = false.
    bool start(bool listen, std::string& error);
    void stop();
    ServerStats stats() const;
    struct Impl;
    Impl& impl() { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core::kvib
