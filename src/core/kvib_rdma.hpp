// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// kvib's RDMA side (kvib.hpp): libibverbs and librdmacm through dlopen, so nothing links them.  Built when their headers
// are found (rdma-core's -dev packages, or STRATA_KVIB_RDMA_INCLUDE); else every call says RDMA is not in this build.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core::kvib::rdma {

/// Whether this build has the RDMA transport at all (the headers were there).
bool built();

struct Api;
/// The two libraries opened once per process (later calls return the first result).  nullptr: error says why.
const Api* load(const std::string& libibverbs, const std::string& librdmacm, std::string& error);

/// The daemon's side: every RDMA device librdmacm uses, a protection domain each; extents registered on all of them;
/// rdma_cm listeners whose connections (client queue pairs) are accepted onto the device they arrived on.
class ServerSide {
public:
    ServerSide();
    ~ServerSide();
    bool open(const Api* api, std::string& error);
    int devices() const;
    std::string device_name(int dev) const;
    /// One region on every device: mrs / rkeys, one per device.
    bool reg(void* p, size_t n, std::vector<void*>& mrs, std::vector<uint32_t>& rkeys, std::string& error);
    void dereg(std::vector<void*>& mrs);
    /// `on_connect(token, dev)`: the session a connection's private data names joins device `dev`; false rejects it.
    bool listen(const std::vector<std::pair<std::string, int>>& endpoints,
                std::function<bool(uint64_t token, int dev)> on_connect, bool verbose, std::string& error);
    void stop();
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

/// The client's queue pair to one daemon, with its staging memory registered.
class ClientQp {
public:
    ClientQp();
    ~ClientQp();
    /// `outstanding`: the most work requests the caller keeps in flight (its staging buffers times the extents one
    /// buffer can touch); the send queue and the completion queue are sized for it, or the connect is refused.
    bool connect(const Api* api, const std::string& ip, int port, uint64_t token, void* stage, size_t stage_bytes,
                 size_t outstanding, int timeout_ms, std::string& error);
    /// One RDMA WRITE (`write`) or READ between `local` (inside the staging memory) and the daemon's `raddr`.
    bool post(uint64_t wr_id, bool write, void* local, size_t len, uint64_t raddr, uint32_t rkey, std::string& error);
    /// Completions: at least one (their wr_ids appended to `done`), or false on a failed one or the timeout.
    bool poll(std::vector<uint64_t>& done, int timeout_ms, std::string& error);
    std::string device() const;
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core::kvib::rdma
