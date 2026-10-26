// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "kvib_rdma.hpp"

#if !defined(_WIN32) && __has_include(<infiniband/verbs.h>) && __has_include(<rdma/rdma_cma.h>)
#define STRATA_KVIB_RDMA 1
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#endif

namespace strata::core::kvib::rdma {

#if defined(STRATA_KVIB_RDMA)

// Only the functions that are symbols: ibv_post_send and ibv_poll_cq are inline (through the context's ops), and
// the members avoid the ibv_ names, which newer verbs.h turns into macros (ibv_reg_mr).
struct Api {
    ibv_pd* (*alloc_pd)(ibv_context*);
    int (*dealloc_pd)(ibv_pd*);
    ibv_mr* (*reg_mr)(ibv_pd*, void*, size_t, int);
    int (*dereg_mr)(ibv_mr*);
    ibv_cq* (*create_cq)(ibv_context*, int, void*, ibv_comp_channel*, int);
    int (*destroy_cq)(ibv_cq*);
    int (*query_device)(ibv_context*, ibv_device_attr*);
    const char* (*wc_status_str)(ibv_wc_status);
    const char* (*get_device_name)(ibv_device*);
    rdma_event_channel* (*create_event_channel)();
    void (*destroy_event_channel)(rdma_event_channel*);
    int (*create_id)(rdma_event_channel*, rdma_cm_id**, void*, rdma_port_space);
    int (*destroy_id)(rdma_cm_id*);
    int (*bind_addr)(rdma_cm_id*, sockaddr*);
    int (*listen)(rdma_cm_id*, int);
    int (*resolve_addr)(rdma_cm_id*, sockaddr*, sockaddr*, int);
    int (*resolve_route)(rdma_cm_id*, int);
    int (*connect)(rdma_cm_id*, rdma_conn_param*);
    int (*accept)(rdma_cm_id*, rdma_conn_param*);
    int (*reject)(rdma_cm_id*, const void*, uint8_t);
    int (*disconnect)(rdma_cm_id*);
    int (*get_cm_event)(rdma_event_channel*, rdma_cm_event**);
    int (*ack_cm_event)(rdma_cm_event*);
    int (*create_qp)(rdma_cm_id*, ibv_pd*, ibv_qp_init_attr*);
    void (*destroy_qp)(rdma_cm_id*);
    ibv_context** (*get_devices)(int*);
    const char* (*event_str)(rdma_cm_event_type);
};

bool built() { return true; }

const Api* load(const std::string& libibverbs, const std::string& librdmacm, std::string& error) {
    static std::mutex m;
    static std::unique_ptr<Api> api;
    static std::string failed;
    std::lock_guard<std::mutex> lock(m);
    if (api) return api.get();
    if (!failed.empty()) {
        error = failed;
        return nullptr;
    }
    void* v = ::dlopen(libibverbs.c_str(), RTLD_NOW | RTLD_GLOBAL);
    void* c = v == nullptr ? nullptr : ::dlopen(librdmacm.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (v == nullptr || c == nullptr) {
        const char* e = ::dlerror();
        failed = std::string("cannot open ") + (v == nullptr ? libibverbs : librdmacm) + (e ? std::string(": ") + e : "");
        error = failed;
        return nullptr;
    }
    auto a = std::make_unique<Api>();
    bool ok = true;
    auto sym = [&](void* lib, const char* name, auto& fn) {
        void* p = ::dlsym(lib, name);
        if (p == nullptr && ok) {
            ok = false;
            failed = std::string("missing symbol ") + name;
        }
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(p);
    };
    sym(v, "ibv_alloc_pd", a->alloc_pd);
    sym(v, "ibv_dealloc_pd", a->dealloc_pd);
    sym(v, "ibv_reg_mr", a->reg_mr);
    sym(v, "ibv_dereg_mr", a->dereg_mr);
    sym(v, "ibv_create_cq", a->create_cq);
    sym(v, "ibv_destroy_cq", a->destroy_cq);
    sym(v, "ibv_query_device", a->query_device);
    sym(v, "ibv_wc_status_str", a->wc_status_str);
    sym(v, "ibv_get_device_name", a->get_device_name);
    sym(c, "rdma_create_event_channel", a->create_event_channel);
    sym(c, "rdma_destroy_event_channel", a->destroy_event_channel);
    sym(c, "rdma_create_id", a->create_id);
    sym(c, "rdma_destroy_id", a->destroy_id);
    sym(c, "rdma_bind_addr", a->bind_addr);
    sym(c, "rdma_listen", a->listen);
    sym(c, "rdma_resolve_addr", a->resolve_addr);
    sym(c, "rdma_resolve_route", a->resolve_route);
    sym(c, "rdma_connect", a->connect);
    sym(c, "rdma_accept", a->accept);
    sym(c, "rdma_reject", a->reject);
    sym(c, "rdma_disconnect", a->disconnect);
    sym(c, "rdma_get_cm_event", a->get_cm_event);
    sym(c, "rdma_ack_cm_event", a->ack_cm_event);
    sym(c, "rdma_create_qp", a->create_qp);
    sym(c, "rdma_destroy_qp", a->destroy_qp);
    sym(c, "rdma_get_devices", a->get_devices);
    sym(c, "rdma_event_str", a->event_str);
    if (!ok) {
        error = failed;
        return nullptr;
    }
    api = std::move(a);
    return api.get();
}

namespace {
constexpr int kAccess = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

bool sockaddr_of(const std::string& ip, int port, sockaddr_storage& out, std::string& error) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const int rc = ::getaddrinfo(ip.c_str(), std::to_string(port).c_str(), &hints, &res);
    if (rc != 0 || res == nullptr) {
        error = "cannot resolve " + ip + ": " + ::gai_strerror(rc);
        return false;
    }
    std::memcpy(&out, res->ai_addr, res->ai_addrlen);
    ::freeaddrinfo(res);
    return true;
}

// One event of `want` on `ch` within the timeout (the channel's fd is non-blocking).
bool wait_event(const Api* a, rdma_event_channel* ch, rdma_cm_event_type want, int timeout_ms, std::string& error) {
    pollfd p{ch->fd, POLLIN, 0};
    if (::poll(&p, 1, timeout_ms) <= 0) {
        error = std::string("rdma_cm: timed out waiting for ") + a->event_str(want);
        return false;
    }
    rdma_cm_event* ev = nullptr;
    if (a->get_cm_event(ch, &ev) != 0) {
        error = std::string("rdma_cm: no event: ") + std::strerror(errno);
        return false;
    }
    const rdma_cm_event_type got = ev->event;
    const int status = ev->status;
    a->ack_cm_event(ev);
    if (got != want) {
        error = std::string("rdma_cm: ") + a->event_str(got) + " (status " + std::to_string(status) + ") instead of " +
                a->event_str(want);
        return false;
    }
    return true;
}
} // namespace

struct ServerSide::Impl {
    const Api* api = nullptr;
    std::vector<ibv_context*> ctx;
    std::vector<ibv_pd*> pd;
    std::vector<int> rd_atom;
    rdma_event_channel* ch = nullptr;
    std::vector<rdma_cm_id*> listeners;
    std::thread th;
    std::atomic<bool> stopping{false};
    std::mutex m;
    struct Conn { rdma_cm_id* id; ibv_cq* cq; };
    std::vector<Conn> conns;
    std::function<bool(uint64_t, int)> on_connect;
    bool verbose = false;

    int dev_of(ibv_context* c) const {
        for (size_t i = 0; i < ctx.size(); ++i) if (ctx[i] == c) return (int) i;
        return -1;
    }
    void drop(rdma_cm_id* id) {
        std::lock_guard<std::mutex> lock(m);
        for (size_t i = 0; i < conns.size(); ++i)
            if (conns[i].id == id) {
                api->destroy_qp(id);
                api->destroy_cq(conns[i].cq);
                api->destroy_id(id);
                conns.erase(conns.begin() + (std::ptrdiff_t) i);
                return;
            }
    }
    void loop() {
        while (!stopping) {
            pollfd p{ch->fd, POLLIN, 0};
            if (::poll(&p, 1, 200) <= 0) continue;
            rdma_cm_event* ev = nullptr;
            if (api->get_cm_event(ch, &ev) != 0) continue;
            rdma_cm_id* id = ev->id;
            const rdma_cm_event_type type = ev->event;
            if (type == RDMA_CM_EVENT_CONNECT_REQUEST) {
                uint64_t token = 0;
                const rdma_conn_param& rq = ev->param.conn;
                if (rq.private_data != nullptr && rq.private_data_len >= sizeof token)
                    std::memcpy(&token, rq.private_data, sizeof token);
                const int dev = dev_of(id->verbs);
                const uint8_t want_rd = rq.initiator_depth, want_init = rq.responder_resources;
                api->ack_cm_event(ev);
                accept(id, dev, token, want_rd, want_init);
                continue;
            }
            api->ack_cm_event(ev);
            if (type == RDMA_CM_EVENT_DISCONNECTED) {
                if (verbose) std::fprintf(stderr, "kvib-store: rdma connection closed\n");
                drop(id);
            }
        }
    }
    void accept(rdma_cm_id* id, int dev, uint64_t token, uint8_t want_rd, uint8_t want_init) {
        if (dev < 0 || !on_connect || !on_connect(token, dev)) {
            std::fprintf(stderr, "kvib-store: rdma connection refused (device %d, unknown session)\n", dev);
            api->reject(id, nullptr, 0);
            api->destroy_id(id);
            return;
        }
        ibv_cq* cq = api->create_cq(id->verbs, 16, nullptr, nullptr, 0);
        ibv_qp_init_attr qa{};
        qa.send_cq = qa.recv_cq = cq;
        qa.qp_type = IBV_QPT_RC;
        qa.cap.max_send_wr = 4;
        qa.cap.max_recv_wr = 4;
        qa.cap.max_send_sge = qa.cap.max_recv_sge = 1;
        if (cq == nullptr || api->create_qp(id, pd[(size_t) dev], &qa) != 0) {
            std::fprintf(stderr, "kvib-store: rdma: cannot create a queue pair: %s\n", std::strerror(errno));
            if (cq != nullptr) api->destroy_cq(cq);
            api->reject(id, nullptr, 0);
            api->destroy_id(id);
            return;
        }
        rdma_conn_param cp{};
        cp.responder_resources = (uint8_t) std::min<int>(want_rd, rd_atom[(size_t) dev]);
        cp.initiator_depth = (uint8_t) std::min<int>(want_init, rd_atom[(size_t) dev]);
        cp.rnr_retry_count = 7;
        if (api->accept(id, &cp) != 0) {
            std::fprintf(stderr, "kvib-store: rdma: accept failed: %s\n", std::strerror(errno));
            api->destroy_qp(id);
            api->destroy_cq(cq);
            api->destroy_id(id);
            return;
        }
        std::lock_guard<std::mutex> lock(m);
        conns.push_back({id, cq});
        if (verbose)
            std::fprintf(stderr, "kvib-store: rdma connection on %s (responder resources %d)\n",
                         api->get_device_name(id->verbs->device), cp.responder_resources);
    }
};

ServerSide::ServerSide() : impl_(std::make_unique<Impl>()) {}
ServerSide::~ServerSide() { stop(); }

bool ServerSide::open(const Api* api, std::string& error) {
    Impl& s = *impl_;
    s.api = api;
    int n = 0;
    ibv_context** list = api->get_devices(&n);
    if (list == nullptr || n <= 0) {
        error = "no RDMA device";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        ibv_pd* pd = api->alloc_pd(list[i]);
        ibv_device_attr at{};
        if (pd == nullptr || api->query_device(list[i], &at) != 0) continue;
        s.ctx.push_back(list[i]);
        s.pd.push_back(pd);
        s.rd_atom.push_back(std::max(1, std::min(at.max_qp_rd_atom, at.max_qp_init_rd_atom)));
    }
    // rdma_free_devices would release the list only; the contexts stay librdmacm's for the process
    if (s.ctx.empty()) {
        error = "no usable RDMA device (protection domain)";
        return false;
    }
    return true;
}

int ServerSide::devices() const { return (int) impl_->ctx.size(); }
std::string ServerSide::device_name(int dev) const { return impl_->api->get_device_name(impl_->ctx[(size_t) dev]->device); }

bool ServerSide::reg(void* p, size_t n, std::vector<void*>& mrs, std::vector<uint32_t>& rkeys, std::string& error) {
    Impl& s = *impl_;
    mrs.clear();
    rkeys.clear();
    for (ibv_pd* pd : s.pd) {
        ibv_mr* mr = s.api->reg_mr(pd, p, n, kAccess);
        if (mr == nullptr) {
            error = std::string("ibv_reg_mr: ") + std::strerror(errno) + " (memlock limit?)";
            dereg(mrs);
            return false;
        }
        mrs.push_back(mr);
        rkeys.push_back(mr->rkey);
    }
    return true;
}

void ServerSide::dereg(std::vector<void*>& mrs) {
    for (void* mr : mrs) impl_->api->dereg_mr(static_cast<ibv_mr*>(mr));
    mrs.clear();
}

bool ServerSide::listen(const std::vector<std::pair<std::string, int>>& endpoints,
                        std::function<bool(uint64_t, int)> on_connect, bool verbose, std::string& error) {
    Impl& s = *impl_;
    s.on_connect = std::move(on_connect);
    s.verbose = verbose;
    s.ch = s.api->create_event_channel();
    if (s.ch == nullptr) {
        error = std::string("rdma_cm: no event channel: ") + std::strerror(errno);
        return false;
    }
    ::fcntl(s.ch->fd, F_SETFL, ::fcntl(s.ch->fd, F_GETFL) | O_NONBLOCK);
    for (const auto& [ip, port] : endpoints) {
        sockaddr_storage sa{};
        rdma_cm_id* id = nullptr;
        if (!sockaddr_of(ip, port, sa, error)) return false;
        if (s.api->create_id(s.ch, &id, nullptr, RDMA_PS_TCP) != 0 || s.api->bind_addr(id, (sockaddr*) &sa) != 0 ||
            s.api->listen(id, 16) != 0) {
            error = "rdma_cm: cannot listen on " + ip + ":" + std::to_string(port) + ": " + std::strerror(errno);
            if (id != nullptr) s.api->destroy_id(id);
            return false;
        }
        s.listeners.push_back(id);
    }
    s.th = std::thread([&s] { s.loop(); });
    return true;
}

void ServerSide::stop() {
    Impl& s = *impl_;
    s.stopping = true;
    if (s.th.joinable()) s.th.join();
    if (s.api == nullptr) return;
    for (auto& c : s.conns) {
        s.api->disconnect(c.id);
        s.api->destroy_qp(c.id);
        s.api->destroy_cq(c.cq);
        s.api->destroy_id(c.id);
    }
    s.conns.clear();
    for (rdma_cm_id* id : s.listeners) s.api->destroy_id(id);
    s.listeners.clear();
    if (s.ch != nullptr) s.api->destroy_event_channel(s.ch);
    s.ch = nullptr;
    for (ibv_pd* pd : s.pd) s.api->dealloc_pd(pd);
    s.pd.clear();
    s.ctx.clear();
}

struct ClientQp::Impl {
    const Api* api = nullptr;
    rdma_event_channel* ch = nullptr;
    rdma_cm_id* id = nullptr;
    ibv_pd* pd = nullptr;
    ibv_cq* cq = nullptr;
    ibv_mr* mr = nullptr;
    bool qp = false, connected = false;
    ~Impl() {
        if (api == nullptr) return;
        if (connected) api->disconnect(id);
        if (qp) api->destroy_qp(id);
        if (mr != nullptr) api->dereg_mr(mr);
        if (cq != nullptr) api->destroy_cq(cq);
        if (pd != nullptr) api->dealloc_pd(pd);
        if (id != nullptr) api->destroy_id(id);
        if (ch != nullptr) api->destroy_event_channel(ch);
    }
};

ClientQp::ClientQp() : impl_(std::make_unique<Impl>()) {}
ClientQp::~ClientQp() = default;

bool ClientQp::connect(const Api* api, const std::string& ip, int port, uint64_t token, void* stage,
                       size_t stage_bytes, size_t outstanding, int timeout_ms, std::string& error) {
    Impl& c = *impl_;
    c.api = api;
    sockaddr_storage sa{};
    if (!sockaddr_of(ip, port, sa, error)) return false;
    c.ch = api->create_event_channel();
    if (c.ch == nullptr || api->create_id(c.ch, &c.id, nullptr, RDMA_PS_TCP) != 0) {
        error = std::string("rdma_cm: ") + std::strerror(errno);
        return false;
    }
    ::fcntl(c.ch->fd, F_SETFL, ::fcntl(c.ch->fd, F_GETFL) | O_NONBLOCK);
    if (api->resolve_addr(c.id, nullptr, (sockaddr*) &sa, timeout_ms) != 0 ||
        !wait_event(api, c.ch, RDMA_CM_EVENT_ADDR_RESOLVED, timeout_ms, error))
        return error.empty() ? (error = "rdma_cm: cannot resolve " + ip, false) : false;
    if (api->resolve_route(c.id, timeout_ms) != 0 ||
        !wait_event(api, c.ch, RDMA_CM_EVENT_ROUTE_RESOLVED, timeout_ms, error))
        return error.empty() ? (error = "rdma_cm: no route to " + ip, false) : false;
    ibv_device_attr at{};
    if (api->query_device(c.id->verbs, &at) != 0) {
        error = std::string("rdma: cannot query the device: ") + std::strerror(errno);
        return false;
    }
    // every work request is signalled (sq_sig_all), so the completion queue holds as many as the send queue
    const size_t want = std::max<size_t>(outstanding, 16);
    if (want > (size_t) at.max_qp_wr || want > (size_t) at.max_cqe) {
        error = "rdma: the staging ring keeps " + std::to_string(want) + " work requests in flight, the device allows " +
                std::to_string(std::min(at.max_qp_wr, at.max_cqe)) +
                " (lower STRATA_KVIB_STAGE_DEPTH or STRATA_KVIB_STAGE_MIB)";
        return false;
    }
    c.pd = api->alloc_pd(c.id->verbs);
    c.cq = c.pd == nullptr ? nullptr : api->create_cq(c.id->verbs, (int) want, nullptr, nullptr, 0);
    c.mr = c.cq == nullptr ? nullptr : api->reg_mr(c.pd, stage, stage_bytes, kAccess);
    if (c.mr == nullptr) {
        error = std::string("rdma: cannot set up the staging memory: ") + std::strerror(errno) + " (memlock limit?)";
        return false;
    }
    ibv_qp_init_attr qa{};
    qa.send_cq = qa.recv_cq = c.cq;
    qa.qp_type = IBV_QPT_RC;
    qa.sq_sig_all = 1;
    qa.cap.max_send_wr = (uint32_t) want;
    qa.cap.max_recv_wr = 1;
    qa.cap.max_send_sge = qa.cap.max_recv_sge = 1;
    if (api->create_qp(c.id, c.pd, &qa) != 0) {
        error = std::string("rdma: cannot create a queue pair: ") + std::strerror(errno);
        return false;
    }
    c.qp = true;
    rdma_conn_param cp{};
    cp.private_data = &token;
    cp.private_data_len = sizeof token;
    cp.initiator_depth = (uint8_t) std::max(1, std::min(at.max_qp_init_rd_atom, 255));
    cp.responder_resources = (uint8_t) std::max(1, std::min(at.max_qp_rd_atom, 255));
    cp.retry_count = 7;
    cp.rnr_retry_count = 7;
    if (api->connect(c.id, &cp) != 0 || !wait_event(api, c.ch, RDMA_CM_EVENT_ESTABLISHED, timeout_ms, error))
        return error.empty() ? (error = "rdma_cm: cannot connect to " + ip, false) : false;
    c.connected = true;
    return true;
}

bool ClientQp::post(uint64_t wr_id, bool write, void* local, size_t len, uint64_t raddr, uint32_t rkey,
                    std::string& error) {
    Impl& c = *impl_;
    ibv_sge sge{};
    sge.addr = (uint64_t) (uintptr_t) local;
    sge.length = (uint32_t) len;
    sge.lkey = c.mr->lkey;
    ibv_send_wr wr{}, *bad = nullptr;
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = write ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey = rkey;
    const int rc = ibv_post_send(c.id->qp, &wr, &bad);
    if (rc != 0) {
        error = std::string("ibv_post_send: ") + std::strerror(rc);
        return false;
    }
    return true;
}

bool ClientQp::poll(std::vector<uint64_t>& done, int timeout_ms, std::string& error) {
    Impl& c = *impl_;
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    ibv_wc wc[16];
    for (;;) {
        const int n = ibv_poll_cq(c.cq, 16, wc);
        if (n < 0) {
            error = "ibv_poll_cq failed";
            return false;
        }
        for (int i = 0; i < n; ++i) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                error = std::string("rdma: work request failed: ") + c.api->wc_status_str(wc[i].status);
                return false;
            }
            done.push_back(wc[i].wr_id);
        }
        if (n > 0) return true;
        if (std::chrono::steady_clock::now() > until) {
            error = "rdma: timed out waiting for a completion";
            return false;
        }
    }
}

std::string ClientQp::device() const { return impl_->api->get_device_name(impl_->id->verbs->device); }

#else  // no RDMA headers in this build

struct Api {};
bool built() { return false; }
const Api* load(const std::string&, const std::string&, std::string& error) {
    error = "this build has no RDMA transport (rdma-core headers were missing)";
    return nullptr;
}
struct ServerSide::Impl {};
ServerSide::ServerSide() : impl_(std::make_unique<Impl>()) {}
ServerSide::~ServerSide() = default;
bool ServerSide::open(const Api*, std::string& error) { return (void) load("", "", error), false; }
int ServerSide::devices() const { return 0; }
std::string ServerSide::device_name(int) const { return {}; }
bool ServerSide::reg(void*, size_t, std::vector<void*>&, std::vector<uint32_t>&, std::string& error) {
    return (void) load("", "", error), false;
}
void ServerSide::dereg(std::vector<void*>&) {}
bool ServerSide::listen(const std::vector<std::pair<std::string, int>>&, std::function<bool(uint64_t, int)>, bool,
                        std::string& error) {
    return (void) load("", "", error), false;
}
void ServerSide::stop() {}
struct ClientQp::Impl {};
ClientQp::ClientQp() : impl_(std::make_unique<Impl>()) {}
ClientQp::~ClientQp() = default;
bool ClientQp::connect(const Api*, const std::string&, int, uint64_t, void*, size_t, size_t, int,
                       std::string& error) {
    return (void) load("", "", error), false;
}
bool ClientQp::post(uint64_t, bool, void*, size_t, uint64_t, uint32_t, std::string& error) {
    return (void) load("", "", error), false;
}
bool ClientQp::poll(std::vector<uint64_t>&, int, std::string& error) { return (void) load("", "", error), false; }
std::string ClientQp::device() const { return {}; }

#endif

} // namespace strata::core::kvib::rdma
