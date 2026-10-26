// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host/model-free controls: the production ownership primitive with deterministic implementations of gpu.hpp.
#include "strata/core/batch_outbound.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <unordered_map>

void check(bool ok) { if (!ok) throw std::runtime_error("batch outbound causal control failed"); }
namespace fake {
struct Job {
    std::function<void()> fn;
    std::vector<std::shared_ptr<Job>> deps;
    bool done = false, failed = false;
    bool compute = false, copy = false;
    int ticks = 1;
};
struct Queue { std::deque<std::shared_ptr<Job>> jobs; };
std::vector<Queue*> queues;
std::unordered_map<void*, size_t> allocations;
bool refuse_record = false, refuse_submit = false, fail_copy = false, fail_wait = false;
int copies = 0, drains = 0, overlap = 0;
int copy_ticks = 1;
std::shared_ptr<Job> submit(Queue* q, std::function<void()> fn, std::shared_ptr<Job> after = {}) {
    auto j = std::make_shared<Job>();
    j->fn = std::move(fn);
    if (!q->jobs.empty()) j->deps.push_back(q->jobs.back());
    if (after) j->deps.push_back(std::move(after));
    q->jobs.push_back(j);
    return j;
}
bool step(Queue* q) {
    for (auto& j : q->jobs) {
        if (j->done) continue;
        for (auto& dep : j->deps) if (!dep->done) return false;
        for (auto& dep : j->deps) j->failed |= dep->failed;
        if (j->compute)
            for (auto* other : queues) if (other != q)
                for (auto& copy : other->jobs) if (copy->copy && !copy->done) ++overlap;
        if (--j->ticks > 0) return true;
        if (!j->failed) j->fn();
        j->done = true;
        return true;
    }
    return false;
}
void settle() {
    bool progress;
    do { progress = false; for (auto* q : queues) progress |= step(q); } while (progress);
}
bool pending(Queue* q) {
    return std::any_of(q->jobs.begin(), q->jobs.end(), [](auto& j) { return !j->done; });
}
}  // namespace fake

namespace strata::gpu {
struct Event { std::shared_ptr<fake::Job> job; };
const char* last_error() { return "injected gpu event/copy failure"; }
Stream stream_create_copy() {
    auto* q = new fake::Queue;
    fake::queues.push_back(q);
    return q;
}
bool stream_sync(Stream s) {
    ++fake::drains;
    fake::settle();
    check(!fake::pending(static_cast<fake::Queue*>(s)));
    return true;
}
void stream_destroy(Stream s) {
    auto* q = static_cast<fake::Queue*>(s);
    check(!fake::pending(q));
    fake::queues.erase(std::find(fake::queues.begin(), fake::queues.end(), q));
    delete q;
}
void* alloc_device(size_t bytes) {
    auto* p = new unsigned char[bytes];
    fake::allocations[p] = bytes;
    return p;
}
void free(const void* p) {
    // Destroying a buffer before the outbound stream drains is a control failure, even if a runtime free would wait.
    for (auto* q : fake::queues) check(!fake::pending(q));
    check(fake::allocations.erase(const_cast<void*>(p)) == 1);
    delete[] static_cast<const unsigned char*>(p);
}
bool event_create(Event** e, bool) { *e = new Event; return true; }
void event_destroy(Event* e) {
    if (e->job) check(e->job->done);   // event must outlive its submitted use, including a failed transfer
    delete e;
}
bool event_record(Event* e, Stream s) {
    if (fake::refuse_record) return false;
    e->job = fake::submit(static_cast<fake::Queue*>(s), [] {});
    return true;
}
bool copy_async_after(void* dst, const void* src, size_t bytes, Stream s, const Event* after, Event* done) {
    const bool failed = fake::fail_copy;
    auto j = fake::submit(static_cast<fake::Queue*>(s), [=] {
        if (!failed) { std::memcpy(dst, src, bytes); ++fake::copies; }
    }, after ? after->job : nullptr);
    j->copy = true;
    j->ticks = fake::copy_ticks;
    if (failed) j->failed = true;
    // A submission can have side effects before reporting failure: teardown must still drain it.
    if (fake::refuse_submit) return false;
    done->job = j;
    return true;
}
Status event_status(const Event* e) {
    if (!e->job || !e->job->done) return Status::Pending;
    return e->job->failed ? Status::Failed : Status::Done;
}
bool event_sync(const Event* e) {
    while (event_status(e) == Status::Pending) {
        bool progress = false;
        for (auto* q : fake::queues) progress |= fake::step(q);
        check(progress);
    }
    return !fake::fail_wait && event_status(e) == Status::Done;
}
}  // namespace strata::gpu

using strata::core::BatchOutbound;
namespace gpu = strata::gpu;
void ownership_controls() {
    auto* producer = static_cast<fake::Queue*>(gpu::stream_create_copy());
    float host[12];
    std::fill(host, host + 12, -1);
    {
        BatchOutbound out;
        check(out.init(host, 6, 2));
        check(!out.writable(-1, 1) && !out.writable(0, 0) && !out.writable(5, 2));
        check(out.poll(0) == -1 && !out.acknowledge(0));
        auto* packed = out.device_rows();
        // The ready event really is a producer dependency, not a decorative event recorded on the copy stream.
        fake::submit(producer, [=] { packed[0] = 10; packed[1] = 11; });
        check(out.start(0, 1, producer));
        check(out.poll(0) == 0 && !out.acknowledge(0));
        for (auto* q : fake::queues) if (q != producer) check(!fake::step(q));
        check(host[0] == -1 && !out.writable(0, 1) && !out.start(0, 2, producer));
        check(fake::step(producer));   // pack only; the ready event is still pending
        check(out.poll(0) == 0);
        check(fake::step(producer));   // ready
        fake::settle();
        check(out.poll(0) == 1 && host[0] == 10 && host[1] == 11);
        check(!out.writable(0, 1));   // completion alone is not receiver acknowledgement
        check(out.acknowledge(0) && out.writable(0, 1) && !out.acknowledge(0));
        // Several group/ring wraps, with independently owned rows and ordered acknowledgements.
        for (int turn = 0; turn < 9; ++turn) {
            for (int base : {0, 2, 4}) {
                packed[base * 2] = (float) (100 * turn + base);
                packed[base * 2 + 1] = (float) (100 * turn + base + 1);
                check(out.start(base, 1, producer));
            }
            fake::settle();
            for (int base : {0, 2, 4}) {
                check(out.poll(base) == 1 && host[base * 2] == 100 * turn + base);
                check(out.count(base) == 1 && out.acknowledge(base));
            }
        }
        // The synchronous batch-graph consumer uses the very same start/wait/ack ownership path.
        packed[2] = 123;
        check(out.start(1, 1, producer) && out.wait(1));
        check(host[2] == 123 && out.poll(1) == 1 && out.acknowledge(1));
        // Stop/cancel with an incomplete copy: destructor must drain, even without polling or acknowledgement.
        packed[10] = 456;
        check(out.start(5, 1, producer));
    }
    check(host[10] == 456 && fake::allocations.empty());
    for (int fault = 0; fault < 4; ++fault) {
        BatchOutbound out;
        check(out.init(host, 6, 2));
        fake::refuse_record = fault == 0;
        fake::refuse_submit = fault == 1;
        fake::fail_copy = fault == 2;
        fake::fail_wait = fault == 3;
        const bool submitted = out.start(0, 2, producer);
        check(submitted == (fault >= 2));
        fake::settle();
        check(out.poll(0) == -1 && !out.acknowledge(0) && !out.writable(1, 1));
        fake::refuse_record = fake::refuse_submit = fake::fail_copy = fake::fail_wait = false;
    }
    check(fake::allocations.empty());
    gpu::stream_destroy(producer);
    check(fake::queues.empty());
    std::printf("ownership: dependency, incomplete, failed, acknowledgement, wrap and stop controls passed\n");
}
void timeout_controls() {
    auto* producer = static_cast<fake::Queue*>(gpu::stream_create_copy());
    float host[2] = {-1, -1};
    {
        BatchOutbound out;
        check(out.init(host, 2, 1));
        out.device_rows()[0] = 42;
        const auto before = BatchOutbound::Clock::now();
        check(out.start(0, 1, producer));
        check(out.poll(0, before + BatchOutbound::timeout()) == 0);
        const auto late = BatchOutbound::Clock::now() + BatchOutbound::timeout() + std::chrono::seconds(1);
        check(out.poll(0, late) == -2 && host[0] == -1);
        check(!out.writable(0, 1) && !out.acknowledge(0) && !out.wait(0));
        fake::settle();   // late completion must never turn a timed-out transfer into delivery
        check(out.poll(0) == -2 && !out.acknowledge(0));
    }
    gpu::stream_destroy(producer);
    check(fake::allocations.empty() && fake::queues.empty());
    std::printf("timeout=%lld late delivery refused, ownership retained, teardown drained\n",
                (long long) BatchOutbound::timeout().count());
}
int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "timeout") == 0) { timeout_controls(); return 0; }
    if (argc > 1) { std::printf("%d\n", BatchOutbound::enabled() ? 1 : 0); return 0; }
    ownership_controls();
    return 0;
}
