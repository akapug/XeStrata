# KV over the 56G triangle (homelab lever): design

Labels: MEASURED = run or read during the probe;
INFERRED = derived, not run.

## 1. Probe (2026-10-25)

### Boxes and links (MEASURED)

| box | RAM total / available | RDMA device | triangle ports |
|---|---|---|---|
| odie | 121 GiB / 80 GiB (model loaded) | `rocep3s0` | p1 to droopy, p2 to lassie |
| droopy | 429 GiB / 401 GiB (+ Optane PMem, not used here) | `rocep153s0` | one to odie, one to lassie |
| lassie | 1007 GiB / 981 GiB | `rocep19s0` | one to odie, one to droopy |
| snoopy | 182 GiB | none (10G only) | - |

All six ports are ACTIVE, 56 Gb/s 4X FDR, link layer **Ethernet** (the triangle runs 56GbE RoCE v1). The user memlock limit is 8 MiB on odie and droopy: RDMA needs root or a raised limit.

### RDMA READ, odie client (perftest `-R`, rdma_cm, 1 QP) (MEASURED)

| size | read BW odie<-droopy | read BW odie<-lassie | read latency typical droopy / lassie |
|---|---|---|---|
| 2 B | - | - | 1.74 / 2.09 us |
| 4 KiB | 5,950 MB/s | 5,803 MB/s | 3.67 / 4.08 us |
| 64 KiB | 5,992 MB/s | 5,113 MB/s | 13.4 / 14.9 us |
| 1 MiB | 5,782 MB/s | 5,308 MB/s | 176.6 / 176.5 us |
| 8 MiB | 5,780 MB/s | 5,629 MB/s | 1,389 / 1,389 us |

A single queue pair is at line rate (~48 Gb/s of the 52.6 measured for writes). TCP on the same links: 19.9-22.6 Gb/s with one stream, ~50 Gb/s with 8 streams. odie's ConnectX-3 sits in a PCIe
3.0 x8 slot shared by both ports, so droopy and lassie together cannot give odie much more than one link
(INFERRED: ~6.5 GB/s ceiling).

### What a resolve miss is today (MEASURED from the code)

- A block (page) is 4 cells x 2 KV heads: **3,264 B in K8V4** (816 B a cell), 4,224 B in INT8 (`kv_block_bytes`).
- A query's selection is `idx_top_k` 2048 cells + 3, so up to **513 blocks per query per QSA layer**.
- 12 QSA layers (INFERRED: 2.414 GiB a slot at 262k = 12 x 262,144 x 816 B).
- The miss copy is a **device kernel** that reads the host copy zero-copy over PCIe, inside the captured
  graph; which blocks miss is decided on the device by the resolve (`kv_stream_resolve_slots`). The host never
  sees a miss list on the hot path.
- Hit rates recorded in earlier homelab runs (`KV streaming: ... hit VRAM`, cumulative, early in a run): 61-81 %,
  so 19-39 % of selected blocks miss. Tonight's matrix logs end before that summary line.

### Is a live remote tier viable? (INFERRED from the numbers above)

Bytes a decode step would fetch remotely if every miss went to droopy/lassie (one token a row, `--no-batch-mtp`):
per row 12 x 513 x 3,264 B = 20.1 MB selected; at 20 % misses 4.0 MB, at 39 % 7.8 MB.

| rows | miss 20 % | miss 39 % | at 5.8 GB/s (20 % / 39 %) | share of a ~128 ms step (n32/131k, 7.8 tok/s) |
|---|---|---|---|---|
| 32 | 129 MB | 251 MB | 22 ms / 43 ms | 17 % / 34 % |
| 64 | 257 MB | 502 MB | 44 ms / 87 ms | 34 % / 68 % (of the same step; a 64-row step is longer) |

Bandwidth alone already costs a sixth to two thirds of every step. The control path is the bigger obstacle:

1. The miss list exists only on the device, per QSA layer, in the middle of a captured graph. A remote block
   needs a host round trip (device -> host sync, RDMA READ, host -> device) **12 times a step**, which means
   giving up graph replay for the decode windows.
2. Greedy outputs must stay bit-identical, so a missing block can never be skipped or approximated: a step
   either has every selected block or waits.
3. The selection may name any block of the history at any step, so "cold" blocks are not reliably cold.
4. The local tier it replaces is PCIe 4.0 x16 zero-copy (~25 GB/s, INFERRED for odie's Threadripper), four
   times the link.

**Verdict: a live-streaming remote tier is not viable** without rewriting the decode path around host syncs,
and it would slow every step even then. The viable shape is the **parked tier**.

### Why the parked tier raises the agent count

The cap is batch slots, not conversations: every slot pins a full `--max-context` host copy (2.41 GiB at 262k),
whether its conversation is 2k or 262k tokens long. An agent fleet does not decode all agents at once: agents wait
on tool calls. With B slots (27-28 at 262k fit in odie's RAM) and 64 agents, the agents not holding a slot live
as **parked conversations**. The engine already parks the outgoing conversation of the main session on every
switch (`ConversationCache`, `--conversation-cache-mib`) and can spill parked images to a folder
(`ConversationDisk`, `--conversation-save`). Today the parked images sit in odie's RAM, the very resource that
caps the slots.

| quantity | value |
|---|---|
| parked snapshot, 14,796 tokens (earlier homelab log) | 420 MB, parked in 57-204 ms (MEASURED) |
| a 200k-token conversation | 1.96 GB K/V alone; up to 5.7 GB scaled linearly from 28.4 KB a token (INFERRED) |
| restore from remote RAM at 5.8 GB/s | 0.34-0.98 s (INFERRED) |
| re-reading that prompt instead | minutes (upstream measured 270k tokens in 96 s, code comment) |
| remote capacity | droopy 401 GiB + lassie 981 GiB available: hundreds of 200k conversations |

## 2. Design: the parked tier in droopy/lassie RAM

### Shape

`--conversation-save kvib://HOST:PORT[,HOST:PORT...]` puts the existing conversation folder in remote RAM.
Everything `ConversationDisk` does (fingerprint, header and body checksums, `--conversation-save-mib` budget,
expiry, checkpoint selection, LRU, validation before restore) stays the same code. Only the folder operations
change: list, open for read, open for write, rename, remove, touch. They now go through a small folder interface
(`include/strata/core/conversation_folder.hpp`: `ConversationFolder` and its `FolderStream`), with a local-folder
implementation (the old `FILE*` code) and the kvib one. The file format code writes and reads through a
`FolderStream`, so **the bytes written and read are the same `.xsc` bytes** as on disk. (The first draft used glibc
`fopencookie`; it was dropped because a cookie stream copies every read through the `FILE` buffer once more and does
not exist on Windows.) The hot path (kernels, graphs, resolve) is not touched, so greedy outputs cannot change: a restored
image is the same bytes the RAM cache would have restored, checked by the body hash and by
`conversation_snapshot_validate`.

Existence sweep (git grep `KvHostPools`, `kv_stream`, `conversation_cache`, `park`, `snapshot`,
`STRATA_PARK`): the park/snapshot machinery, the RAM cache and the disk folder exist and are tested
(`conversation_snapshot_test`, `conversation_cache_test`). No remote or network store exists. This extends the
folder tier rather than adding a parallel system.

### Pieces

1. **`strata-kvib-store`** (daemon, droopy and lassie): one TCP control socket; RAM objects made of extents
   (default 64 MiB), each `mmap`ed and registered for RDMA when allocated and released when its object goes; a hard
   cap on the arena (`--arena-gib`). Object writes are atomic: a write goes to a handle, and `commit` publishes it
   under its final name, replacing the old object (the folder's write-.tmp-then-rename). RAM only: a daemon restart
   empties it and the engine reads those prompts again (the folder is a computation cache, never the chat record).
2. **Engine client** (`include/strata/core/kvib.hpp`, `src/core/kvib.cpp`, `src/core/kvib_rdma.{hpp,cpp}`): control protocol, transports, placement over several
   daemons (a new object goes to the daemon with the most free arena; listings merge; a read goes to the holder).
3. **Transports**: RDMA (RC queue pair through rdma_cm, the client RDMA-WRITEs and RDMA-READs the daemon's
   extents through a registered staging ring, pipelined); TCP (the same control socket carries the bytes; N
   parallel streams); a local in-process store for tests. `libibverbs`/`librdmacm` are opened with `dlopen` only when
   the RDMA transport is used, so the engine binary gets no new runtime dependency and the setting off costs nothing.

### Transport choice (from the measurements)

| option | measured / expected | why not / why |
|---|---|---|
| ibverbs RDMA, daemon-registered regions | 5.8 GB/s on one QP, ~0 CPU (MEASURED perftest) | **chosen** for speed; needs the serving container to see `/dev/infiniband`, a memlock limit and the rdma-core libraries |
| TCP over the 56GbE triangle | 2.5-2.8 GB/s one stream, ~6 GB/s with 8 streams (MEASURED 10-24) | **fallback** (`auto` falls back to it): no image change, costs a few cores during a park |
| NVMe-oF RDMA to a RAM-disk target | same wire | rejected: buffered writes go through odie's page cache (the RAM we are short of), a filesystem on a remote RAM disk, root mounts in the container |
| rsocket (LD_PRELOAD) | TCP API over RDMA | rejected: an extra copy, process-wide preload, flaky on mlx4 |
| odie's own disk (`--conversation-save DIR`) | exists | flash wear at fleet park rates (GBs per park, TBs a day); keeps working as before |

### Settings (all with a named default; the tier is off unless `--conversation-save` names `kvib://`)

Engine (environment):

| setting | default | meaning |
|---|---|---|
| `STRATA_KVIB_TRANSPORT` | `auto` | `rdma`, `tcp` or `auto` (RDMA, else TCP, logged) |
| `STRATA_KVIB_RDMA_PORT_OFFSET` | 1 | the daemon's rdma_cm port = control port + this |
| `STRATA_KVIB_STAGE_MIB` | 16 | one staging buffer |
| `STRATA_KVIB_STAGE_DEPTH` | 4 | staging buffers in flight |
| `STRATA_KVIB_TCP_STREAMS` | 4 | parallel TCP data streams per daemon |
| `STRATA_KVIB_TIMEOUT_MS` | 10000 | control and data timeouts |
| `STRATA_KVIB_RECONNECT_MS` | 5000 | a lost daemon (restart, reboot) is dialled again at most this often; what it held is gone |
| `STRATA_KVIB_LIBIBVERBS` / `STRATA_KVIB_LIBRDMACM` | `libibverbs.so.1` / `librdmacm.so.1` | dlopen names |

Daemon (flags): `--listen IP:PORT` (default `127.0.0.1:7471`; repeat it for each link address), `--rdma-port-offset 1`, `--arena-gib 64`,
`--extent-mib 64`, `--prealloc-gib 0` (freed extents always stay registered for reuse; this registers them at the
start), `--transport auto|rdma|tcp`.

Measured transport and engine-path rates (droopy -> lassie, RDMA and TCP; lab notes, not in this tree) say the engine's
conversation-file path, not the wire, bounds a park and a restore.

### Slot pressure: what turns agents beyond the slots into parked conversations

Today (MEASURED from the code, `serve/server.py` and `src/program/generate.cpp`):

- `pick_slot` gives a request a free slot that holds its prompt's start, else an empty slot, else the least recently
  used idle slot. It never preempts a busy slot; with every slot busy a request waits on `slot_cv`.
- The request path parks the MAIN session's conversation (`park_current`) into the RAM conversation cache; what the
  cache evicts goes to `--conversation-save` (`save_evicted`, synchronous).
- GAP: the conversation an idle SLOT holds is never parked. When `pick_slot` hands that slot to another conversation,
  the request's `copy_to_slot` overwrites it (with `STRATA_PARK_SKIP_SLOT_HELD=1` it is then in no place at all), and
  its next turn reads its whole history again. Without closing this, a remote tier only moves storage.

Added (`STRATA_SLOT_PARK`, default 0; `include/strata/program/slot_park.hpp`):

- **Eviction trigger**: a `BGEN <slot>` whose prompt does not go on with the conversation the idle slot holds (all of
  it, or its deepest turn checkpoint), when `--conversation-save` is on and the conversation has at least
  `STRATA_SLOT_PARK_MIN_TOKENS` (2048) tokens. Window-row prefill declines such an admission so the request path
  handles it.
- **Park**: in the request path, after the windows are drained and before the source search loads anything, the
  slot's sessions (every stage of a layer split), its checkpoints and its draft K/V are captured
  (`conversation_snapshot_save` / `conversation_split_save`, RAM-admitted first) and written with
  `ConversationDisk::save` - to `kvib://` when so configured. The capture is freed once written.
- **Admission and restore**: the conversation's next turn is a `BGEN` to whatever slot `pick_slot` gives it. The
  request path's existing source search finds the file (`disk.best`), loads it (`disk.load`, RAM-admitted, checksum
  and validation), restores it into the main session and reads only the new delta, then `copy_to_slot` moves it into
  its slot.
- **What a client sees**: nothing while parked (it is in its tool call). On its next turn: the wait for a free slot
  as before, then the restore (load + validate + restore) instead of a full prompt read, then the delta.
- **Cost**: the park runs on the serve thread, so every slot's decoding waits for capture + write (INFERRED from the
  measured capture rate and the kvib transport numbers). A streamed save without the host capture
  (`conversation_snapshot_sources`, used today by SAVE) is the next step if it shows in the card runs.
  Built since (`STRATA_SLOT_PARK_ASYNC=1`, `conversation_park_queue.hpp`): the serve thread keeps only the capture; a
  writer thread with its own folder connection serialises and writes, and the next turn takes its own image back from
  the queue or mid-write. Measured in our lab notes (not in this tree).

### Costs and limits (INFERRED)

- A park that evicts to the folder writes synchronously today (`save_evicted` in the serve loop): 2 GB adds about
  0.35 s to that admission at 5.8 GB/s. An asynchronous writer is the next step if it shows in the card runs.
- A restore allocates the image in odie RAM for the copy (the folder's existing `load_bytes` admission).
- One link is enough; striping one object over droopy and lassie would gain little (odie's x8 slot).

### Tests

- Host (`conversation_kvib_test`): object semantics over the in-process store and loopback TCP (odd sizes, the header
  rewrite after a seek back, replace, abort, remove, a full arena, block reads, a client that dies mid-write), the
  `.xsc` bytes in a kvib store equal to the bytes in a folder, a damaged object caught by the checksum, a daemon that
  stops (a miss, never an engine error), restarts (empty) and is dialled again, bad settings refused by name.
- CPU device (`conversation_kvib_parity`): every K/V format, resident and streamed (mode 1, the host pools), captured,
  parked in the store (in-process and TCP), the slot overwritten, then loaded and restored: the state reads back byte
  for byte and verifies, a streamed layer's stale VRAM pages are invalidated. The attention reads only these bytes,
  so its outputs are the same; the end-to-end greedy check on the B70s is owed to a card window.
- `slot_park_test`: the settings, which admissions overwrite a slot's conversation, which slots are parked.
- Transport: the daemon on lassie, the client on droopy over link A (a point-to-point /30), RDMA and TCP: GB/s for
  conversation-sized objects, and p50/p99 per block read (3,264 B) through the code path.
