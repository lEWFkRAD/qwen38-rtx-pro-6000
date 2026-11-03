# Native Strata LSA integration audit

Date: 2026-11-03. Scope: static source inspection. No engine modification, build, GPU workload, production stop, restoration or promotion was performed by this audit.

Source inspected: `/path/to/lsa-prep-20261103/flashnext-maps/strata-src`, clean capture-only commit `3f83801ea1862652806fc6ebddab7f6cf57f3af6`. Its root `AGENTS.md` was read. It requires a new feature to be opt-in and the default path to stay byte-identical; measured claims need measurements. This local audit is not an upstream contribution or proof of runtime behavior.

## Recovered handoff and scope

The authoritative saved handoff is `/path/to/lsa-prep-20261103/checkpoints/flashnext-maps-20261103T141636Z/RESUME.md`. The working tree is `/path/to/lsa-prep-20261103/flashnext-maps`.

The previous session completed 12 learned query/key projection pairs, PCA comparisons, 600-step model checkpoints, temperature sidecars, 96 native activation captures, compatible prepared data, and a standalone CUDA smoke. The maps passed their declared 4–6K-token pilot gate. All 12 layers passed selection checks; layers 3, 27 and 47 passed gathered attention checks. None of this establishes native engine integration, 128K behavior, MTP equivalence, multiple-slot behavior, whole-model quality, speed superiority or promotion readiness.

The previous session explicitly deferred native integration until after work. The user's current request resumes that work. The capture plan and checkpoint refer to an older production configuration hash; they must not be used as the current stop/restore contract. Production maintenance belongs to the current recovery procedure and current configuration hashes.

## Map and algorithm contract

The learned map file is `flashnext-maps/maps/pilot-v1/flashnext-learned-384x64.pt`, SHA-256 `4d8a606e0ddd33858cb662e7f82f723dd54ca96776f3c383af28b30935a85d02`. Its paired JSON carries 12 temperatures and provenance. Layer IDs are 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47. There are 24 FP32 matrices of shape `[384,64]`, 589,824 parameters, 2.25 MiB payload.

`prepare_captures.py:55` defines the inputs precisely:

- K: actual INT8 cache codes multiplied in FP32 by their stored FP16 scales, then coordinates `[64:256]` of KV head 0 followed by the same coordinates of KV head 1.
- Q: post-norm/post-RoPE attention Q, coordinates `[64:256]`, sum the 12 contiguous Q heads belonging to each KV head, concatenate the two groups. The grouping is `query_head / 12`, not `query_head % 2`.
- Both input vectors are FP32. Apply the `[384,64]` matrix, then L2-normalize the projected row. Do not normalize the 384-dimensional input first.
- The real-map smoke retains FP32 projected keys during pooling/clustering; normalized queries and final normalized centroids are cast to BF16 for selection. Nearly tied scores need explicit margin/objective checks rather than assuming identical labels.
- Temperatures are separate from normalized maps. A uniform positive temperature does not change a layer's ranking, but preserve the sidecars and validate their positivity/finiteness; do not silently fold an undocumented scale into export.

The checked standalone algorithm is: cluster closed 512-token blocks; sum consecutive groups of four normalized keys into 128 initial Ward items, sizes four; merge to 64 buckets with maximum 16 members; repeat initial labels across each four-token group; sum original projected keys per resulting bucket, normalize, cast centroid to BF16; store token IDs with `-1` padding. Select up to 385 centroid buckets, expand in selected order to 1024 long-range keys, then add the 512-token recent window and four sinks, removing overlap and invalid/future IDs. The long-range condition is `4 <= key_id <= query_position - 512`. At mature positions the total selected set has 1540 IDs. At short positions its actual length differs and must be measured/returned, not hard-coded.

A query at position `p` may see only `floor(p/512)*64` finalized centroids. This is intentionally based on the query's position, not the latest appended end of a prefill chunk. The current 512-token block is covered by the recent window. Clustering a block completed later in a batched prefill must never expose it to earlier queries.

Native QSA is a different representation: four indexer query heads, width 128, raw projections from layer input, RMSNorm/RoPE, pooling each four cells, native top-2048 policy. The learned matrices cannot replace its BF16 model indexer tensors or reuse its tail layout. Scanning all projected token keys without Ward buckets is a different algorithm and must be named/qualified separately.

## Three execution paths that require wiring

| Path | Source location | Necessary insertion |
|---|---|---|
| Single-token decode | `src/core/layer.cpp:qsa_layer`, approximately 1130–1300 | After normalized K has been appended, read the actual stored INT8 K and append its learned projection. After normalized attention Q exists, project/group Q. Replace native score/top-k with LSA IDs, publish exact width, then retain native KV resolve and attention. |
| Batched prompt prefill | `src/prefill/prefill.cpp`, QSA branch approximately 3283–3511 | Project appended INT8 cache keys and normalized `m.q`; build closed blocks across chunk boundaries; select each query using its own causal centroid limit; fill `m.sel_ids` and widths before existing prompt attention. |
| Speculative verification and multiple slots | `src/core/verify.cpp`, QSA branch approximately 1560–1890 | Project actual per-slot cache K and actual Q for every proposal; preserve per-row slot ownership and causal limits; produce each row's IDs/width. Respect branch-stream joins; do not call existing native selectors afterward. |

Changing only `qsa_layer` is insufficient. Production prompt prefill and the Verifier implement their own QSA chains. All three must share one explicitly selected map/selector contract before a native-serving claim is valid.

Existing readers can remain: `qsa_kv_resolve`, `qsa_decode_attn_step`, `qsa_decode_attn_batch`, slot/row variants and `qsa_prompt_attn_batch`. Keep the selected same-model weights, control vector, gating and output projection intact. Keeping native attention avoids replacing the actual INT8 reads with the standalone smoke's FP16 gather adapter.

## Width, page layout and graph capture

`include/strata/kernels/qsa.hpp` defines four device step fields: position, KV count, completed native block count, selection width. The stock width is `min(n_kv,2048+4-1)`, up to 2051. The attention and gather kernels read `step[kStepWidth]`; allocation capacity is separate.

For the first bridge keep the existing 2051-element capacity and scratch strides. An LSA selector must publish the actual unique-ID length into a separate step record or into the attention record only after all native consumers of its width are finished. Unused IDs must be initialized. A 1540-ID output with a stale 2051 width reads stale/uninitialized IDs. Do not globally change `qsa_step_fill` until every native indexer consumer is accounted for. Use sorted ascending unique IDs for native readers unless a targeted test demonstrates the alternate order is supported and acceptable numerically. Sorting preserves the set but changes accumulation order; its output tolerance must be measured.

`QsaState` in `include/strata/core/layer.hpp` owns persistent per-layer/per-sequence state. `QsaBuffers` is scratch shared across layers. LSA finalized buckets/open-block state belong to the former; maps are shared immutable per-layer data. Per-query projection/select scratch belongs to the latter or verifier/prefill scratch arenas, never a global cross-slot mutable buffer.

INT8 codes are `[physical_page][kv_head][page_size][256]`, scales `[physical_page][kv_head][page_size][4]`; production geometry uses `page_size=4`. Logical page `token/4` maps through `page_table`, offset `token%4`. Recover each needed coordinate as FP32(code) times FP32(FP16 scale). Reading `b.kcur` before cache quantization or the FP16 gather scratch violates the learned-input evidence. The initial candidate should explicitly require plain unrotated INT8 K with fully resident main KV. Q4, rotated INT8, hybrid and streamed main KV need separate adapters and validation; a residency request may be normalized to `kv_mode=0`, so inspect runtime state rather than a CLI spelling.

Captured graphs require fixed pointer addresses and capacity-sized launches with dynamic counts read from device memory. `qsa_layer` explicitly records the earlier dangling-host-pointer failure: host staging must be pinned and persistent. No stack/vector temporary may become a captured H2D source. Do not freeze positions, valid bucket counts or accepted counts as scalar launch arguments. No allocation, CPU readback, Python/Triton call or host Ward implementation belongs inside graph capture. Preallocate clustering/select scratch and any per-slot pointer tables; define stream order between cache append, projection, block closure, Q projection, selection, resolve and attention.

The side-mixer and Q branches in `verify.cpp` previously relied on the native indexer reading layer input independently of K/Q. LSA reads the attention K/Q, so those old independence assumptions no longer hold. The LSA path must join both K append and Q normalization before selecting. A minimal first candidate can keep these stages on the main stream, with that performance difference recorded.

## Speculative commit, rejection and conversation reuse

The Verifier currently appends all proposed K/V but leaves GDN uncommitted. Its header explains that KV past the accepted prefix is overwritten before reuse. Native indexer tails are snapshotted and restored/replayed because rejected writes can overwrite needed tail slots.

Exact integration points:

- `verify.cpp:idx_snap` around 1627: snapshot the LSA open-block state before the first proposal per slot. Native `tail_snap_` has only `(4-1)*128` floats and cannot hold a 512-key LSA tail.
- During proposal execution retain the projected key rows for later replay, per QSA layer and proposal. Finalized blocks may be tentatively appended, but every query must use its own position bound.
- `Verifier::capture_commit` around 2594–2700 and `capture_commit_batch` around 3525–3618: restore the LSA open tail and pre-window logical counts, then replay accepted projected key rows. Reclose only accepted complete blocks; leave rejected finalized rows beyond the valid count inaccessible and overwrite them later. Retain/correct the existing accepted-count device semantics, including zero kept where allowed.
- Batched self-commit can skip the commit graph. Its one-row/per-slot state update must advance LSA state too. Multiple proposal rows for the same slot must process in sequence, not in independent racing blocks.

An efficient cache keeps immutable closed block rows plus at most 512 FP32 projected keys for the open block. With speculative windows much shorter than 512, a full open-tail snapshot is simple and bounded. Crossing a 512 boundary is the dangerous case: proposals for the next block may overwrite the start of the old tail before rejection. Snapshot/replay handles it. Merely decrementing a closed-bucket count does not restore the raw tail.

Conversation reuse is another rollback path. `src/core/conversation_state.cpp` saves/restores/copies native indexer tails, dead row and block position; `conversation_snapshot.cpp` saves pooled native indexer rows and KV. LSA state must be included in checkpoint save/restore, slot copy/move, parked snapshots and on-disk identity/version checks, or deterministically rebuilt from the preserved KV before reuse. Map hash and selector parameters must be part of cache identity. Keeping eight resident chats does not remove prefix rewind and slot-copy behavior. Disable unsupported cache modes explicitly in an early isolated candidate; doing so is a recorded limitation, not preserved production behavior.

## Memory at eight slots and 131072 tokens

Static sizing estimates, not measured allocations. These exclude CUDA context, allocator alignment, Ward temporary matrices, graph storage, expert memory and prefill scratch. There are 96 main QSA states: 12 layers times eight slots.

| State | Per layer/slot | All 96 states |
|---|---:|---:|
| BF16 centroids: 256 closed blocks × 64 buckets × 64 dims × 2 bytes | 2 MiB | 192 MiB |
| Members: 16384 buckets × 16 int32 IDs | 1 MiB | 96 MiB |
| Bucket sizes if separate int32 array | 64 KiB | 6 MiB |
| Open projected tail: 512 × 64 FP32 | 128 KiB | 12 MiB |
| One full open-tail snapshot per state | 128 KiB | 12 MiB |
| Full FP32 per-token projected-key history, if retained instead | 32 MiB | 3 GiB |

Maps are shared: 2.25 MiB FP32 payload, not eight copies. A flat centroid score scan needs at most 16384 FP32 scores = 64 KiB per query, plus 385 picks, 1540 output IDs and reduction/sort workspace. Eight simultaneous query rows therefore need at least 512 KiB scores. A 256-query prefill tile needs 16 MiB scores if materialized. Do not allocate the whole prompt-by-centroid score matrix. A 512-key block closure begins with 128 pooled vectors; an explicit FP32 128×128 Gram matrix alone is 64 KiB per block in flight, plus vectors, labels and reduction state. Choose a bounded closure batch and include its actual allocator/capture cost in later measurements.

Keeping the old native indexer alongside LSA during bring-up avoids changing the default memory layout but does not realize a memory saving. Removing its arena requires adjusting session planners, snapshot sizing, Verifier arenas and legacy paths consistently. Do not reduce expert residency, context or slot count to conceal unbudgeted LSA memory.

## Next wiring and validation steps

1. Export a versioned native binary artifact from the immutable learned `.pt` and JSON; validate exactly the 12 required layer pairs, dimensions, FP32 values, finite positive temperatures, input contract, hashes and exact file length. Load once before any graphs or sessions. Missing or incompatible explicit opt-in data must fail startup.
2. Implement standalone native CUDA projection and centroid ranking/expansion bridges. Compare actual captured inputs and immutable Ward-produced buckets to the existing smoke/reference. These establish component parity only; they do not establish a native LSA cache or engine integration.
3. Implement/port bounded native Ward closure and incremental per-slot cache state. Preserve the reviewed 512/4/64/16 geometry and exact visibility/expansion policy. Offline preclustered captures cannot stand in for live incremental closure.
4. Add opt-in configuration, sizing and all three engine call sites. Preserve the default route. Include persistent graph-safe inputs, real per-query lengths, slot ownership, boundary masks, native attention, MTP commit/reject and conversation cache behavior.
5. Verify short contexts, 511/512/513 and 1023/1024 boundaries; split vs monolithic prefill; arbitrary chunk boundaries; speculative keep counts from zero through all proposals; boundary-crossing rejection followed by different replacement tokens; interleaved slots; fresh start vs snapshot restore/copy; map-hash mismatch rejection; final context capacity. Compare incremental state and selected sets against reconstruction from the accepted KV prefix.
6. Run actual same-model native candidate/control comparisons with the retained IQ3_S/control vector, INT8 KV, MTP, eight slots, 131072 context and current runtime behavior. Use new independent held-out quality documents: the pilot test set is consumed. Measure whole-model TTFT, prefill, decode, latency distribution, peak VRAM, repeated-turn cache behavior, tools/vision behavior where required and quality. Component timings or improved KL versus PCA cannot justify deployment.

Promotion remains conditional on a complete candidate proving better and preserving required behavior. Standalone components, this static audit, source compilation, synthetic parity and the prior pilot do not satisfy that condition.
