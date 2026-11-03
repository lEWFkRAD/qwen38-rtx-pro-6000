#include "strata/lsa/runtime.hpp"
#include "strata/lsa/cuda_adapter.h"
#include "strata/lsa/ward.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iterator>
#include <mutex>
#include <string>

namespace strata::lsa {
namespace {
constexpr int D = 64, P = 385, W = 1540;
void* maps_handle = nullptr; // Immutable, intentionally retained until process exit.
bool configured = false;
std::string maps_path;
std::mutex configuration_lock;

size_t rounded(size_t bytes) { return (bytes + 255) & ~size_t(255); }
int bucket_capacity(int64_t capacity) { return std::max<int64_t>(1, (capacity + 511) / 512) * 64; }
bool status(int code, const char* what, std::string& error) {
    if (code == cudaSuccess) return true;
    error = std::string("LSA runtime ") + what + ": " + cudaGetErrorString((cudaError_t) code);
    return false;
}
bool valid_capacity(int64_t capacity) { return capacity > 0 && capacity <= 131072; }

size_t bytes_for(int64_t capacity, size_t sort_bytes) {
    const size_t buckets = bucket_capacity(capacity);
    size_t bytes = 256; // Up to 255 bytes to align the supplied arena's first address.
    for (size_t n : {size_t(capacity) * D * 4, buckets * D * 4, buckets * 16 * 4,
                     size_t(D * 4), size_t(D * 4), buckets * 8, buckets * 8,
                     size_t(2 * 4), size_t(P * 4), size_t(W * 4), size_t(4 * 4),
                     sort_bytes, ward_scratch_bytes()}) bytes += rounded(n);
    return bytes;
}

size_t prefill_bytes_for(int64_t capacity, int queries, size_t sort_bytes) {
    const size_t buckets = bucket_capacity(capacity), rows = size_t(queries);
    size_t bytes = 256;
    for (size_t n : {rows * D * 4, rows * 4, rows * buckets * 8, rows * buckets * 8,
                     (rows + 1) * 4, rows * P * 4, rows * W * 4, rows * 4 * 4, sort_bytes})
        bytes += rounded(n);
    return bytes;
}

struct Arena {
    uintptr_t next;
    uintptr_t end;
    void* take(size_t bytes) {
        next = (next + 255) & ~uintptr_t(255);
        if (next > end || bytes > end - next) return nullptr;
        void* pointer = reinterpret_cast<void*>(next);
        next += rounded(bytes);
        return pointer;
    }
};

__global__ void store_projected(const float* row, const int32_t* step, int capacity, float* history) {
    const int position = step[0];
    if (position < 0 || position >= capacity) { asm volatile("trap;"); return; }
    history[size_t(position) * D + threadIdx.x] = row[threadIdx.x];
}

__global__ void closure_guard(const int32_t* diagnostics) {
    // Fail closed before a failed closure's cleared output can become attention.
    // The enclosing engine synchronization surfaces a candidate CUDA failure.
    if (*diagnostics < 0) asm volatile("trap;");
}

__global__ void publish(const int32_t* cells, const int32_t* selected_step,
                        int32_t* ids, int cap, int32_t* step) {
    for (int i = threadIdx.x; i < cap; i += blockDim.x) ids[i] = i < W ? cells[i] : -1;
    if (threadIdx.x == 0) step[3] = selected_step[3];
}

__global__ void positions_from_steps(const int32_t* steps, int64_t stride, int rows,
                                    int first_position, int32_t* positions) {
    const int row = threadIdx.x;
    if (row >= rows) return;
    const int position = steps[size_t(row) * stride];
    if (position != first_position + row) { asm volatile("trap;"); return; }
    positions[row] = position;
}

__global__ void publish_batch(const int32_t* cells, const int32_t* selected_steps,
                              int32_t* ids, int cap, int32_t* steps, int64_t step_stride) {
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < cap; i += blockDim.x)
        ids[size_t(row) * cap + i] = i < W ? cells[size_t(row) * W + i] : -1;
    if (threadIdx.x == 0) steps[size_t(row) * step_stride + 3] = selected_steps[row * 4 + 3];
}

bool validate(const RuntimeState& s, int layer, const strata::kernels::QsaAttnPools& pools,
              int64_t cap, void* stream, std::string& error) {
    if (!maps_handle || !s.history || !s.centroids || !s.members || !stream || cap < W || cap > INT32_MAX ||
        layer < 3 || layer > 47 || (layer - 3) % 4 || s.model_layer != layer ||
        !pools.k_q || !pools.k_scale || !pools.page_table || pools.k_q4 || pools.k_pool ||
        !valid_capacity(s.capacity)) {
        error = "LSA runtime: incompatible state, map layer, INT8 pool or output capacity";
        return false;
    }
    return true;
}

bool select_row(const RuntimeState& s, int layer, const float* query, int32_t* step,
                int32_t* ids, int64_t cap, void* stream, std::string& error) {
    if (!query || !step || !ids) { error = "LSA runtime: missing query/step/selection buffer"; return false; }
    if (!status(lsa_project_queries(query, 1, lsa_maps_query(maps_handle, layer), s.query_row, stream),
                "query projection", error)) return false;
    if (!status(lsa_select(s.query_row, s.centroids, s.members, step, 1, s.buckets, s.keys_in, s.keys_out,
                           s.offsets, s.sort_temporary, s.sort_bytes, s.selected, s.cells, s.steps, stream),
                "selection", error)) return false;
    publish<<<1, 256, 0, (cudaStream_t) stream>>>(s.cells, s.steps, ids, (int) cap, step);
    return status(cudaGetLastError(), "selection publication", error);
}
} // namespace

bool runtime_configure_from_env(std::string& error) {
    std::lock_guard<std::mutex> lock(configuration_lock);
    const char* value = std::getenv("STRATA_LSA_MAPS");
    const std::string requested = value ? value : "";
    if (configured) {
        if (requested != maps_path) { error = "LSA map configuration cannot change after startup"; return false; }
        error.clear();
        return true;
    }
    if (requested.empty()) { configured = true; error.clear(); return true; }
    char why[1024] = {};
    void* candidate = lsa_maps_open(requested.c_str(), why, sizeof(why));
    if (!candidate) { error = std::string("LSA map startup refused: ") + why; return false; }
    maps_handle = candidate;
    maps_path = requested;
    configured = true;
    error.clear();
    return true;
}

bool runtime_enabled() { return maps_handle != nullptr; }
const std::string& runtime_map_path() { return maps_path; }
uint64_t runtime_map_identity() { return lsa_maps_identity(maps_handle); }

size_t runtime_state_bytes(int64_t capacity) {
    if (!runtime_enabled() || !valid_capacity(capacity)) return 0;
    size_t sort_bytes = 0;
    if (lsa_select_temp_bytes(1, bucket_capacity(capacity), &sort_bytes) != cudaSuccess) return 0;
    return bytes_for(capacity, sort_bytes);
}

bool runtime_state_init(RuntimeState& out, int64_t capacity, int physical_pages, int page_size,
                        void* memory, size_t bytes, std::string& error) {
    if (!runtime_enabled() || !valid_capacity(capacity) || !memory || physical_pages <= 0 ||
        page_size <= 0 || page_size > 512 || int64_t(physical_pages) * page_size < capacity) {
        error = "LSA runtime: unsupported state allocation geometry"; return false;
    }
    size_t sort_bytes = 0;
    if (!status(lsa_select_temp_bytes(1, bucket_capacity(capacity), &sort_bytes), "sort sizing", error)) return false;
    if (bytes < bytes_for(capacity, sort_bytes)) { error = "LSA runtime: undersized state arena"; return false; }
    Arena arena{reinterpret_cast<uintptr_t>(memory), reinterpret_cast<uintptr_t>(memory) + bytes};
    RuntimeState s;
    s.capacity = (int) capacity; s.buckets = bucket_capacity(capacity);
    s.physical_pages = physical_pages; s.page_size = page_size; s.sort_bytes = sort_bytes; s.arena_bytes = bytes;
    s.history = static_cast<float*>(arena.take(size_t(capacity) * D * 4));
    s.centroids = static_cast<float*>(arena.take(size_t(s.buckets) * D * 4));
    s.members = static_cast<int32_t*>(arena.take(size_t(s.buckets) * 16 * 4));
    s.key_row = static_cast<float*>(arena.take(D * 4));
    s.query_row = static_cast<float*>(arena.take(D * 4));
    s.keys_in = static_cast<int64_t*>(arena.take(size_t(s.buckets) * 8));
    s.keys_out = static_cast<int64_t*>(arena.take(size_t(s.buckets) * 8));
    s.offsets = static_cast<int32_t*>(arena.take(2 * 4));
    s.selected = static_cast<int32_t*>(arena.take(P * 4));
    s.cells = static_cast<int32_t*>(arena.take(W * 4));
    s.steps = static_cast<int32_t*>(arena.take(4 * 4));
    s.sort_temporary = arena.take(sort_bytes);
    s.ward_scratch = arena.take(ward_scratch_bytes());
    if (!s.ward_scratch || !s.sort_temporary) { error = "LSA runtime: state carving overflow"; return false; }
    if (!status(cudaMemset(s.ward_scratch, 0, ward_scratch_bytes()), "state initialization", error)) return false;
    out = s;
    return true;
}

void runtime_zero(const RuntimeState& s, void* stream) {
    // No reader can name unwritten projected history or unfinished blocks. A
    // reset only clears diagnostics; later closures overwrite their whole block.
    if (s.history) cudaMemsetAsync(s.ward_scratch, 0, ward_scratch_bytes(), (cudaStream_t) stream);
}

size_t runtime_prefill_scratch_bytes(int64_t capacity, int queries) {
    if (!runtime_enabled() || !valid_capacity(capacity) || queries < 1 || queries > 128) return 0;
    size_t sort_bytes = 0;
    if (lsa_select_temp_bytes(queries, bucket_capacity(capacity), &sort_bytes) != cudaSuccess) return 0;
    return prefill_bytes_for(capacity, queries, sort_bytes);
}

bool runtime_prefill_scratch_init(RuntimePrefillScratch& out, int64_t capacity, int queries,
                                 void* memory, size_t bytes, std::string& error) {
    if (!runtime_enabled() || !valid_capacity(capacity) || queries < 1 || queries > 128 || !memory) {
        error = "LSA runtime: unsupported prefill workspace geometry"; return false;
    }
    size_t sort_bytes = 0;
    if (!status(lsa_select_temp_bytes(queries, bucket_capacity(capacity), &sort_bytes),
                "prefill sort sizing", error)) return false;
    if (bytes < prefill_bytes_for(capacity, queries, sort_bytes)) {
        error = "LSA runtime: undersized prefill workspace"; return false;
    }
    Arena arena{reinterpret_cast<uintptr_t>(memory), reinterpret_cast<uintptr_t>(memory) + bytes};
    RuntimePrefillScratch scratch;
    scratch.queries = queries; scratch.buckets = bucket_capacity(capacity);
    scratch.sort_bytes = sort_bytes; scratch.arena_bytes = bytes;
    const size_t rows = size_t(queries), buckets = size_t(scratch.buckets);
    scratch.query_rows = static_cast<float*>(arena.take(rows * D * 4));
    scratch.positions = static_cast<int32_t*>(arena.take(rows * 4));
    scratch.keys_in = static_cast<int64_t*>(arena.take(rows * buckets * 8));
    scratch.keys_out = static_cast<int64_t*>(arena.take(rows * buckets * 8));
    scratch.offsets = static_cast<int32_t*>(arena.take((rows + 1) * 4));
    scratch.selected = static_cast<int32_t*>(arena.take(rows * P * 4));
    scratch.cells = static_cast<int32_t*>(arena.take(rows * W * 4));
    scratch.steps = static_cast<int32_t*>(arena.take(rows * 4 * 4));
    scratch.sort_temporary = arena.take(sort_bytes);
    if (!scratch.sort_temporary) { error = "LSA runtime: prefill workspace carving overflow"; return false; }
    out = scratch;
    return true;
}

bool runtime_step(const RuntimeState& s, int layer, const strata::kernels::QsaAttnPools& pools,
                  const float* query, int32_t* step, int32_t* ids, int64_t cap, void* stream, std::string& error) {
    if (!validate(s, layer, pools, cap, stream, error) || !step) return false;
    if (!status(lsa_project_key_steps(pools.k_q, pools.k_scale, pools.page_table, s.physical_pages, s.page_size,
                                     step, 1, lsa_maps_key(maps_handle, layer), s.key_row, stream),
                "key projection", error)) return false;
    store_projected<<<1, D, 0, (cudaStream_t) stream>>>(s.key_row, step, s.capacity, s.history);
    if (!status(cudaGetLastError(), "projected history append", error)) return false;
    if (!status(ward_close_step(s.history, step, s.capacity, s.centroids, s.members, s.ward_scratch, stream),
                "block closure", error)) return false;
    closure_guard<<<1, 1, 0, (cudaStream_t) stream>>>((const int32_t*) s.ward_scratch);
    if (!status(cudaGetLastError(), "closure guard", error)) return false;
    return select_row(s, layer, query, step, ids, cap, stream, error);
}

bool runtime_prefill(const RuntimeState& s, int layer, const strata::kernels::QsaAttnPools& pools,
                     const float* queries, int64_t start, int64_t count, int32_t* steps, int64_t step_stride,
                     int32_t* ids, int64_t cap, void* stream, std::string& error,
                     const RuntimePrefillScratch* scratch) {
    if (!validate(s, layer, pools, cap, stream, error) || !queries || !steps || !ids ||
        start < 0 || count < 1 || count > s.capacity || start > s.capacity - count || step_stride < 4) {
        if (error.empty()) error = "LSA runtime: invalid prompt range or buffers";
        return false;
    }
    if (!status(lsa_project_keys(pools.k_q, pools.k_scale, pools.page_table, s.physical_pages, s.page_size,
                                (int) start, (int) count, lsa_maps_key(maps_handle, layer),
                                s.history + size_t(start) * D, stream), "prompt key projection", error)) return false;
    const int first = (int) (start / 512), blocks = (int) ((start + count) / 512) - first;
    if (blocks && !status(ward_close_range(s.history, s.capacity, first, blocks, s.centroids, s.members,
                                          s.ward_scratch, stream), "prompt block closure", error)) return false;
    closure_guard<<<1, 1, 0, (cudaStream_t) stream>>>((const int32_t*) s.ward_scratch);
    if (!status(cudaGetLastError(), "prompt closure guard", error)) return false;
    if (scratch) {
        const auto& b = *scratch;
        if (b.queries < 1 || b.queries > 128 || b.buckets < s.buckets || !b.query_rows || !b.positions ||
            !b.keys_in || !b.keys_out || !b.offsets || !b.selected || !b.cells || !b.steps || !b.sort_temporary) {
            error = "LSA runtime: missing or incompatible batched prefill workspace";
            return false;
        }
        for (int64_t first_row = 0; first_row < count; first_row += b.queries) {
            const int rows = (int) std::min<int64_t>(b.queries, count - first_row);
            const int last_position = (int) (start + first_row + rows - 1);
            // Prompt counts are host-known outside capture. Keep the exact
            // device causal mask per query while avoiding scans of unused
            // future-capacity buckets. Each packed row uses this active stride.
            const int active = std::max(64, std::min(s.buckets, (last_position / 512) * 64));
            size_t needed = 0;
            if (!status(lsa_select_temp_bytes(rows, active, &needed), "active prefill sort sizing", error)) return false;
            if (needed > b.sort_bytes) { error = "LSA runtime: active sort exceeds planned prefill workspace"; return false; }
            int32_t* original_steps = steps + first_row * step_stride;
            positions_from_steps<<<1, 128, 0, (cudaStream_t) stream>>>(original_steps, step_stride, rows,
                                                                       (int) (start + first_row), b.positions);
            if (!status(cudaGetLastError(), "prefill position packing", error)) return false;
            if (!status(lsa_project_queries(queries + size_t(first_row) * 24 * 256, rows,
                                            lsa_maps_query(maps_handle, layer), b.query_rows, stream),
                        "batched prompt query projection", error)) return false;
            if (!status(lsa_select(b.query_rows, s.centroids, s.members, b.positions, rows, active,
                                   b.keys_in, b.keys_out, b.offsets, b.sort_temporary, b.sort_bytes,
                                   b.selected, b.cells, b.steps, stream), "batched prompt selection", error)) return false;
            publish_batch<<<rows, 256, 0, (cudaStream_t) stream>>>(b.cells, b.steps, ids + first_row * cap,
                                                                  (int) cap, original_steps, step_stride);
            if (!status(cudaGetLastError(), "batched prompt selection publication", error)) return false;
        }
        return true;
    }
    for (int64_t row = 0; row < count; ++row)
        if (!select_row(s, layer, queries + size_t(row) * 24 * 256, steps + row * step_stride,
                         ids + row * cap, cap, stream, error)) return false;
    return true;
}

bool runtime_stats(const RuntimeState& s, int32_t (&values)[4], std::string& error) {
    if (!s.history) { std::fill(std::begin(values), std::end(values), 0); return true; }
    return status(cudaDeviceSynchronize(), "statistics synchronization", error) &&
           status(cudaMemcpy(values, s.ward_scratch, sizeof(values), cudaMemcpyDeviceToHost), "statistics read", error);
}

bool runtime_check(const RuntimeState& s, std::string& error) {
    int32_t values[4] = {};
    if (!runtime_stats(s, values, error)) return false;
    if (values[0] < 0) { error = "LSA native Ward closure failed: " + std::to_string(values[0]); return false; }
    return true;
}

bool runtime_rebuild(const RuntimeState& s, const strata::kernels::QsaAttnPools& pools,
                     int64_t tokens, std::string& error) {
    if (!s.history) return true;
    if (tokens < 0 || tokens > s.capacity) { error = "LSA runtime: restored prefix exceeds capacity"; return false; }
    cudaStream_t stream = nullptr;
    if (!status(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "restore stream", error)) return false;
    bool ok = validate(s, s.model_layer, pools, W, stream, error);
    if (ok) ok = status(cudaMemsetAsync(s.ward_scratch, 0, ward_scratch_bytes(), stream), "restore reset", error);
    if (ok && tokens)
        ok = status(lsa_project_keys(pools.k_q, pools.k_scale, pools.page_table, s.physical_pages, s.page_size,
                                     0, (int) tokens, lsa_maps_key(maps_handle, s.model_layer), s.history, stream),
                     "restored key projection", error);
    if (ok && tokens >= 512)
        ok = status(ward_close_range(s.history, s.capacity, 0, (int) (tokens / 512), s.centroids, s.members,
                                     s.ward_scratch, stream), "restored block closure", error);
    int32_t diagnostics[4] = {};
    if (ok) ok = status(cudaMemcpyAsync(diagnostics, s.ward_scratch, sizeof(diagnostics),
                                       cudaMemcpyDeviceToHost, stream), "restored closure diagnostics", error);
    if (ok) ok = status(cudaStreamSynchronize(stream), "restored cache synchronization", error);
    const cudaError_t destroy = cudaStreamDestroy(stream);
    if (ok) ok = status(destroy, "restore stream release", error);
    if (ok && diagnostics[0] < 0) {
        error = "LSA restored native Ward closure failed: " + std::to_string(diagnostics[0]);
        ok = false;
    }
    return ok;
}

bool runtime_copy(const RuntimeState& from, const RuntimeState& to, int64_t tokens,
                  void* stream, std::string& error) {
    if (!from.history && !to.history) return true;
    // A short-context MTP drafter can request ring_cells==0 and receive an
    // unused, correctly priced arena. Only session-owned main QSA layers bind
    // a map layer; the drafter still uses its own unchanged attention path.
    if (from.model_layer < 0 && to.model_layer < 0) return true;
    if (!from.history || !to.history || from.capacity != to.capacity || from.model_layer != to.model_layer ||
        tokens < 0 || tokens > from.capacity) { error = "LSA runtime: incompatible copied prefix"; return false; }
    const size_t blocks = size_t(tokens / 512), history_bytes = size_t(tokens) * D * 4;
    const auto st = (cudaStream_t) stream;
    auto copy = [&](void* destination, const void* source, size_t bytes) {
        return !bytes || status(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, st),
                                "prefix copy", error);
    };
    return copy(to.history, from.history, history_bytes) &&
           copy(to.centroids, from.centroids, blocks * 64 * D * 4) &&
           copy(to.members, from.members, blocks * 64 * 16 * 4) &&
           copy(to.ward_scratch, from.ward_scratch, ward_scratch_bytes());
}

} // namespace strata::lsa
