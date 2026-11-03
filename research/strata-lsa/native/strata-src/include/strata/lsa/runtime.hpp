#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::lsa {

// Opt-in CUDA runtime state. Every pointer is carved from the owning QSA arena
// before graph capture; no allocation, host readback or pointer changes happen
// during a forward. Full projected history preserves accepted prefixes across
// speculative rejection. Future rows are overwritten before becoming visible.
struct RuntimeState {
    int capacity = 0;
    int buckets = 0;
    int physical_pages = 0;
    int page_size = 4;
    int model_layer = -1;
    float* history = nullptr;        // [capacity,64] FP32 normalized projected K
    float* centroids = nullptr;      // [buckets,64] BF16-rounded FP32
    int32_t* members = nullptr;      // [buckets,16]
    float* key_row = nullptr;        // [64]
    float* query_row = nullptr;      // [64]
    int64_t* keys_in = nullptr;      // [buckets]
    int64_t* keys_out = nullptr;     // [buckets]
    int32_t* offsets = nullptr;     // [2]
    int32_t* selected = nullptr;    // [385]
    int32_t* cells = nullptr;       // [1540]
    int32_t* steps = nullptr;       // [4], private LSA-width record
    void* sort_temporary = nullptr;
    size_t sort_bytes = 0;
    void* ward_scratch = nullptr;   // sticky error and closure diagnostics
    size_t arena_bytes = 0;
};

// One workspace per Prefill object, shared by its serial QSA layers. The model
// planner prices it with the prompt scratch region before expert allocation.
// It is never copied into each resident conversation/layer RuntimeState.
struct RuntimePrefillScratch {
    int queries = 0;
    int buckets = 0;
    float* query_rows = nullptr;    // [queries,64]
    int32_t* positions = nullptr;  // [queries]
    int64_t* keys_in = nullptr;    // [queries,buckets]
    int64_t* keys_out = nullptr;
    int32_t* offsets = nullptr;   // [queries+1]
    int32_t* selected = nullptr;  // [queries,385]
    int32_t* cells = nullptr;     // [queries,1540]
    int32_t* steps = nullptr;     // [queries,4]
    void* sort_temporary = nullptr;
    size_t sort_bytes = 0;
    size_t arena_bytes = 0;
};

// Call after selecting the CUDA device and before arena pricing. An explicit
// incompatible/missing map refuses startup. No environment opt-in leaves the
// existing inference route untouched. Configuration is immutable thereafter.
bool runtime_configure_from_env(std::string& error);
bool runtime_enabled();
const std::string& runtime_map_path();
uint64_t runtime_map_identity();
size_t runtime_state_bytes(int64_t capacity);
bool runtime_state_init(RuntimeState& state, int64_t capacity, int physical_pages,
                        int page_size, void* arena, size_t arena_bytes, std::string& error);
void runtime_zero(const RuntimeState& state, void* stream);
size_t runtime_prefill_scratch_bytes(int64_t capacity, int queries = 128);
bool runtime_prefill_scratch_init(RuntimePrefillScratch& scratch, int64_t capacity, int queries,
                                 void* arena, size_t arena_bytes, std::string& error);

bool runtime_step(const RuntimeState& state, int model_layer,
                  const strata::kernels::QsaAttnPools& pools, const float* query,
                  int32_t* step, int32_t* ids, int64_t cap, void* stream, std::string& error);
bool runtime_prefill(const RuntimeState& state, int model_layer,
                     const strata::kernels::QsaAttnPools& pools, const float* queries,
                     int64_t start, int64_t count, int32_t* steps, int64_t step_stride,
                     int32_t* ids, int64_t cap, void* stream, std::string& error,
                     const RuntimePrefillScratch* scratch = nullptr);

// Outside graph capture only. Restored INT8 KV is authoritative; rebuilding
// derives fresh maps/clusters, so no LSA bytes are trusted from old snapshots.
bool runtime_rebuild(const RuntimeState& state, const strata::kernels::QsaAttnPools& pools,
                     int64_t tokens, std::string& error);
// Ordered on stream. Copies the accepted prefix plus closed block data only;
// callers must order this after any nonblocking KV copies themselves.
bool runtime_copy(const RuntimeState& from, const RuntimeState& to, int64_t tokens,
                  void* stream, std::string& error);
// Outside capture: synchronize the device and report Ward diagnostics. A failed
// closure also traps on device before inference can consume its bucket output.
bool runtime_check(const RuntimeState& state, std::string& error);
bool runtime_stats(const RuntimeState& state, int32_t (&values)[4], std::string& error);

} // namespace strata::lsa
