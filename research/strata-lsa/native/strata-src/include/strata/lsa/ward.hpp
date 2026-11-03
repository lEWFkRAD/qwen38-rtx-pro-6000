#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::lsa {
// Native spherical Ward closure for the pinned Flash-Next LSA geometry:
// 512 FP32 normalized projected keys of64 dims; consecutive pools of4;
// mutual-cheapest Ward rounds128->64, max16 original keys/bucket.
// Output arrays are indexed by absolute512-token block:
// centroids [ceil(capacity/512),64,64] FP32 storage of BF16-rounded unit rows;
// members [ceil(capacity/512),64,16] absolute token IDs, ascending, -1 padded.
//
// Positive capacities below512 are valid for step calls: no complete block
// can close, so valid positions are no-ops. Range calls require complete blocks.
// Caller owns all storage and an explicit nondefault CUDA stream. No allocation,
// host readback, synchronization or position-dependent host launch occurs.
// Scratch is16 bytes, aligned for int32 and zero-initialized before first use:
// [sticky_error, completed_closures, max_rounds, reserved]. Error codes:
// -1 invalid device position; -2 nonfinite input/arithmetic; -3 Ward stalled;
// -4 invalid final partition. A skipped nonboundary step does not clear errors.
// Inspect sticky_error at the enclosing request's validation boundary. Do not
// use a failed closure's zero centroids and -1 members as a successful index.
std::size_t ward_scratch_bytes();
int ward_close_step(const float* history, const int32_t* position, int capacity,
                    float* centroids, int32_t* members, void* scratch, void* stream);
// Uncaptured prefill: first_block and block_count name fully projected blocks.
// Graph capture is also legal when these host bounds are fixed for that graph.
int ward_close_range(const float* history, int capacity, int first_block, int block_count,
                     float* centroids, int32_t* members, void* scratch, void* stream);
} // namespace strata::lsa

extern "C" {
std::size_t lsa_ward_scratch_bytes();
int lsa_ward_close_step(const float*, const int32_t*, int, float*, int32_t*, void*, void*);
int lsa_ward_close_range(const float*, int, int, int, float*, int32_t*, void*, void*);
// Qualification-only variant. labels [capacity/512,128] and objective
// [capacity/512] expose partition evidence; they may not be null. The labels
// repeat across each consecutive4-key input pool. Runtime does not call it.
int lsa_ward_close_range_diagnostic(const float*, int, int, int, float*, int32_t*,
                                   void*, int32_t*, float*, void*);
}
