#pragma once
#include <stddef.h>
#include <stdint.h>

// Research-only CUDA bridge. No engine path calls this API. All device buffers
// are caller-owned and must remain alive until the explicit stream completes.
// No allocation or synchronization occurs during projection/selection/attention.
// Geometry is Flash-Next INT8 KV, 24 Q heads, 2 KV heads, 256 coordinates,
// trailing 192 nonrotary coordinates, learned 384 x 64 row-major FP32 maps.
// Returns cudaError_t as int. Inputs must be finite. Projection ignores invalid
// page IDs defensively. Native attention requires validated page and cell tables;
// its existing reader does not validate physical allocation bounds.
#ifdef __cplusplus
extern "C" {
#endif
// Startup/teardown only: strictly load the native map artifact and copy weights
// to persistent device storage. These calls allocate and synchronize. Getter
// pointers are borrowed until close, and layer is the model layer (3,7,...,47).
// A handle belongs to the CUDA device/context current at open. Keep that device
// current for its use and close; finish every stream and destroy captured graphs
// referencing its weights before close. No reference-counted lifetime is implied.
void* lsa_maps_open(const char* path, char* error, size_t error_capacity);
const float* lsa_maps_query(void* handle, int layer);
const float* lsa_maps_key(void* handle, int layer);
float lsa_maps_temperature(void* handle, int layer);
// Immutable full-byte FNV-1a 64 identity of the artifact actually loaded by open.
// Cache compatibility only, not authentication. Null handle returns zero.
uint64_t lsa_maps_identity(void* handle);
void lsa_maps_close(void* handle);
int lsa_project_keys(const int8_t* codes, const uint16_t* scales,
                     const int32_t* page_table, int physical_pages, int page_size,
                     int start, int count, const float* map, float* output, void* stream);
// Capturable decode/verification form: positions are read from device memory.
// Fully resident KV only: page_table has physical_pages logical entries and
// pools have physical_pages pages. Positions outside that span produce zeros.
int lsa_project_key_steps(const int8_t* codes, const uint16_t* scales,
                          const int32_t* page_table, int physical_pages, int page_size,
                          const int32_t* positions, int count, const float* map,
                          float* output, void* stream);
int lsa_project_queries(const float* query, int count, const float* map,
                        float* output, void* stream);
// Queries and centroids are FP32 storage of BF16-rounded normalized vectors.
// The index covers closed 512-token blocks: 64 buckets/block, 16 members/bucket.
// Members partition their blocks, are ordered, unique and -1 padded. Selection
// masks each row by its device position; equal packed score bits resolve toward
// lower bucket IDs. Signed zeros follow the pinned upstream packed-key ordering.
// Capacity <= 16384 buckets (131072 context), <= 128 queries. top_p is 385.
// Single-row calls sort only the device-position-visible prefix; the buffers
// and temporary-storage requirement still use buckets capacity. Multirow calls
// retain their capacity-strided segments. Empty visible prefixes are valid.
int lsa_select_temp_bytes(int queries, int buckets, size_t* bytes);
int lsa_select(const float* query, const float* centroids, const int32_t* members,
               const int32_t* positions, int queries, int buckets,
               int64_t* keys_in, int64_t* keys_out,
               int32_t* offsets, void* temporary, size_t temporary_bytes,
               int32_t* selected, int32_t* cells, int32_t* steps, void* stream);
// Scratch floats per query for Strata's unchanged native paged INT8 attention.
size_t lsa_attention_scratch_floats(void);
int lsa_attention(const float* query, const int8_t* k, const int8_t* v,
                  const uint16_t* k_scale, const uint16_t* v_scale,
                  const int32_t* page_table, int page_size, int queries,
                  const int32_t* cells, const int32_t* steps,
                  float* scratch, float* output, void* stream);
#ifdef __cplusplus
}
#endif
