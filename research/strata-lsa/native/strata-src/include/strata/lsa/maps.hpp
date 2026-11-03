#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::lsa {

inline constexpr uint32_t kInputDim = 384;
inline constexpr uint32_t kIndexDim = 64;
inline constexpr uint32_t kLayerCount = 12;
inline constexpr uint32_t kMapVersion = 1;
inline constexpr uint64_t kMapHeaderBytes = 96;
inline constexpr uint64_t kMatrixValues = uint64_t{kInputDim} * kIndexDim;
inline constexpr uint64_t kMapRecordBytes = 16 + 2 * kMatrixValues * 4;
inline constexpr uint64_t kMapFileBytes = kMapHeaderBytes + kLayerCount * kMapRecordBytes;

// Row-major [384,64]. Inputs are two concatenated, 192-coordinate nonrotary
// key heads, or sums of each head's twelve grouped queries. Temperature is
// separate from the maps; projected rows are normalized by the indexer.
struct LayerMap {
    int32_t layer = -1;
    float temperature = 0;
    std::vector<float> query;
    std::vector<float> key;
};

struct Maps {
    std::array<LayerMap, kLayerCount> layers;
    // Nonserialized cache identity over every exact byte validated by load_maps.
    // FNV-1a 64 matches the existing 64-bit identity strength; not authentication.
    uint64_t artifact_fingerprint = 0;
    // Provenance identifiers of the original .pt and JSON files. They do not
    // authenticate this file: the caller must pin the native file's SHA-256.
    std::array<uint8_t, 32> source_weights_sha256{};
    std::array<uint8_t, 32> source_metadata_sha256{};
};

// An explicitly invoked, CPU-only loader. No process-global state, allocation
// on a device, or serving/default-path change. Failure leaves `out` unchanged.
// The native format is fixed little-endian IEEE754 binary32, version 1.
bool load_maps(const std::string& path, Maps& out, std::string& error);

}  // namespace strata::lsa
