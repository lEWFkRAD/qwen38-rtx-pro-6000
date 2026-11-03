#include "strata/lsa/maps.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

namespace strata::lsa {
namespace {

constexpr uint8_t kMagic[8] = {'S', 'T', 'L', 'S', 'A', 'M', '0', '1'};

uint32_t u32(const uint8_t* p) {
    return uint32_t{p[0]} | (uint32_t{p[1]} << 8) | (uint32_t{p[2]} << 16) | (uint32_t{p[3]} << 24);
}

uint64_t u64(const uint8_t* p) {
    return uint64_t{u32(p)} | (uint64_t{u32(p + 4)} << 32);
}

float f32(const uint8_t* p) {
    const uint32_t bits = u32(p);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool fail(std::string& error, const std::string& message) {
    error = "LSA maps: " + message;
    return false;
}

}  // namespace

bool load_maps(const std::string& path, Maps& out, std::string& error) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "LSA map loader requires IEEE754 binary32");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return fail(error, "cannot open file");
    const std::streampos size = input.tellg();
    // Fixed geometry bounds every allocation and rejects trailing payloads.
    if (size != std::streampos(kMapFileBytes)) return fail(error, "wrong file length (truncated or trailing bytes)");
    input.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(kMapFileBytes));
    if (!input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        return fail(error, "cannot read complete file");
    // A file that grew between sizing and reading is also invalid.
    if (input.peek() != std::char_traits<char>::eof()) return fail(error, "trailing bytes");
    const uint8_t* p = bytes.data();
    if (!std::equal(std::begin(kMagic), std::end(kMagic), p)) return fail(error, "bad magic");
    if (u32(p + 8) != kMapVersion) return fail(error, "unsupported version");
    if (u32(p + 12) != kInputDim || u32(p + 16) != kIndexDim || u32(p + 20) != kLayerCount)
        return fail(error, "wrong dimensions or layer count");
    if (u64(p + 24) != kLayerCount * kMapRecordBytes) return fail(error, "wrong payload length");
    Maps candidate;
    std::copy(p + 32, p + 64, candidate.source_weights_sha256.begin());
    std::copy(p + 64, p + 96, candidate.source_metadata_sha256.begin());
    p += kMapHeaderBytes;
    for (uint32_t i = 0; i < kLayerCount; ++i) {
        auto& layer = candidate.layers[i];
        const uint32_t expected = 3 + 4 * i;
        if (u32(p) != expected) return fail(error, "wrong, missing, duplicated, or unordered layer");
        if (u32(p + 4) != 0 || u32(p + 12) != 0) return fail(error, "nonzero reserved field");
        layer.layer = static_cast<int32_t>(expected);
        layer.temperature = f32(p + 8);
        if (!(std::isfinite(layer.temperature) && layer.temperature > 0))
            return fail(error, "temperature must be positive and finite");
        p += 16;
        for (auto* matrix : {&layer.query, &layer.key}) {
            matrix->resize(static_cast<size_t>(kMatrixValues));
            for (float& value : *matrix) {
                value = f32(p);
                if (!std::isfinite(value)) return fail(error, "matrix contains a nonfinite value");
                p += 4;
            }
        }
    }
    uint64_t fingerprint = 14695981039346656037ULL;
    for (const uint8_t byte : bytes) {
        fingerprint ^= byte;
        fingerprint *= 1099511628211ULL;
    }
    candidate.artifact_fingerprint = fingerprint;
    out = std::move(candidate);
    error.clear();
    return true;
}

}  // namespace strata::lsa
