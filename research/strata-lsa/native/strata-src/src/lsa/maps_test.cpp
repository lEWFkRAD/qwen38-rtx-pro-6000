// CPU tests for the explicit learned-map interchange. No engine or CUDA calls.
#include "strata/lsa/maps.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace strata::lsa;

namespace {
int failed = 0;
int checks = 0;

void check(bool ok, const char* name) {
    ++checks;
    if (!ok) { ++failed; std::fprintf(stderr, "FAIL %s\n", name); }
}

void put32(std::vector<uint8_t>& v, size_t offset, uint32_t n) {
    for (int i = 0; i < 4; ++i) v[offset + i] = static_cast<uint8_t>(n >> (i * 8));
}

void putf(std::vector<uint8_t>& v, size_t offset, float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    put32(v, offset, u);
}

std::vector<uint8_t> fixture() {
    std::vector<uint8_t> v(static_cast<size_t>(kMapFileBytes));
    std::memcpy(v.data(), "STLSAM01", 8);
    put32(v, 8, 1); put32(v, 12, kInputDim); put32(v, 16, kIndexDim); put32(v, 20, kLayerCount);
    const uint64_t payload = kMapRecordBytes * kLayerCount;
    put32(v, 24, static_cast<uint32_t>(payload)); put32(v, 28, static_cast<uint32_t>(payload >> 32));
    for (size_t j = 32; j < 96; ++j) v[j] = static_cast<uint8_t>(j);
    for (uint32_t i = 0; i < kLayerCount; ++i) {
        const size_t off = static_cast<size_t>(kMapHeaderBytes + i * kMapRecordBytes);
        put32(v, off, 3 + 4 * i); putf(v, off + 8, 0.05f + i * 0.001f);
        for (size_t j = 0; j < 2 * kMatrixValues; ++j) putf(v, off + 16 + j * 4, float(int(j % 127) - 63) / 128);
    }
    return v;
}

bool read_fixture(const std::string& path, const std::vector<uint8_t>& bytes, Maps& maps, std::string& error) {
    { std::ofstream f(path, std::ios::binary); f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()); }
    return load_maps(path, maps, error);
}

void rejection(const std::string& path, const std::vector<uint8_t>& bytes, const char* name) {
    Maps maps;
    maps.layers[0].layer = 777;
    maps.layers[0].query = {123.0f};
    maps.artifact_fingerprint = 456;
    std::string error;
    check(!read_fixture(path, bytes, maps, error) && !error.empty(), name);
    check(maps.layers[0].layer == 777 && maps.layers[0].query == std::vector<float>{123.0f}
          && maps.artifact_fingerprint == 456, "failure is transactional");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: lsa_maps_test writable-fixture-path [exported-native-map]\n");
        return 2;
    }
    const std::string path = argv[1];
    auto bytes = fixture();
    Maps maps;
    std::string error = "stale";
    check(read_fixture(path, bytes, maps, error) && error.empty(), "valid file");
    check(maps.layers[0].layer == 3 && maps.layers[11].layer == 47, "complete ordered layers");
    check(maps.layers[0].temperature == 0.05f && maps.layers[0].query[0] == -63.0f / 128, "float values exact");
    check(maps.layers[11].key.size() == kMatrixValues && maps.source_weights_sha256[0] == 32
          && maps.source_metadata_sha256[31] == 95, "matrices and provenance");
    const uint64_t identity = maps.artifact_fingerprint;
    check(identity != 0, "full artifact identity exists");
    Maps identical;
    check(read_fixture(path, bytes, identical, error) && identical.artifact_fingerprint == identity,
          "unchanged bytes preserve identity");
    // This float lies in the middle gap omitted by the legacy 1 MiB head/tail
    // sampler. Keep geometry, provenance, length and finite-value validity.
    const size_t middle = (bytes.size() / 2) & ~size_t(3);
    check(middle >= (1u << 20) && middle + 4 <= bytes.size() - (1u << 20), "middle edit lies outside sampled ends");
    auto changed = bytes;
    putf(changed, middle, 1.25f);
    Maps edited;
    check(read_fixture(path, changed, edited, error) && edited.artifact_fingerprint != identity,
          "valid finite middle edit changes full artifact identity");
    check(edited.source_weights_sha256 == maps.source_weights_sha256 &&
          edited.source_metadata_sha256 == maps.source_metadata_sha256, "middle identity change does not depend on provenance changes");
    check(read_fixture(path, bytes, maps, error) && maps.artifact_fingerprint == identity,
          "restored exact bytes restore identity");
    auto bad = bytes; bad[0] ^= 1; rejection(path, bad, "magic");
    bad = bytes; put32(bad, 8, 2); rejection(path, bad, "version");
    bad = bytes; put32(bad, 12, 768); rejection(path, bad, "input dimension");
    bad = bytes; put32(bad, 16, 128); rejection(path, bad, "index dimension");
    bad = bytes; put32(bad, 20, 11); rejection(path, bad, "layer count");
    bad = bytes; put32(bad, 24, 0); rejection(path, bad, "payload length");
    bad = bytes; bad.pop_back(); rejection(path, bad, "truncation");
    bad = bytes; bad.push_back(0); rejection(path, bad, "trailing bytes");
    bad = bytes; put32(bad, kMapHeaderBytes + kMapRecordBytes, 3); rejection(path, bad, "duplicate layer");
    bad = bytes; put32(bad, kMapHeaderBytes, 7); rejection(path, bad, "unordered layer");
    bad = bytes; put32(bad, kMapHeaderBytes + 4, 1); rejection(path, bad, "reserved field");
    bad = bytes; putf(bad, kMapHeaderBytes + 8, 0); rejection(path, bad, "zero temperature");
    bad = bytes; putf(bad, kMapHeaderBytes + 8, -1); rejection(path, bad, "negative temperature");
    bad = bytes; putf(bad, kMapHeaderBytes + 8, std::numeric_limits<float>::infinity()); rejection(path, bad, "infinite temperature");
    bad = bytes; putf(bad, kMapHeaderBytes + 16, std::numeric_limits<float>::quiet_NaN()); rejection(path, bad, "NaN matrix");
    bad = bytes; putf(bad, kMapHeaderBytes + 16 + kMatrixValues * 4, std::numeric_limits<float>::infinity());
    rejection(path, bad, "infinite key matrix");
    std::remove(path.c_str());
    if (argc == 3) {
        check(load_maps(argv[2], maps, error), "actual exported pilot maps load");
        if (error.empty()) {
            std::ifstream input(argv[2], std::ios::binary);
            std::vector<uint8_t> raw(static_cast<size_t>(kMapFileBytes));
            input.read(reinterpret_cast<char*>(raw.data()), raw.size());
            bool exact = true;
            for (uint32_t i = 0; i < kLayerCount; ++i) {
                const size_t off = static_cast<size_t>(kMapHeaderBytes + i * kMapRecordBytes);
                for (const auto* matrix : {&maps.layers[i].query, &maps.layers[i].key}) {
                    const size_t which = matrix == &maps.layers[i].query ? 0 : 1;
                    for (size_t j = 0; j < kMatrixValues; ++j) {
                        const size_t p = off + 16 + which * kMatrixValues * 4 + j * 4;
                        uint32_t bits;
                        std::memcpy(&bits, &(*matrix)[j], 4);
                        for (int k = 0; k < 4; ++k) exact &= raw[p + k] == static_cast<uint8_t>(bits >> (8 * k));
                    }
                }
            }
            check(exact, "all exported coefficients read bit-exactly");
        }
    }
    std::printf("lsa_maps_test: %d checks, %d failed\n", checks, failed);
    return failed ? 1 : 0;
}
