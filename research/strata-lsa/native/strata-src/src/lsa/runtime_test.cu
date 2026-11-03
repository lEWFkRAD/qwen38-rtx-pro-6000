#include "strata/lsa/runtime.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int C = 2048, PAGE = 4, CAP = 2051, QROW = 24 * 256;
int checks = 0;

void demand(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
void cuda_ok(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
template<class T> T* allocate(size_t count) {
    T* pointer = nullptr;
    cuda_ok(cudaMalloc((void**) &pointer, count * sizeof(T)), "test allocation");
    return pointer;
}

struct Fixture {
    strata::lsa::RuntimeState state;
    strata::lsa::RuntimePrefillScratch prefill;
    cudaStream_t stream = nullptr;
    void* arena = nullptr;
    void* prefill_arena = nullptr;
    int8_t* codes = nullptr;
    uint16_t* scales = nullptr;
    float* queries = nullptr;
    int32_t* steps = nullptr;
    int32_t* ids = nullptr;
    int32_t* page_table = nullptr;
    std::vector<int8_t> host_codes;
    std::vector<uint16_t> host_scales;
    std::vector<float> host_queries;
    std::vector<int32_t> host_steps;
    int layer;
    uint32_t seed;

    Fixture(int model_layer, uint32_t initial_seed) :
        host_codes(size_t(C) * 2 * 256), host_scales(size_t(C) * 2 * 4, 0x3000),
        host_queries(size_t(C) * QROW), host_steps(size_t(C) * 4), layer(model_layer), seed(initial_seed) {
        cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "test stream");
        codes = allocate<int8_t>(host_codes.size());
        scales = allocate<uint16_t>(host_scales.size());
        queries = allocate<float>(host_queries.size());
        steps = allocate<int32_t>(host_steps.size());
        ids = allocate<int32_t>(size_t(C) * CAP);
        page_table = allocate<int32_t>(C / PAGE);
        const size_t bytes = strata::lsa::runtime_state_bytes(C);
        demand(bytes > 0, "runtime did not price its arena");
        arena = allocate<uint8_t>(bytes);
        std::string error;
        demand(strata::lsa::runtime_state_init(state, C, C / PAGE, PAGE, arena, bytes, error), error);
        state.model_layer = layer;
        const size_t prefill_bytes = strata::lsa::runtime_prefill_scratch_bytes(C);
        demand(prefill_bytes > 0, "runtime did not price batched prefill workspace");
        prefill_arena = allocate<uint8_t>(prefill_bytes);
        demand(strata::lsa::runtime_prefill_scratch_init(prefill, C, 128, prefill_arena, prefill_bytes, error), error);
        std::vector<int32_t> table(C / PAGE);
        for (int p = 0; p < C / PAGE; ++p) table[p] = p;
        cuda_ok(cudaMemcpy(page_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice), "page table upload");
        for (int p = 0; p < C; ++p) {
            host_steps[p * 4] = p; host_steps[p * 4 + 1] = p + 1;
            host_steps[p * 4 + 2] = (p + 1) / 4;
            host_steps[p * 4 + 3] = std::min(p + 1, CAP);
        }
        fill_from(0, initial_seed);
        upload();
        cuda_ok(cudaMemcpy(steps, host_steps.data(), host_steps.size() * 4, cudaMemcpyHostToDevice), "steps upload");
        cuda_ok(cudaMemset(ids, 0xff, size_t(C) * CAP * 4), "selection initialization");
    }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
    ~Fixture() {
        if (stream) cudaStreamSynchronize(stream);
        for (void* p : {arena, prefill_arena, (void*) codes, (void*) scales, (void*) queries, (void*) steps,
                        (void*) ids, (void*) page_table}) if (p) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
    void fill_from(int start, uint32_t salt) {
        for (int p = start; p < C; ++p) {
            for (int h = 0; h < 2; ++h)
                for (int d = 0; d < 256; ++d) {
                    const size_t i = ((size_t(p / PAGE) * 2 + h) * PAGE + p % PAGE) * 256 + d;
                    host_codes[i] = int8_t(int(mix(salt ^ uint32_t(p * 769 + h * 263 + d)) % 127) - 63);
                }
            for (int h = 0; h < 24; ++h)
                for (int d = 0; d < 256; ++d) {
                    const uint32_t v = mix((salt + 17) ^ uint32_t(p * 7919 + h * 257 + d));
                    host_queries[size_t(p) * QROW + h * 256 + d] = (int(v % 1025) - 512) * (1.0f / 4096);
                }
        }
    }
    void upload() {
        cuda_ok(cudaMemcpyAsync(codes, host_codes.data(), host_codes.size(), cudaMemcpyHostToDevice, stream), "K upload");
        cuda_ok(cudaMemcpyAsync(scales, host_scales.data(), host_scales.size() * 2, cudaMemcpyHostToDevice, stream), "scale upload");
        cuda_ok(cudaMemcpyAsync(queries, host_queries.data(), host_queries.size() * 4, cudaMemcpyHostToDevice, stream), "Q upload");
        cuda_ok(cudaStreamSynchronize(stream), "input upload synchronization");
    }
    void same_inputs(const Fixture& other) {
        host_codes = other.host_codes;
        host_scales = other.host_scales;
        host_queries = other.host_queries;
        upload();
    }
    strata::kernels::QsaAttnPools pools() const {
        strata::kernels::QsaAttnPools p;
        p.k_q = codes; p.k_scale = scales; p.page_table = page_table;
        return p;
    }
    void prompt(int start, int count, bool batched = false) {
        std::string error;
        demand(strata::lsa::runtime_prefill(state, layer, pools(), queries + size_t(start) * QROW,
            start, count, steps + start * 4, 4, ids + size_t(start) * CAP, CAP, stream, error,
            batched ? &prefill : nullptr), error);
    }
    void step(int position) {
        std::string error;
        demand(strata::lsa::runtime_step(state, layer, pools(), queries + size_t(position) * QROW,
            steps + position * 4, ids + size_t(position) * CAP, CAP, stream, error), error);
    }
    void sync() {
        cuda_ok(cudaStreamSynchronize(stream), "runtime test synchronization");
        std::string error;
        demand(strata::lsa::runtime_check(state, error), error);
    }
    void rebuild(int tokens) {
        std::string error;
        demand(strata::lsa::runtime_rebuild(state, pools(), tokens, error), error);
    }
};

std::vector<uint8_t> read_bytes(const void* source, size_t bytes) {
    std::vector<uint8_t> result(bytes);
    if (bytes) cuda_ok(cudaMemcpy(result.data(), source, bytes, cudaMemcpyDeviceToHost), "test readback");
    return result;
}
void equal_bytes(const void* a, const void* b, size_t bytes, const char* what) {
    demand(read_bytes(a, bytes) == read_bytes(b, bytes), what);
}
void equal_cache(Fixture& a, Fixture& b, int tokens) {
    a.sync(); b.sync();
    equal_bytes(a.state.history, b.state.history, size_t(tokens) * 64 * 4, "projected prefix differs from reconstruction");
    const size_t buckets = size_t(tokens / 512) * 64;
    equal_bytes(a.state.centroids, b.state.centroids, buckets * 64 * 4, "closed centroids differ from reconstruction");
    equal_bytes(a.state.members, b.state.members, buckets * 16 * 4, "closed members differ from reconstruction");
}
void equal_selection(Fixture& a, Fixture& b, int first, int count) {
    a.sync(); b.sync();
    equal_bytes(a.ids + size_t(first) * CAP, b.ids + size_t(first) * CAP,
                size_t(count) * CAP * 4, "selected token IDs differ");
    equal_bytes(a.steps + first * 4, b.steps + first * 4, size_t(count) * 4 * 4,
                "causal count or published selection width differs");
}

void split_prefill(int layer) {
    Fixture whole(layer, 41), batched(layer, 41), split(layer, 41);
    whole.prompt(0, C);
    batched.prompt(0, C, true);
    equal_cache(whole, batched, C);
    equal_selection(whole, batched, 0, C);
    split.prompt(0, 511, true);
    split.prompt(511, 1, true);
    split.prompt(512, 1, true);
    split.prompt(513, 510, true);
    split.prompt(1023, 1, true);
    split.prompt(1024, C - 1024, true);
    equal_cache(batched, split, C);
    equal_selection(batched, split, 0, C);
    const auto packed = read_bytes(whole.steps + (C - 1) * 4, 4 * 4);
    int32_t final_step[4];
    std::memcpy(final_step, packed.data(), sizeof(final_step));
    demand(final_step[3] == 1540, "mature runtime selection did not enforce its long-range budget");
    const auto selected = read_bytes(whole.ids + size_t(C - 1) * CAP, 1540 * 4);
    std::vector<int32_t> cells(1540);
    std::memcpy(cells.data(), selected.data(), selected.size());
    std::sort(cells.begin(), cells.end());
    demand(cells.front() >= 0 && cells.back() < C &&
            std::adjacent_find(cells.begin(), cells.end()) == cells.end(),
            "mature runtime selection contains invalid or duplicate cells");
    for (int p = 0; p < 4; ++p)
        demand(std::binary_search(cells.begin(), cells.end(), p), "mature runtime selection dropped a sink");
    for (int p = C - 512; p < C; ++p)
        demand(std::binary_search(cells.begin(), cells.end(), p), "mature runtime selection dropped a recent-window token");
}

void rejected_replacement(int layer, int base, int accepted) {
    Fixture candidate(layer, 83), rebuilt(layer, 83);
    candidate.prompt(0, base);
    for (int p = base; p < base + 8; ++p) candidate.step(p);
    candidate.sync();
    const int retained = base + accepted;
    // Caller acceptance leaves only this prefix semantically live. Rejected
    // physical rows and closed blocks remain until corrected writes replace them.
    candidate.fill_from(retained, 901);
    candidate.upload();
    rebuilt.same_inputs(candidate);
    candidate.prompt(retained, 1025 - retained);
    rebuilt.rebuild(1025);
    equal_cache(candidate, rebuilt, 1025);
    rebuilt.step(1024);
    equal_selection(candidate, rebuilt, 1024, 1);
}

void copy_and_interleave(int layer) {
    Fixture a(layer, 109), b(layer, 211), copied(layer, 109), rebuilt(layer, 109);
    a.prompt(0, 511); b.prompt(0, 511);
    a.step(511); b.step(511); a.step(512); b.step(512);
    a.sync(); b.sync();
    rebuilt.rebuild(513);
    equal_cache(a, rebuilt, 513);
    Fixture b_rebuilt(layer, 211);
    b_rebuilt.rebuild(513);
    equal_cache(b, b_rebuilt, 513);
    demand(read_bytes(a.state.history, 513 * 64 * 4) != read_bytes(b.state.history, 513 * 64 * 4),
            "independent slots accidentally share projected keys");
    // The retained KV is copied independently, as the engine's slot move does.
    cuda_ok(cudaMemcpyAsync(copied.codes, a.codes, a.host_codes.size(), cudaMemcpyDeviceToDevice, copied.stream), "copied KV");
    cuda_ok(cudaMemcpyAsync(copied.scales, a.scales, a.host_scales.size() * 2,
                            cudaMemcpyDeviceToDevice, copied.stream), "copied scales");
    std::string error;
    demand(strata::lsa::runtime_copy(a.state, copied.state, 513, copied.stream, error), error);
    equal_cache(a, copied, 513);
    copied.step(513);
    rebuilt.rebuild(514);
    equal_cache(copied, rebuilt, 514);
    rebuilt.step(513);
    equal_selection(copied, rebuilt, 513, 1);
}

void graph_replay_positions(int layer) {
    Fixture captured(layer, 313), rebuilt(layer, 313);
    captured.prompt(0, 511);
    captured.sync();
    float* query_row = allocate<float>(QROW);
    int32_t* step_row = allocate<int32_t>(4);
    int32_t* selected_row = allocate<int32_t>(CAP);
    cuda_ok(cudaMemcpyAsync(query_row, captured.queries + size_t(511) * QROW, QROW * 4,
                            cudaMemcpyDeviceToDevice, captured.stream), "graph query setup");
    cuda_ok(cudaMemcpyAsync(step_row, captured.host_steps.data() + 511 * 4, 4 * 4,
                            cudaMemcpyHostToDevice, captured.stream), "graph position setup");
    cuda_ok(cudaStreamSynchronize(captured.stream), "graph input synchronization");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_ok(cudaStreamBeginCapture(captured.stream, cudaStreamCaptureModeThreadLocal), "runtime graph begin");
    std::string error;
    demand(strata::lsa::runtime_step(captured.state, layer, captured.pools(), query_row,
                                     step_row, selected_row, CAP, captured.stream, error), error);
    cuda_ok(cudaStreamEndCapture(captured.stream, &graph), "runtime graph end");
    cuda_ok(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "runtime graph instantiate");
    for (int p = 511; p <= 1024; ++p) {
        cuda_ok(cudaMemcpyAsync(query_row, captured.queries + size_t(p) * QROW, QROW * 4,
                                cudaMemcpyDeviceToDevice, captured.stream), "graph query change");
        cuda_ok(cudaMemcpyAsync(step_row, captured.host_steps.data() + p * 4, 4 * 4,
                                cudaMemcpyHostToDevice, captured.stream), "graph position change");
        cuda_ok(cudaGraphLaunch(executable, captured.stream), "runtime graph replay");
    }
    captured.sync();
    rebuilt.rebuild(1025);
    equal_cache(captured, rebuilt, 1025);
    rebuilt.step(1024);
    rebuilt.sync();
    equal_bytes(selected_row, rebuilt.ids + size_t(1024) * CAP, CAP * 4, "graph dynamic position selected wrong keys");
    equal_bytes(step_row, rebuilt.steps + 1024 * 4, 4 * 4, "graph dynamic position froze causal counts");
    cuda_ok(cudaGraphExecDestroy(executable), "runtime graph exec release");
    cuda_ok(cudaGraphDestroy(graph), "runtime graph release");
    cudaFree(query_row); cudaFree(step_row); cudaFree(selected_row);
}
} // namespace

int main() {
    try {
        std::string error;
        demand(strata::lsa::runtime_configure_from_env(error), error);
        demand(strata::lsa::runtime_enabled(), "set STRATA_LSA_MAPS to the pinned native artifact");
        for (int layer : {3, 47}) {
            split_prefill(layer);
            rejected_replacement(layer, 511, 0);
            rejected_replacement(layer, 509, 3);
        }
        copy_and_interleave(3);
        graph_replay_positions(47);
        std::cout << "{\"status\":\"passed\",\"scope\":\"native runtime state, rollback, copy, rebuild and graph correctness\","
                     "\"layers\":[3,47],\"context\":2048,\"prefill\":\"batched and per-row bit-exact\",\"checks\":" << checks << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LSA runtime test failed after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
