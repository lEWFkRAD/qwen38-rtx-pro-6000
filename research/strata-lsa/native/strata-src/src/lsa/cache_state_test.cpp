#include "strata/lsa/cache_state.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using State = strata::lsa::CacheState;
namespace {
int checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template <class Exception, class F> void throws(F fn, const char* message) {
    ++checks;
    try { fn(); }
    catch (const Exception&) { return; }
    throw std::runtime_error(message);
}

void append_all(State& state, State::Ticket ticket, uint32_t count) {
    for (uint32_t l = 3; l <= 47; l += 4) state.append(ticket, l, count);
}

void seed(State& state, uint32_t slot, uint32_t tokens) {
    auto ticket = state.begin(slot);
    append_all(state, ticket, tokens);
    state.accept_prefix(ticket, tokens);
}

void block_boundaries() {
    State state;
    auto first = state.begin(0);
    append_all(state, first, 511);
    require(state.published_blocks(0) == 0, "uncommitted 511 tokens were published");
    auto view = state.view(first, 3, 510);
    require(view.visible_tokens == 511 && view.complete_blocks == 0 && view.tail_tokens == 511,
            "511-token tail was miscounted");
    throws<std::out_of_range>([&] { state.view(0, 3, 0); }, "ordinary view saw speculative KV");
    state.accept_prefix(first, 511);
    auto second = state.begin(0);
    auto range = state.append(second, 3, 1);
    require(range.begin_token == 511 && range.end_token == 512 && range.first_new_block == 0 &&
            range.end_complete_block == 1, "512th token did not complete block zero");
    view = state.view(second, 3, 511);
    require(view.complete_blocks == 1 && view.provisional_blocks == 1 && view.published_blocks == 0,
            "newly completed block escaped speculative quarantine");
    for (uint32_t l = 7; l <= 47; l += 4) state.append(second, l, 1);
    auto commit = state.accept_prefix(second, 1);
    require(commit.published_blocks_before == 0 && commit.published_blocks_after == 1 &&
            commit.discard_from_block == 1, "completed accepted block was not retained");
    auto third = state.begin(0);
    append_all(state, third, 1);
    view = state.view(third, 47, 512);
    require(view.visible_tokens == 513 && view.tail_begin == 512 && view.tail_tokens == 1 &&
            view.published_blocks == 1 && view.provisional_blocks == 0 && view.long_range_end == 1,
            "513-token view used wrong causal or local-window bounds");
    state.reject(third);
    require(state.committed_tokens(0) == 512 && state.published_blocks(0) == 1,
            "reject after boundary changed committed blocks");
}

void partial_accept_and_replay() {
    State state;
    seed(state, 0, 509);
    auto candidate = state.begin(0);
    append_all(state, candidate, 8);
    auto plan = state.accept_prefix(candidate, 3);
    require(plan.old_tokens == 509 && plan.new_tokens == 512 && plan.discard_from_token == 512 &&
            plan.discard_from_block == 1 && plan.published_blocks_after == 1,
            "partial acceptance retained rejected token or block state");
    throws<std::logic_error>([&] { state.available_tokens(candidate, 3); }, "accepted ticket remained valid");
    throws<std::out_of_range>([&] { state.view(0, 3, 512); }, "rejected suffix was visible after acceptance");
    auto replay = state.begin(0);
    auto range = state.append(replay, 3, 1);
    require(range.begin_token == 512 && range.end_token == 513, "replay appended past rejected suffix");
    require(state.available_tokens(replay, 3) == 513 && state.available_tokens(replay, 7) == 512,
            "a layer saw another layer's unexecuted append");
    throws<std::out_of_range>([&] { state.view(replay, 7, 512); }, "later layer saw unappended KV");
    auto reject = state.reject(replay); // intentionally incomplete per-layer forward
    require(reject.new_tokens == 512 && !state.transaction_open(0), "incomplete forward could not be rejected");
    throws<std::logic_error>([&] { state.append(replay, 3, 1); }, "rejected ticket allowed a later append");
}

void causal_queries_with_chunk_appended_first() {
    State state;
    auto ticket = state.begin(0);
    append_all(state, ticket, 4096);
    auto early = state.view(ticket, 27, 600);
    auto late = state.view(ticket, 27, 4095);
    require(early.visible_tokens == 601 && early.complete_blocks == 1 && early.provisional_blocks == 1 &&
            early.long_range_end == 89, "prefill query saw future chunk keys");
    require(late.visible_tokens == 4096 && late.complete_blocks == 8 && late.long_range_end == 3584 &&
            late.long_range_complete_blocks == 7, "large speculative chunk lost its causal provisional view");
    state.accept_prefix(ticket, 513);
    auto next = state.begin(0);
    append_all(state, next, 1);
    auto now = state.view(next, 27, 513);
    require(now.complete_blocks == 1 && now.tail_tokens == 2 && now.published_blocks == 1,
            "accepted prefix inherited discarded speculative clusters");
    throws<std::logic_error>([&] { state.view(ticket, 27, 4095); }, "old chunk ticket regained access to rejected KV");
}

void selector_visibility_at_block_edges() {
    State state;
    auto ticket = state.begin(0);
    append_all(state, ticket, 2048);
    struct Boundary {
        uint32_t position;
        uint32_t complete;
        uint32_t selectable;
        uint32_t long_end;
        uint32_t wholly_long;
    };
    // The expected selector bounds are fixed examples of the upstream contract,
    // including a completed current block and a block crossing the local window.
    for (const Boundary& b : {Boundary{511, 1, 0, 0, 0}, {512, 1, 1, 1, 0},
                              {513, 1, 1, 2, 0}, {1023, 2, 1, 512, 1},
                              {1024, 2, 2, 513, 1}, {1535, 3, 2, 1024, 2},
                              {1536, 3, 3, 1025, 2}}) {
        const auto staged = state.view(ticket, 3, b.position);
        require(staged.complete_blocks == b.complete && staged.selectable_blocks == b.selectable,
                "selector confused completed KV with rankable blocks");
        require(staged.long_range_end == b.long_end && staged.long_range_complete_blocks == b.wholly_long,
                "selector lost the partially visible block at a local-window edge");
        require(staged.published_blocks == 0 && staged.provisional_blocks == b.complete,
                "selector visibility prematurely published speculative blocks");
    }
    state.accept_prefix(ticket, 2048);
    const auto committed = state.view(0, 3, 1536);
    require(committed.selectable_blocks == 3 && committed.long_range_complete_blocks == 2 &&
            committed.long_range_end == 1025 && committed.published_blocks == 3,
            "committed view suppressed the ranked block containing long-range key 1024");
}

void prefix_reuse_invalidates_partial_block() {
    State state;
    seed(state, 2, 1537);
    auto ticket = state.begin(2);
    append_all(state, ticket, 16);
    auto trim = state.truncate(2, 511);
    require(trim.old_tokens == 1537 && trim.new_tokens == 511 && trim.discard_from_token == 511 &&
            trim.discard_from_block == 0 && state.published_blocks(2) == 0,
            "trim kept a block with a discarded suffix");
    throws<std::logic_error>([&] { state.append(ticket, 3, 1); }, "prefix trim did not invalidate open ticket");
    auto next = state.begin(2);
    append_all(state, next, 513);
    auto commit = state.accept_prefix(next, 513);
    require(commit.new_tokens == 1024 && commit.published_blocks_after == 2,
            "reuse failed to rebuild boundary-spanning blocks");
    trim = state.truncate(2, 512);
    require(trim.discard_from_block == 1 && state.published_blocks(2) == 1,
            "exact-block trim discarded or retained wrong block");
    auto revision = state.generation(2);
    state.reset(2);
    require(state.committed_tokens(2) == 0 && state.generation(2) > revision,
            "reset did not invalidate sequence cache identity");
}

void slots_and_layers_remain_independent() {
    State state;
    for (uint32_t slot = 0; slot < State::kSlots; ++slot) seed(state, slot, slot * 512 + 1);
    auto a = state.begin(0);
    auto b = state.begin(7);
    append_all(state, a, 4);
    state.append(b, 47, 8);
    require(state.available_tokens(b, 47) == 3593 && state.available_tokens(b, 3) == 3585,
            "per-layer stage state leaked");
    state.reject(a);
    require(state.transaction_open(7) && state.available_tokens(b, 47) == 3593,
            "one slot's rejection invalidated another slot");
    for (uint32_t slot = 1; slot < 7; ++slot)
        require(state.committed_tokens(slot) == slot * 512 + 1,
                "transaction changed an unrelated committed slot");
    state.reset(0);
    require(state.available_tokens(b, 47) == 3593, "one slot's reset invalidated another slot");
    state.reject(b);
}

void invalid_mutations_are_atomic() {
    State state;
    auto ticket = state.begin(0);
    throws<std::logic_error>([&] { state.begin(0); }, "nested transaction accepted");
    state.append(ticket, 3, 2);
    throws<std::logic_error>([&] { state.accept_prefix(ticket, 1); }, "incomplete forward committed");
    require(state.transaction_open(0) && state.committed_tokens(0) == 0 &&
            state.available_tokens(ticket, 3) == 2, "failed commit mutated slot");
    for (uint32_t l = 7; l <= 47; l += 4) state.append(ticket, l, 2);
    throws<std::out_of_range>([&] { state.accept_prefix(ticket, 3); }, "accepted unavailable tokens");
    throws<std::out_of_range>([&] { state.truncate(0, 1); }, "prefix reuse invented committed KV");
    require(state.transaction_open(0) && state.available_tokens(ticket, 3) == 2,
            "failed truncate changed live transaction");
    State::Ticket forged = ticket;
    ++forged.base_tokens;
    throws<std::logic_error>([&] { state.append(forged, 3, 1); }, "ticket with wrong prefix was accepted");
    state.reject(ticket);
    auto fresh = state.begin(0);
    require(fresh.generation != ticket.generation, "new transaction reused old ticket generation");
    throws<std::logic_error>([&] { state.reject(ticket); }, "old ticket rejected a newer transaction");
    require(state.transaction_open(0), "stale mutation terminated new transaction");
    state.reject(fresh);
}

void limits_and_layer_geometry() {
    State state;
    require(State::kInputDim == 384 && State::kIndexDim == 64 && State::kLayers == 12 && State::kSlots == 8,
            "learned-map geometry drifted");
    for (uint32_t ordinal = 0; ordinal < 12; ++ordinal)
        require(State::layer_ordinal(3 + 4 * ordinal) == ordinal, "QSA layer ordinal mismatch");
    for (uint32_t bad : {0u, 2u, 4u, 48u, UINT32_MAX})
        throws<std::invalid_argument>([&] { State::layer_ordinal(bad); }, "non-QSA layer accepted");
    throws<std::out_of_range>([&] { state.begin(8); }, "ninth slot accepted");
    seed(state, 0, State::kContext - 1);
    auto ticket = state.begin(0);
    throws<std::out_of_range>([&] { state.append(ticket, 3, UINT32_MAX); }, "overflow append accepted");
    require(state.available_tokens(ticket, 3) == State::kContext - 1, "overflow append mutated count");
    append_all(state, ticket, 1);
    state.accept_prefix(ticket, 1);
    require(state.committed_tokens(0) == State::kContext && state.published_blocks(0) == 256,
            "full context was not representable");
    auto full = state.begin(0);
    throws<std::out_of_range>([&] { state.append(full, 3, 1); }, "context overflow accepted");
    require(state.view(0, 47, State::kContext - 1).visible_tokens == State::kContext,
            "last context token was not visible");
    throws<std::out_of_range>([&] { state.view(full, 47, State::kContext); }, "past-capacity query accepted");
    state.reject(full);
}
} // namespace

int main() {
    try {
        block_boundaries();
        partial_accept_and_replay();
        causal_queries_with_chunk_appended_first();
        selector_visibility_at_block_edges();
        prefix_reuse_invalidates_partial_block();
        slots_and_layers_remain_independent();
        invalid_mutations_are_atomic();
        limits_and_layer_geometry();
        std::cout << "LSA cache state: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LSA cache state failed after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
