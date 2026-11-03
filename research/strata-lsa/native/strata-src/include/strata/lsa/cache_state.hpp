#pragma once

#include <array>
#include <cstdint>

namespace strata::lsa {

// Metadata contract for the experimental learned-map cache. This class owns no
// device buffers and is not wired into Strata's sessions. The adapter must apply
// each returned invalidation to its own projected keys, clusters and members
// before using a slot again. Calls are serialized by the session owner.
//
// The pilot maps consume the trailing 192 nonrotary coordinates of each of two
// 256-wide KV heads, concatenated in head order. Keys are dequantized INT8-cache
// values. Queries sum the 12 matching Q heads within each KV group before the
// same concatenation. The separate query/key 384x64 maps project these inputs;
// normalize each projected row. Temperature remains separate from both maps.
class CacheState {
public:
    static constexpr uint32_t kContext = 131072;
    static constexpr uint32_t kSlots = 8;
    static constexpr uint32_t kLayers = 12;
    static constexpr uint32_t kInputDim = 384;
    static constexpr uint32_t kIndexDim = 64;
    static constexpr uint32_t kBlockTokens = 512;
    static constexpr uint32_t kWindowTokens = 512;
    static constexpr uint32_t kSinkTokens = 4;

    struct Ticket {
        uint32_t slot = 0;
        uint64_t generation = 0;
        uint32_t base_tokens = 0;
    };

    struct AppendRange {
        uint32_t begin_token;
        uint32_t end_token;          // exclusive
        uint32_t first_new_block;    // previously incomplete block
        uint32_t end_complete_block; // exclusive; provisional until commit
    };

    struct Transition {
        uint32_t old_tokens;
        uint32_t new_tokens;
        uint32_t published_blocks_before;
        uint32_t published_blocks_after;
        uint32_t discard_from_token;
        uint32_t discard_from_block;
        uint64_t generation;
    };

    struct View {
        uint32_t visible_tokens; // exactly query_position + 1, never future KV
        uint32_t complete_blocks;
        // Rank only these blocks (64 buckets per block): floor(position / 512).
        // The block containing this query remains unranked even when its final
        // token has just made it complete. This is not complete_blocks, nor the
        // number of blocks wholly before the local window. Expansion still
        // filters each selected member by long_range_end; at position 1536 the
        // third ranked block can contribute key 1024 across that window edge.
        uint32_t selectable_blocks;
        uint32_t published_blocks;
        uint32_t provisional_blocks;
        uint32_t tail_begin;
        uint32_t tail_tokens;
        uint32_t long_range_end; // exclusive; expansion must filter by this
        uint32_t long_range_complete_blocks;
    };

    static uint32_t layer_ordinal(uint32_t model_layer);

    // begin never publishes speculative data. Only the matching live ticket can
    // observe its appends; the ordinary slot view remains at committed_tokens.
    Ticket begin(uint32_t slot);
    AppendRange append(const Ticket& ticket, uint32_t model_layer, uint32_t count);

    // All layers must have appended the same speculative length before a commit.
    // Only the accepted prefix is published. Everything after it is invalidated,
    // including a partly accepted 512-token block's provisional cluster state.
    Transition accept_prefix(const Ticket& ticket, uint32_t accepted_tokens);
    Transition reject(const Ticket& ticket); // also accepts an incomplete forward

    // Reuse of the SAME stored KV prefix, not a declaration that another prompt
    // matches it. Also abandons any open transaction and invalidates its ticket.
    // Dropping a partial block invalidates that block and every later block.
    Transition truncate(uint32_t slot, uint32_t retained_tokens);
    Transition reset(uint32_t slot);

    uint32_t committed_tokens(uint32_t slot) const;
    uint32_t available_tokens(const Ticket& ticket, uint32_t model_layer) const;
    uint32_t published_blocks(uint32_t slot) const;
    bool transaction_open(uint32_t slot) const;
    uint64_t generation(uint32_t slot) const;

    View view(uint32_t slot, uint32_t model_layer, uint32_t query_position) const;
    View view(const Ticket& ticket, uint32_t model_layer, uint32_t query_position) const;

private:
    struct Slot {
        uint32_t committed = 0;
        uint64_t generation = 0;
        bool open = false;
        std::array<uint32_t, kLayers> appended{};
    };
    std::array<Slot, kSlots> slots_{};

    Slot& slot_at(uint32_t slot);
    const Slot& slot_at(uint32_t slot) const;
    Slot& live(const Ticket& ticket);
    const Slot& live(const Ticket& ticket) const;
    static View make_view(uint32_t committed, uint32_t available, uint32_t query_position);
    static void bump(Slot& slot);
    static Transition finish(Slot& slot, uint32_t new_tokens);
};

} // namespace strata::lsa
