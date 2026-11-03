#include "strata/lsa/cache_state.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strata::lsa {

uint32_t CacheState::layer_ordinal(uint32_t layer) {
    if (layer < 3 || layer > 47 || (layer - 3) % 4 != 0)
        throw std::invalid_argument("LSA requires QSA model layers 3,7,...,47");
    return (layer - 3) / 4;
}

CacheState::Slot& CacheState::slot_at(uint32_t slot) {
    if (slot >= kSlots) throw std::out_of_range("LSA slot is outside the eight-slot capacity");
    return slots_[slot];
}

const CacheState::Slot& CacheState::slot_at(uint32_t slot) const {
    if (slot >= kSlots) throw std::out_of_range("LSA slot is outside the eight-slot capacity");
    return slots_[slot];
}

const CacheState::Slot& CacheState::live(const Ticket& ticket) const {
    const Slot& slot = slot_at(ticket.slot);
    if (!slot.open || slot.generation != ticket.generation || slot.committed != ticket.base_tokens)
        throw std::logic_error("LSA transaction ticket is stale or has the wrong committed prefix");
    return slot;
}

CacheState::Slot& CacheState::live(const Ticket& ticket) {
    (void) static_cast<const CacheState*>(this)->live(ticket);
    return slots_[ticket.slot];
}

void CacheState::bump(Slot& slot) {
    if (slot.generation == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("LSA slot generation exhausted");
    ++slot.generation;
}

CacheState::Ticket CacheState::begin(uint32_t id) {
    Slot& slot = slot_at(id);
    if (slot.open) throw std::logic_error("LSA slot already has an open transaction");
    bump(slot);
    slot.open = true;
    slot.appended.fill(0);
    return {id, slot.generation, slot.committed};
}

CacheState::AppendRange CacheState::append(const Ticket& ticket, uint32_t layer, uint32_t count) {
    Slot& slot = live(ticket);
    uint32_t& appended = slot.appended[layer_ordinal(layer)];
    const uint32_t begin_token = slot.committed + appended;
    if (count > kContext - begin_token)
        throw std::out_of_range("LSA append exceeds 131072-token context");
    appended += count;
    return {begin_token, begin_token + count, begin_token / kBlockTokens,
            (begin_token + count) / kBlockTokens};
}

CacheState::Transition CacheState::finish(Slot& slot, uint32_t new_tokens) {
    const uint32_t old_tokens = slot.committed;
    // Bump first: an exhausted counter must not leave a partly mutated slot.
    bump(slot);
    slot.committed = new_tokens;
    slot.open = false;
    slot.appended.fill(0);
    return {old_tokens, new_tokens, old_tokens / kBlockTokens, new_tokens / kBlockTokens,
            new_tokens, new_tokens / kBlockTokens, slot.generation};
}

CacheState::Transition CacheState::accept_prefix(const Ticket& ticket, uint32_t accepted) {
    Slot& slot = live(ticket);
    const uint32_t appended = slot.appended[0];
    if (!std::all_of(slot.appended.begin(), slot.appended.end(),
                     [appended](uint32_t n) { return n == appended; }))
        throw std::logic_error("LSA cannot commit an incomplete per-layer forward");
    if (accepted > appended)
        throw std::out_of_range("LSA accepted prefix exceeds the speculative tokens");
    return finish(slot, slot.committed + accepted);
}

CacheState::Transition CacheState::reject(const Ticket& ticket) {
    Slot& slot = live(ticket);
    return finish(slot, slot.committed);
}

CacheState::Transition CacheState::truncate(uint32_t id, uint32_t retained) {
    Slot& slot = slot_at(id);
    if (retained > slot.committed)
        throw std::out_of_range("LSA prefix reuse cannot invent uncommitted tokens");
    return finish(slot, retained);
}

CacheState::Transition CacheState::reset(uint32_t id) { return truncate(id, 0); }

uint32_t CacheState::committed_tokens(uint32_t id) const { return slot_at(id).committed; }
uint32_t CacheState::published_blocks(uint32_t id) const { return slot_at(id).committed / kBlockTokens; }
bool CacheState::transaction_open(uint32_t id) const { return slot_at(id).open; }
uint64_t CacheState::generation(uint32_t id) const { return slot_at(id).generation; }

uint32_t CacheState::available_tokens(const Ticket& ticket, uint32_t layer) const {
    const Slot& slot = live(ticket);
    return slot.committed + slot.appended[layer_ordinal(layer)];
}

CacheState::View CacheState::make_view(uint32_t committed, uint32_t available, uint32_t position) {
    if (position >= available)
        throw std::out_of_range("LSA query would read unavailable or rejected KV tokens");
    const uint32_t visible = position + 1;
    const uint32_t complete = visible / kBlockTokens;
    const uint32_t published = std::min(committed / kBlockTokens, complete);
    const uint32_t tail = complete * kBlockTokens;
    const uint32_t long_end = visible > kWindowTokens ? visible - kWindowTokens : 0;
    return {visible, complete, position / kBlockTokens, published, complete - published, tail, visible - tail,
            long_end, long_end / kBlockTokens};
}

CacheState::View CacheState::view(uint32_t id, uint32_t layer, uint32_t position) const {
    (void) layer_ordinal(layer);
    const Slot& slot = slot_at(id);
    return make_view(slot.committed, slot.committed, position);
}

CacheState::View CacheState::view(const Ticket& ticket, uint32_t layer, uint32_t position) const {
    const Slot& slot = live(ticket);
    return make_view(slot.committed, slot.committed + slot.appended[layer_ordinal(layer)], position);
}

} // namespace strata::lsa
