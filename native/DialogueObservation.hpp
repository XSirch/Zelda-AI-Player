#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace zelda_ai {
// Message modes from the pinned z64.h. Only phases with Message_DrawText
// expose the decoded page. Start/growth/next/closing buffers are not text.
constexpr bool DialogueDrawsText(uint8_t mode) {
    switch (mode) {
        case 6: case 7: case 8:
        case 9: case 10: case 11: case 12: case 13: case 14:
        case 18: case 19: case 20: case 21: case 22:
        case 24: case 25: case 26: case 27: case 28: case 29: case 31:
        case 33: case 34: case 36: case 37: case 39: case 46: case 47:
        case 52: case 53:
            return true;
        default:
            return false;
    }
}
constexpr size_t DialogueVisibleBytes(bool active, uint8_t mode, size_t decoded,
                                     size_t drawPosition, size_t capacity) {
    return active && DialogueDrawsText(mode) ? std::min({decoded, drawPosition, capacity}) : 0;
}

template <typename Actor>
const Actor* OfferedTalkActor(const Actor* offered, const Actor* focused, const Actor* self) {
    if (offered && offered != self) return offered;
    return focused && focused != self ? focused : nullptr;
}
}
