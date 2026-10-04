#pragma once

#include <array>
#include <bit>
#include <cstdint>

namespace zelda_ai {

enum class StartupPhase { Unknown, Title, SelectFile, ConfirmFile, Busy };

struct StartupObservation {
    StartupPhase phase = StartupPhase::Unknown;
    int cursor = -1;
    int selectedSlot = -1;
    std::array<bool, 3> existing{};
};

// This is an input fence, not a save loader or a semantic button mapping.
// Only ordinary controller input for an explicitly selected existing file is
// admitted. Copy, erase, name-entry, options and unknown states fail closed.
inline bool StartupInputAllowed(const StartupObservation& observation, int requestedSlot,
                                uint16_t buttons, int stickX, int stickY) {
    if (requestedSlot < 0 || requestedSlot >= 3 || !observation.existing[requestedSlot]) return false;
    if (std::popcount(buttons) > 1 || stickX != 0 || stickY < -80 || stickY > 80) return false;
    switch (observation.phase) {
        case StartupPhase::Title:
            return stickY == 0;
        case StartupPhase::SelectFile:
            return buttons == 0 || (observation.cursor == requestedSlot && stickY == 0);
        case StartupPhase::ConfirmFile:
            return observation.selectedSlot == requestedSlot &&
                   (buttons == 0 || (observation.cursor == 0 && stickY == 0));
        default:
            return buttons == 0 && stickY == 0;
    }
}

} // namespace zelda_ai
