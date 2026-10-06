#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>

namespace zelda_ai {
// Read only the currently rendered skeleton's lid joint. Unknown/hidden poses
// stay unknown; treasure flags, item contents and actor action functions are
// deliberately absent from this interface.
template <typename Joint>
std::optional<int16_t> VisibleContainerLidRotation(bool drawn, uint8_t alpha,
                                                 const Joint* joints, size_t count) {
    if (!drawn || alpha == 0 || joints == nullptr || count < 4 || count > 5) {
        return std::nullopt;
    }
    return static_cast<int16_t>(joints[3].z);
}
}
