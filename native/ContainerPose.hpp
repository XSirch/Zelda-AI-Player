#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <cstdlib>

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

enum class ContainerLidPose { Unknown, Closed, Open };

inline int LidAngleDistance(int16_t a, int16_t b) {
    const int difference = static_cast<int>(a) - static_cast<int>(b);
    return std::abs(((difference + 98304) % 65536) - 32768);
}

// Calibrate from the rendered animation's two endpoint poses, never a treasure
// flag or actor action function. Intermediate/ambiguous poses remain unknown.
inline ContainerLidPose ClassifyContainerLid(std::optional<int16_t> current,
                                            int16_t closed, int16_t open) {
    if (!current || LidAngleDistance(closed, open) < 4096) return ContainerLidPose::Unknown;
    if (LidAngleDistance(*current, open) <= 512) return ContainerLidPose::Open;
    if (LidAngleDistance(*current, closed) <= 512) return ContainerLidPose::Closed;
    return ContainerLidPose::Unknown;
}
}
