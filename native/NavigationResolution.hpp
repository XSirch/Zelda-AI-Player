#pragma once
// Resolution choice over current collision links only; never a route edge.
#include <array>
#include <cstdint>

namespace zelda_ai {
struct NavigationCellLinks {
    bool valid = false;
    uint8_t links = 0;
};

template <typename ReadCell>
int NavigationComponentSize(int halfExtent, ReadCell readCell) {
    constexpr int MAX_HALF = 8;
    constexpr int SIDE = MAX_HALF * 2 + 1;
    constexpr std::array<int, 8> dx{0, 1, 1, 1, 0, -1, -1, -1};
    constexpr std::array<int, 8> dz{1, 1, 0, -1, -1, -1, 0, 1};
    if (halfExtent < 0 || halfExtent > MAX_HALF || !readCell(0, 0).valid) return 0;
    struct Key { int x, z; };
    std::array<Key, SIDE * SIDE> queue{};
    std::array<bool, SIDE * SIDE> visited{};
    const auto index = [](int x, int z) { return (z + MAX_HALF) * SIDE + x + MAX_HALF; };
    queue[0] = {0, 0};
    visited[index(0, 0)] = true;
    int count = 1;
    for (int head = 0; head < count; ++head) {
        const auto key = queue[head];
        const auto cell = readCell(key.x, key.z);
        for (int direction = 0; direction < 8; ++direction) {
            if (!(cell.links & (1u << direction))) continue;
            const int x = key.x + dx[direction], z = key.z + dz[direction];
            if (x < -halfExtent || x > halfExtent || z < -halfExtent || z > halfExtent) continue;
            if (visited[index(x, z)] || !readCell(x, z).valid) continue;
            visited[index(x, z)] = true;
            queue[count++] = {x, z};
        }
    }
    return count;
}

inline bool NeedsNavigationRefinement(float step, int componentCells) {
    return step > 35.0f && componentCells > 0 && componentCells <= 4;
}

inline int NavigationHalfExtent(float step) {
    return step > 35.0f ? 4 : 8; // Same 280-unit coverage at either native resolution.
}
} // namespace zelda_ai
