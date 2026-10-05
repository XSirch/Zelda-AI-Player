#pragma once
// Bounded observation queries only. No game state writes or route memory.
#include <algorithm>
#include <cmath>

namespace zelda_ai {
struct WalkingEdgeClearance {
    bool blocked = false;
    bool lowerBandBlocked = false;
};

template <typename Point, typename WallQuery>
WalkingEdgeClearance InspectWalkingEdge(const Point& fromFloor, const Point& toFloor,
                                       float bodyRadius, float centerHeight,
                                       float lowerHeight, WallQuery wallQuery) {
    const float vx = toFloor.x - fromFloor.x;
    const float vz = toFloor.z - fromFloor.z;
    const float length = std::sqrt(vx * vx + vz * vz);
    if (length <= 0.001f) return {true, false};
    const float px = -vz / length * bodyRadius;
    const float pz = vx / length * bodyRadius;
    const Point start{fromFloor.x, fromFloor.y + centerHeight, fromFloor.z};
    const Point end{toFloor.x, toFloor.y + centerHeight, toFloor.z};
    const Point leftStart{start.x + px, start.y, start.z + pz};
    const Point leftEnd{end.x + px, end.y, end.z + pz};
    const Point rightStart{start.x - px, start.y, start.z - pz};
    const Point rightEnd{end.x - px, end.y, end.z - pz};
    if (wallQuery(start, end) || wallQuery(leftStart, leftEnd) || wallQuery(rightStart, rightEnd)) {
        return {true, false};
    }
    // The original center band can pass above a low solid wall. Check the
    // lower body's band at the lower observed endpoint floor, rather than
    // lifting this ray over a raised endpoint. These wall-only queries do not
    // collide with a continuous floor slope. A ledge remains a separate
    // traversal proposal, never an automatically authorized walking link.
    const float lowY = std::min(fromFloor.y, toFloor.y) + lowerHeight;
    const Point lowStart{start.x, lowY, start.z};
    const Point lowEnd{end.x, lowY, end.z};
    const Point lowLeftStart{leftStart.x, lowY, leftStart.z};
    const Point lowLeftEnd{leftEnd.x, lowY, leftEnd.z};
    const Point lowRightStart{rightStart.x, lowY, rightStart.z};
    const Point lowRightEnd{rightEnd.x, lowY, rightEnd.z};
    if (wallQuery(lowStart, lowEnd) || wallQuery(lowLeftStart, lowLeftEnd) || wallQuery(lowRightStart, lowRightEnd)) {
        return {true, true};
    }
    return {};
}
} // namespace zelda_ai
