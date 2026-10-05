#include "NavMeshQueries.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define REQUIRE(condition) do { if (!(condition)) { \
    std::cerr << "Check failed at line " << __LINE__ << ": " << #condition << '\n'; \
    std::exit(1); } } while (false)

struct Point { float x, y, z; };
struct Box { Point low, high; };

// Independent segment/box oracle for the query-selection seam. These are
// geometric test obstacles, never game observations or gameplay evidence.
bool Intersects(Point a, Point b, const Box& box) {
    float near = 0.0f, far = 1.0f;
    const float starts[]{a.x, a.y, a.z};
    const float ends[]{b.x, b.y, b.z};
    const float lows[]{box.low.x, box.low.y, box.low.z};
    const float highs[]{box.high.x, box.high.y, box.high.z};
    for (int axis = 0; axis < 3; ++axis) {
        const float delta = ends[axis] - starts[axis];
        if (std::abs(delta) < 0.00001f) {
            if (starts[axis] < lows[axis] || starts[axis] > highs[axis]) return false;
            continue;
        }
        float first = (lows[axis] - starts[axis]) / delta;
        float last = (highs[axis] - starts[axis]) / delta;
        if (first > last) std::swap(first, last);
        near = std::max(near, first);
        far = std::min(far, last);
        if (near > far) return false;
    }
    return true;
}

zelda_ai::WalkingEdgeClearance Inspect(Point a, Point b, const std::vector<Box>& obstacles) {
    return zelda_ai::InspectWalkingEdge(a, b, 18.0f, 26.0f, 12.0f,
        [&](Point start, Point end) {
            for (const auto& obstacle : obstacles) {
                if (Intersects(start, end, obstacle)) return true;
            }
            return false;
        });
}

#define CASE(n) void n()
CASE(low_wall_between_flat_cells) {
    const auto result = Inspect({0, 0, 0}, {0, 0, 70}, {{{-100, 0, 30}, {100, 21, 31}}});
    REQUIRE(result.blocked && result.lowerBandBlocked);
}
CASE(low_wall_before_raised_endpoint) {
    const auto result = Inspect({0, 0, 0}, {0, 21, 70}, {{{-100, 0, 55}, {100, 21, 56}}});
    REQUIRE(result.blocked && result.lowerBandBlocked);
}
CASE(low_wall_in_reverse_direction) {
    const auto result = Inspect({0, 21, 70}, {0, 0, 0}, {{{-100, 0, 55}, {100, 21, 56}}});
    REQUIRE(result.blocked && result.lowerBandBlocked);
}
CASE(low_wall_on_body_side_only) {
    const auto result = Inspect({0, 0, 0}, {0, 0, 70}, {{{15, 0, 30}, {19, 21, 31}}});
    REQUIRE(result.blocked && result.lowerBandBlocked);
}
CASE(tall_wall_retains_high_band_failure) {
    const auto result = Inspect({0, 0, 0}, {0, 0, 70}, {{{-100, 0, 30}, {100, 40, 31}}});
    REQUIRE(result.blocked && !result.lowerBandBlocked);
}
CASE(clear_slope_remains_clear) {
    const auto result = Inspect({0, 0, 0}, {70, 21, 70}, {});
    REQUIRE(!result.blocked && !result.lowerBandBlocked);
}
CASE(obstacle_below_body_remains_clear) {
    const auto result = Inspect({0, 0, 0}, {0, 0, 70}, {{{-100, 0, 30}, {100, 10, 31}}});
    REQUIRE(!result.blocked && !result.lowerBandBlocked);
}
CASE(degenerate_edge_fails_closed) {
    int queries = 0;
    const auto result = zelda_ai::InspectWalkingEdge(Point{0, 0, 0}, Point{0, 21, 0}, 18.0f, 26.0f, 12.0f,
        [&](Point, Point) { ++queries; return false; });
    REQUIRE(result.blocked && !result.lowerBandBlocked && queries == 0);
}

int main(int argc, char** argv) {
    struct Test { const char* name; void (*run)(); };
    const Test tests[]{
        {"low_wall_between_flat_cells", low_wall_between_flat_cells},
        {"low_wall_before_raised_endpoint", low_wall_before_raised_endpoint},
        {"low_wall_in_reverse_direction", low_wall_in_reverse_direction},
        {"low_wall_on_body_side_only", low_wall_on_body_side_only},
        {"tall_wall_retains_high_band_failure", tall_wall_retains_high_band_failure},
        {"clear_slope_remains_clear", clear_slope_remains_clear},
        {"obstacle_below_body_remains_clear", obstacle_below_body_remains_clear},
        {"degenerate_edge_fails_closed", degenerate_edge_fails_closed},
    };
    bool found = argc == 1;
    for (const auto& test : tests) {
        if (argc == 1 || test.name == std::string(argv[1])) {
            found = true;
            test.run();
            std::cout << test.name << " passed\n";
        }
    }
    return found ? 0 : 2;
}
