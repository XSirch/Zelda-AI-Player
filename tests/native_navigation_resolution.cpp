#include "NavigationResolution.hpp"
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#define REQUIRE(condition) do { if (!(condition)) { \
    std::cerr << "Check failed at line " << __LINE__ << ": " << #condition << '\n'; \
    std::exit(1); } } while (false)

using Cells = std::map<std::pair<int, int>, uint8_t>;
int Component(const Cells& cells, int half = 4) {
    return zelda_ai::NavigationComponentSize(half, [&](int x, int z) {
        const auto found = cells.find({x, z});
        return found == cells.end() ? zelda_ai::NavigationCellLinks{}
            : zelda_ai::NavigationCellLinks{true, found->second};
    });
}

#define CASE(n) void n()
CASE(recorded_small_component_requests_refinement) {
    // Exact link masks from the retained native failure. No imagined fine cells.
    const Cells cells{{{1, -1}, 1}, {{-1, 0}, 0}, {{0, 0}, 1}, {{1, 0}, 20},
                      {{2, 0}, 64}, {{0, 1}, 20}, {{1, 1}, 64}};
    const int count = Component(cells);
    REQUIRE(count == 3);
    REQUIRE(zelda_ai::NeedsNavigationRefinement(70, count));
}
CASE(isolated_root_requests_refinement) {
    REQUIRE(Component({{{0, 0}, 0}}) == 1);
    REQUIRE(zelda_ai::NeedsNavigationRefinement(70, 1));
}
CASE(four_cell_component_requests_refinement) {
    const Cells cells{{{0, 0}, 4}, {{1, 0}, 68}, {{2, 0}, 68}, {{3, 0}, 64}};
    REQUIRE(Component(cells) == 4);
    REQUIRE(zelda_ai::NeedsNavigationRefinement(70, 4));
}
CASE(large_component_keeps_coarse_resolution) {
    const Cells cells{{{0, 0}, 4}, {{1, 0}, 68}, {{2, 0}, 68}, {{3, 0}, 68}, {{4, 0}, 64}};
    REQUIRE(Component(cells) == 5);
    REQUIRE(!zelda_ai::NeedsNavigationRefinement(70, 5));
}
CASE(no_floor_does_not_refine) {
    REQUIRE(Component({{{1, 0}, 64}}) == 0);
    REQUIRE(!zelda_ai::NeedsNavigationRefinement(70, 0));
}
CASE(fine_resolution_never_recurses) {
    REQUIRE(!zelda_ai::NeedsNavigationRefinement(35, 1));
    REQUIRE(!zelda_ai::NeedsNavigationRefinement(35, 3));
    REQUIRE(!zelda_ai::NeedsNavigationRefinement(0, 3));
}
CASE(directed_links_do_not_synthesize_reverse_edges) {
    REQUIRE(Component({{{0, 0}, 0}, {{1, 0}, 64}}) == 1);
    REQUIRE(Component({{{0, 0}, 4}, {{1, 0}, 0}}) == 2);
}
CASE(missing_and_out_of_bounds_cells_are_ignored) {
    REQUIRE(Component({{{0, 0}, 255}}) == 1);
    REQUIRE(Component({{{0, 0}, 4}, {{1, 0}, 4}, {{2, 0}, 4}}, 1) == 2);
    REQUIRE(Component({{{0, 0}, 4}}, 9) == 0);
}
CASE(full_fine_grid_stays_bounded) {
    Cells cells;
    for (int z = -8; z <= 8; ++z) for (int x = -8; x <= 8; ++x) cells[{x, z}] = 255;
    REQUIRE(Component(cells, 8) == 289);
    REQUIRE(zelda_ai::NavigationHalfExtent(70) == 4);
    REQUIRE(zelda_ai::NavigationHalfExtent(35) == 8);
    REQUIRE(70 * zelda_ai::NavigationHalfExtent(70) == 35 * zelda_ai::NavigationHalfExtent(35));
}

int main(int argc, char** argv) {
    struct Test { const char* name; void (*run)(); };
    const Test tests[]{
        {"recorded_small_component_requests_refinement", recorded_small_component_requests_refinement},
        {"isolated_root_requests_refinement", isolated_root_requests_refinement},
        {"four_cell_component_requests_refinement", four_cell_component_requests_refinement},
        {"large_component_keeps_coarse_resolution", large_component_keeps_coarse_resolution},
        {"no_floor_does_not_refine", no_floor_does_not_refine},
        {"fine_resolution_never_recurses", fine_resolution_never_recurses},
        {"directed_links_do_not_synthesize_reverse_edges", directed_links_do_not_synthesize_reverse_edges},
        {"missing_and_out_of_bounds_cells_are_ignored", missing_and_out_of_bounds_cells_are_ignored},
        {"full_fine_grid_stays_bounded", full_fine_grid_stays_bounded},
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
