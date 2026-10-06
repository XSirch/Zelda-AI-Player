#include "DialogueObservation.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

#define REQUIRE(condition) do { if (!(condition)) { \
    std::cerr << "Check failed at line " << __LINE__ << ": " << #condition << '\n'; \
    std::exit(1); } } while (false)
#define CASE(n) void n()
CASE(retained_start_and_growth_buffer_is_not_displayed) {
    for (uint8_t mode = 0; mode <= 5; ++mode) {
        // Captured native failures were modes 1/2, with a retained buffer.
        REQUIRE(zelda_ai::DialogueVisibleBytes(true, mode, 200, 200, 200) == 0);
    }
    REQUIRE(zelda_ai::DialogueVisibleBytes(true, 54, 200, 200, 200) == 0);
    REQUIRE(zelda_ai::DialogueVisibleBytes(true, 55, 200, 200, 200) == 0);
}
CASE(typing_exposes_only_drawn_prefix) {
    REQUIRE(zelda_ai::DialogueVisibleBytes(true, 6, 60, 8, 200) == 8);
    REQUIRE(zelda_ai::DialogueVisibleBytes(true, 6, 60, 0, 200) == 0);
}
CASE(visible_span_remains_bounded_and_requires_active_page) {
    REQUIRE(zelda_ai::DialogueVisibleBytes(true, 53, 60, 1000, 200) == 60);
    REQUIRE(zelda_ai::DialogueVisibleBytes(true, 53, 1000, 1000, 200) == 200);
    REQUIRE(zelda_ai::DialogueVisibleBytes(true, 53, 0, 1000, 200) == 0);
    REQUIRE(zelda_ai::DialogueVisibleBytes(false, 53, 60, 60, 200) == 0);
    REQUIRE(zelda_ai::DialogueVisibleBytes(true, 255, 60, 60, 200) == 0);
}
CASE(waiting_and_verified_music_text_phases_remain_supported) {
    for (uint8_t mode : {uint8_t{7}, uint8_t{8}, uint8_t{9}, uint8_t{21}, uint8_t{52}, uint8_t{53}}) {
        REQUIRE(zelda_ai::DialogueVisibleBytes(true, mode, 60, 60, 200) == 60);
    }
    REQUIRE(zelda_ai::DialogueVisibleBytes(true, 32, 60, 60, 200) == 0);
}
CASE(offered_actor_precedes_unrelated_or_missing_focus) {
    const int self = 1, offered = 2, focused = 3;
    REQUIRE(zelda_ai::OfferedTalkActor(&offered, &focused, &self) == &offered);
    REQUIRE(zelda_ai::OfferedTalkActor<int>(&offered, nullptr, &self) == &offered);
    REQUIRE(zelda_ai::OfferedTalkActor<int>(nullptr, &focused, &self) == &focused);
    REQUIRE(zelda_ai::OfferedTalkActor(&self, &self, &self) == nullptr);
}
int main(int argc, char** argv) {
    struct Test { const char* name; void (*run)(); };
    const Test tests[]{
        {"retained_start_and_growth_buffer_is_not_displayed", retained_start_and_growth_buffer_is_not_displayed},
        {"typing_exposes_only_drawn_prefix", typing_exposes_only_drawn_prefix},
        {"visible_span_remains_bounded_and_requires_active_page", visible_span_remains_bounded_and_requires_active_page},
        {"waiting_and_verified_music_text_phases_remain_supported", waiting_and_verified_music_text_phases_remain_supported},
        {"offered_actor_precedes_unrelated_or_missing_focus", offered_actor_precedes_unrelated_or_missing_focus},
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
