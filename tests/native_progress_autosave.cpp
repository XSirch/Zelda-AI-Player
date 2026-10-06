#include "ProgressAutosave.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
#define REQUIRE(c) do { if (!(c)) { std::cerr << "Check failed at line " << __LINE__ << ": " << #c << '\n'; std::exit(1); } } while(false)
#define CASE(n) void n()
using namespace zelda_ai;

CASE(rupees_gain_schedules_without_a_scene_change) {
    ProgressAutosave a; SaveProgress s; s.rupees = 10;
    a.Observe(s, 40, 1, 1000, 1); REQUIRE(!a.pending);
    s.rupees = 15; a.Observe(s, 40, 1, 1100, 2);
    REQUIRE(a.pending && a.reasons == SaveResource && a.scene == 40 && a.file == 1);
    REQUIRE(!a.Ready(1800, 3, 1800, true, 40, 1));
    REQUIRE(a.Ready(1850, 3, 1850, true, 40, 1));
}
CASE(no_delta_full_capacity_or_spending_does_not_schedule) {
    ProgressAutosave a; SaveProgress s; s.rupees = 99; s.health = 48; s.ammo[0] = 10;
    a.Observe(s, 40, 1, 1000, 1); a.Observe(s, 40, 1, 2000, 2); REQUIRE(!a.pending);
    s.rupees = 80; s.health = 32; s.ammo[0] = 9;
    a.Observe(s, 40, 1, 2100, 3); REQUIRE(!a.pending);
}
CASE(real_item_and_resource_deltas_schedule) {
    for (int kind = 0; kind < 12; ++kind) {
        ProgressAutosave a; SaveProgress s; a.Observe(s, 40, 1, 1000, 1);
        if (kind == 0) s.health = 16;
        if (kind == 1) s.magic = 16;
        if (kind == 2) s.ammo[1] = 5;
        if (kind == 3) s.items[0] = 1;
        if (kind == 4) s.equipment = 1;
        if (kind == 5) s.quests = 1;
        if (kind == 6) s.upgrades[2] = 1;
        if (kind == 7) s.dungeonItems[2] = 1;
        if (kind == 8) s.keys[2] = 1;
        if (kind == 9) s.maxHealth = 64;
        if (kind == 10) s.skullTokens = 1;
        if (kind == 11) s.abilities = 1;
        a.Observe(s, 40, 1, 1100, 2); REQUIRE(a.pending);
    }
}
CASE(coalescing_has_a_maximum_wait_and_scene_change_keeps_gains) {
    ProgressAutosave a; SaveProgress s; a.Observe(s, 40, 1, 1000, 1);
    for (int i=1; i<=51; ++i) { s.rupees=i; a.Observe(s, 40, 1, 1000+100*i, 2); }
    REQUIRE(a.scheduledAt == 1100 && a.Ready(6100, 3, 6100, true, 40, 1));
    a.Schedule(SaveScene, 85, 1, 6150, 3);
    REQUIRE(a.reasons == (SaveScene | SaveResource));
    REQUIRE(a.Ready(6150, 4, 6150, true, 85, 1));
}
CASE(modal_death_context_and_missing_peer_cannot_save) {
    ProgressAutosave a; a.Schedule(SaveResource, 40, 1, 1000, 7);
    REQUIRE(!a.Ready(2000, 8, 2000, false, 40, 1));
    REQUIRE(!a.Ready(2000, 8, 2000, true, 85, 1));
    REQUIRE(!a.Ready(2000, 8, 2000, true, 40, 0));
    REQUIRE(!a.Ready(2000, 7, 2000, true, 40, 1));
    REQUIRE(!a.Ready(2000, 8, 0, true, 40, 1));
    REQUIRE(a.PeerExpired(6001, 7, 6001));
    a.CancelPending(); REQUIRE(!a.pending && !a.reasons);
}
CASE(reset_rebases_and_never_inherits_another_save) {
    ProgressAutosave a; SaveProgress s; a.Observe(s, 40, 1, 1000, 1);
    s.rupees=5; a.Observe(s, 40, 1, 1100, 2); REQUIRE(a.pending);
    a.Reset(); s.rupees=99; a.Observe(s, 52, 0, 2000, 3);
    REQUIRE(!a.pending && !a.inFlight);
}
CASE(disk_confirmation_and_gains_during_an_async_save_are_separate) {
    ProgressAutosave a; SaveProgress s; a.Observe(s, 40, 1, 1000, 1);
    s.rupees=5; a.Observe(s, 40, 1, 1100, 2);
    REQUIRE(a.Submit(1900) == SaveResource);
    REQUIRE(a.inFlight && !a.pending && !a.Confirm(0));
    s.rupees=10; a.Observe(s, 40, 1, 1950, 3); REQUIRE(a.pending);
    REQUIRE(!a.Ready(4000, 4, 4000, true, 40, 1));
    REQUIRE(a.Confirm(1) == SaveResource);
    REQUIRE(a.pending && a.Ready(4000, 4, 4000, true, 40, 1));
    REQUIRE(!a.Confirm(1));
}
CASE(chest_transition_saves_even_when_resources_are_full) {
    ProgressAutosave a; a.Schedule(SaveChest, 40, 1, 1000, 1);
    REQUIRE(a.Ready(1750, 2, 1750, true, 40, 1));
    REQUIRE(a.Submit(1750) == SaveChest); REQUIRE(a.Confirm(1) == SaveChest);
}
int main(int argc, char** argv) {
    struct Test { const char* name; void (*run)(); };
    const Test tests[]{
#define ENTRY(n) {#n, n}
        ENTRY(rupees_gain_schedules_without_a_scene_change), ENTRY(no_delta_full_capacity_or_spending_does_not_schedule),
        ENTRY(real_item_and_resource_deltas_schedule), ENTRY(coalescing_has_a_maximum_wait_and_scene_change_keeps_gains),
        ENTRY(modal_death_context_and_missing_peer_cannot_save), ENTRY(reset_rebases_and_never_inherits_another_save),
        ENTRY(disk_confirmation_and_gains_during_an_async_save_are_separate), ENTRY(chest_transition_saves_even_when_resources_are_full)
    };
    bool found = argc == 1;
    for (const auto& t : tests) if (argc == 1 || t.name == std::string(argv[1])) { found=true; t.run(); std::cout << t.name << " passed\n"; }
    return found ? 0 : 2;
}
