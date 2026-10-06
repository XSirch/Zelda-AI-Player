#include "ContainerPose.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

#define REQUIRE(condition) do { if (!(condition)) { \
    std::cerr << "Check failed at line " << __LINE__ << ": " << #condition << '\n'; \
    std::exit(1); } } while (false)
struct Joint { int16_t x{}, y{}, z{}; };
#define CASE(n) void n()
CASE(visible_pose_is_the_live_lid_joint) {
    Joint joints[5]{};
    joints[3].z = 32000;
    REQUIRE(zelda_ai::VisibleContainerLidRotation(true, 255, joints, 5) == 32000);
    joints[3].z = -32768;
    REQUIRE(zelda_ai::VisibleContainerLidRotation(true, 12, joints, 4) == -32768);
    joints[3].z = 0;
    REQUIRE(zelda_ai::VisibleContainerLidRotation(true, 255, joints, 5) == 0);
}
CASE(hidden_or_missing_pose_stays_unknown) {
    Joint joints[5]{};
    REQUIRE(!zelda_ai::VisibleContainerLidRotation(false, 255, joints, 5));
    REQUIRE(!zelda_ai::VisibleContainerLidRotation(true, 0, joints, 5));
    REQUIRE(!zelda_ai::VisibleContainerLidRotation<Joint>(true, 255, nullptr, 5));
}
CASE(incompatible_skeleton_stays_unknown) {
    Joint joints[5]{};
    REQUIRE(!zelda_ai::VisibleContainerLidRotation(true, 255, joints, 3));
    REQUIRE(!zelda_ai::VisibleContainerLidRotation(true, 255, joints, 6));
}
CASE(endpoint_calibration_is_not_an_absolute_angle_guess) {
    using P = zelda_ai::ContainerLidPose;
    REQUIRE(zelda_ai::ClassifyContainerLid(static_cast<int16_t>(13493), 32000, 13493) == P::Open);
    REQUIRE(zelda_ai::ClassifyContainerLid(static_cast<int16_t>(32000), 32000, 13493) == P::Closed);
    REQUIRE(zelda_ai::ClassifyContainerLid(static_cast<int16_t>(22000), 32000, 13493) == P::Unknown);
    REQUIRE(zelda_ai::ClassifyContainerLid(static_cast<int16_t>(13493), 13493, 32000) == P::Closed);
    REQUIRE(zelda_ai::ClassifyContainerLid(std::nullopt, 32000, 13493) == P::Unknown);
    REQUIRE(zelda_ai::ClassifyContainerLid(static_cast<int16_t>(13493), 13000, 13493) == P::Unknown);
}
CASE(wrapped_endpoint_and_tolerance_are_conservative) {
    using P = zelda_ai::ContainerLidPose;
    REQUIRE(zelda_ai::ClassifyContainerLid(static_cast<int16_t>(-32768), 13493, 32767) == P::Open);
    REQUIRE(zelda_ai::ClassifyContainerLid(static_cast<int16_t>(14005), 32000, 13493) == P::Open);
    REQUIRE(zelda_ai::ClassifyContainerLid(static_cast<int16_t>(14006), 32000, 13493) == P::Unknown);
}
int main(int argc, char** argv) {
    struct Test { const char* name; void (*run)(); };
    const Test tests[]{
        {"visible_pose_is_the_live_lid_joint", visible_pose_is_the_live_lid_joint},
        {"hidden_or_missing_pose_stays_unknown", hidden_or_missing_pose_stays_unknown},
        {"incompatible_skeleton_stays_unknown", incompatible_skeleton_stays_unknown},
        {"endpoint_calibration_is_not_an_absolute_angle_guess", endpoint_calibration_is_not_an_absolute_angle_guess},
        {"wrapped_endpoint_and_tolerance_are_conservative", wrapped_endpoint_and_tolerance_are_conservative},
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
