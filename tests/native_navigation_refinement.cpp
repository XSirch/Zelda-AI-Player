#include "NavigationRefinement.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

#define REQUIRE(condition) do { if (!(condition)) { \
    std::cerr << "Check failed at line " << __LINE__ << ": " << #condition << '\n'; \
    std::exit(1); } } while (false)
#define CASE(n) void n()

CASE(fresh_current_query_expires_without_renewal) {
    zelda_ai::NavigationRefinementWindow q;
    REQUIRE(q.Request(10,2,3,11,2,3,12,1000));
    REQUIRE(q.Active(1000,2,3));
    REQUIRE(q.Active(2499,2,3));
    REQUIRE(!q.Active(2500,2,3));
}
CASE(scene_and_control_context_changes_invalidate_query) {
    zelda_ai::NavigationRefinementWindow q;
    REQUIRE(q.Request(10,2,3,11,2,3,12,1000));
    REQUIRE(!q.Active(1001,3,3));
    REQUIRE(!q.Active(1001,2,4));
    REQUIRE(!q.Active(999,2,3));
}
CASE(stale_and_future_samples_never_request_collision) {
    zelda_ai::NavigationRefinementWindow q;
    REQUIRE(!q.Request(10,2,3,11,2,3,20,1000));
    REQUIRE(!q.Request(10,2,3,13,2,3,12,1000));
    REQUIRE(!q.Active(1000,2,3));
}
CASE(wrong_context_or_invalid_identity_never_request_collision) {
    zelda_ai::NavigationRefinementWindow q;
    REQUIRE(!q.Request(10,4,3,11,2,3,12,1000));
    REQUIRE(!q.Request(10,2,4,11,2,3,12,1000));
    REQUIRE(!q.Request(0,2,3,11,2,3,12,1000));
    REQUIRE(!q.Request(12,2,3,11,2,3,12,1000));
    REQUIRE(!q.Request(1,2,3,0,2,3,12,1000));
}
CASE(duplicate_and_frequent_queries_cannot_extend_window) {
    zelda_ai::NavigationRefinementWindow q;
    REQUIRE(q.Request(10,2,3,11,2,3,12,1000));
    REQUIRE(!q.Request(10,2,3,11,2,3,12,1000));
    REQUIRE(!q.Request(10,2,3,12,2,3,12,1499));
    REQUIRE(q.expiresAtMs==2500);
    REQUIRE(q.Request(10,2,3,12,2,3,12,1500));
    REQUIRE(q.expiresAtMs==3000);
}
CASE(cancellation_revokes_finer_observation) {
    zelda_ai::NavigationRefinementWindow q;
    REQUIRE(q.Request(10,2,3,11,2,3,12,1000));
    q.Cancel();
    REQUIRE(!q.Active(1001,2,3));
    REQUIRE(q.requestId==0);
}

int main(int argc,char** argv) {
    struct Test { const char* name; void (*run)(); };
    const Test tests[]{
        {"fresh_current_query_expires_without_renewal",fresh_current_query_expires_without_renewal},
        {"scene_and_control_context_changes_invalidate_query",scene_and_control_context_changes_invalidate_query},
        {"stale_and_future_samples_never_request_collision",stale_and_future_samples_never_request_collision},
        {"wrong_context_or_invalid_identity_never_request_collision",wrong_context_or_invalid_identity_never_request_collision},
        {"duplicate_and_frequent_queries_cannot_extend_window",duplicate_and_frequent_queries_cannot_extend_window},
        {"cancellation_revokes_finer_observation",cancellation_revokes_finer_observation},
    };
    bool found=argc==1;
    for (const auto& test:tests) if (argc==1 || test.name==std::string(argv[1])) {
        found=true; test.run(); std::cout << test.name << " passed\n";
    }
    return found ? 0 : 2;
}
