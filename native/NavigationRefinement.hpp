#pragma once
// Read-only, bounded finer collision observations; never controller input.
#include <cstdint>

namespace zelda_ai {
struct NavigationRefinementWindow {
    uint64_t requestId = 0, sceneEpoch = 0, contextEpoch = 0;
    int64_t expiresAtMs = 0, lastRequestedAtMs = -1000;

    bool Request(uint64_t id, uint64_t scene, uint64_t context, uint64_t baseSeq,
                 uint64_t currentScene, uint64_t currentContext, uint64_t currentSeq, int64_t nowMs) {
        if (!id || !baseSeq || id > baseSeq || baseSeq > currentSeq || currentSeq-baseSeq > 8 ||
            scene != currentScene || context != currentContext || nowMs < lastRequestedAtMs ||
            nowMs-lastRequestedAtMs < 500) return false;
        requestId = id;
        sceneEpoch = scene;
        contextEpoch = context;
        lastRequestedAtMs = nowMs;
        expiresAtMs = nowMs + 1500;
        return true;
    }

    bool Active(int64_t nowMs, uint64_t scene, uint64_t context) const {
        return requestId && nowMs >= lastRequestedAtMs && nowMs < expiresAtMs &&
            scene == sceneEpoch && context == contextEpoch;
    }

    void Cancel() { requestId = 0; expiresAtMs = 0; }
};
} // namespace zelda_ai
