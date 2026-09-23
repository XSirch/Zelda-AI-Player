#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>

namespace zelda_ai {
// Main/game-thread only. IDs survive movement but never a destroy/reinitialization.
class ActorRegistry {
  public:
    void Spawn(const void* actor) {
        if (actor && (ids.count(actor) || ids.size() < 4096)) ids[actor] = ++next;
    }
    void Destroy(const void* actor) { ids.erase(actor); }
    void Clear() { ids.clear(); }
    std::string Get(const void* actor) {
        if (!actor) return "";
        auto it = ids.find(actor);
        if (it == ids.end()) {
            if (ids.size() >= 4096) return ""; // Unknown, never reuse another actor's identity.
            Spawn(actor);
            it = ids.find(actor);
        }
        return std::to_string(it->second);
    }
  private:
    uint64_t next = 0;
    std::unordered_map<const void*, uint64_t> ids;
};
} // namespace zelda_ai
