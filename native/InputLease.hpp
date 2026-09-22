#pragma once
#include <cstdint>

namespace zelda_ai {
struct InputCommand {
    uint64_t seq = 0;
    uint64_t sceneEpoch = 0;
    uint64_t baseSeq = 0;
    uint16_t buttons = 0;
    int stickX = 0;
    int stickY = 0;
    int leaseMs = 0;
    bool active = false;
};

// No game-memory writes: this class owns only temporary controller input.
class InputLease {
  public:
    uint64_t lastCommandSeq = 0;
    bool Apply(const InputCommand& command, uint64_t sceneEpoch, uint64_t stateSeq, int64_t nowMs) {
        if (command.seq <= lastCommandSeq || command.sceneEpoch != sceneEpoch || command.baseSeq > stateSeq ||
            stateSeq - command.baseSeq > 10 || command.leaseMs < 0 || command.leaseMs > 500 ||
            command.stickX < -80 || command.stickX > 80 || command.stickY < -80 || command.stickY > 80) {
            return false;
        }
        lastCommandSeq = command.seq;
        current = command;
        expiresAt = nowMs + command.leaseMs;
        return true;
    }
    bool Read(int64_t nowMs, uint16_t& buttons, int8_t& x, int8_t& y) const {
        if (!current.active || nowMs >= expiresAt) return false;
        buttons = current.buttons;
        x = static_cast<int8_t>(current.stickX);
        y = static_cast<int8_t>(current.stickY);
        return true;
    }
    void Release() { current.active = false; }
  private:
    InputCommand current;
    int64_t expiresAt = 0;
};
} // namespace zelda_ai
