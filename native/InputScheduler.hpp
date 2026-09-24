#pragma once
// Pure controller state: no game-memory writes, networking or clock calls.
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <vector>

namespace zelda_ai {
struct PadState {
    uint16_t buttons = 0;
    int stickX = 0;
    int stickY = 0;
    bool Valid() const { return stickX >= -80 && stickX <= 80 && stickY >= -80 && stickY <= 80; }
};
struct InputStep {
    PadState pad;
    int ticks = 1; // Input consumer invocations, NOT renderer frames.
};
enum class InputKind { Setpoint, Sequence, Renew, Cancel, Release };
struct ScheduledInput {
    uint64_t seq = 0, ownerEpoch = 0, sceneEpoch = 0, contextEpoch = 0, baseSeq = 0;
    InputKind kind = InputKind::Setpoint;
    PadState pad; // Setpoint or the neutral/held baseline following a sequence.
    int leaseMs = 300;
    uint16_t edgeButtons = 0;
    std::vector<InputStep> steps;
};
enum class ReceiptStatus { Accepted, Consumed, Completed, Superseded, Cancelled, Rejected };
inline const char* ReceiptStatusName(ReceiptStatus value) {
    switch (value) {
        case ReceiptStatus::Accepted: return "accepted";
        case ReceiptStatus::Consumed: return "consumed";
        case ReceiptStatus::Completed: return "completed";
        case ReceiptStatus::Superseded: return "superseded";
        case ReceiptStatus::Cancelled: return "cancelled";
        default: return "rejected";
    }
}
struct InputReceipt {
    uint64_t seq = 0, ownerEpoch = 0, firstTick = 0, lastTick = 0;
    ReceiptStatus status = ReceiptStatus::Accepted;
    uint16_t pressed = 0, released = 0;
    int64_t receivedMs = 0, applyLatencyMs = -1;
    const char* reason = "";
};
struct ConsumedInput {
    bool owned = false;
    PadState pad;
    uint16_t pressed = 0, released = 0;
    uint64_t commandSeq = 0;
};

class InputScheduler {
  public:
    static constexpr size_t MaxReceipts = 64;
    static constexpr int MaxStateAgeMs = 300;
    uint64_t lastReceivedSeq = 0, lastCommandSeq = 0, lastAppliedSeq = 0;
    uint64_t ownerEpoch = 0, inputTick = 0;

    void SetContext(uint64_t scene, uint64_t context) {
        if (scene != sceneEpoch || context != contextEpoch) {
            Release("context_changed");
            sceneEpoch = scene;
            contextEpoch = context;
        }
    }
    void ObserveSample(uint64_t seq, int64_t nowMs) {
        samples[seq % samples.size()] = {seq, sceneEpoch, contextEpoch, nowMs};
    }
    const std::deque<InputReceipt>& Receipts() const { return receipts; }
    const InputReceipt* Receipt(uint64_t seq) const {
        for (const auto& row : receipts) if (row.seq == seq) return &row;
        return nullptr;
    }
    bool Active(int64_t nowMs) const { return active && nowMs < expiresAt; }

    bool Accept(const ScheduledInput& cmd, int64_t nowMs) {
        Expire(nowMs);
        // A duplicate never extends the watchdog, changes data, or replays an action.
        if (auto* row = Receipt(cmd.seq)) {
            return row->ownerEpoch == cmd.ownerEpoch && row->status != ReceiptStatus::Rejected;
        }
        if (cmd.seq == 0 || cmd.seq <= lastReceivedSeq) return false;
        lastReceivedSeq = cmd.seq;
        if (cmd.ownerEpoch < ownerEpoch) return Reject(cmd, nowMs, "old_owner");
        if (cmd.kind == InputKind::Release) {
            // An authenticated emergency handoff must work even with stale scene/sample data.
            ownerEpoch = cmd.ownerEpoch;
            Release("released");
            AddReceipt(cmd, nowMs).status = ReceiptStatus::Completed;
            lastCommandSeq = cmd.seq;
            return true;
        }
        if (!cmd.pad.Valid() || cmd.leaseMs < 1 || cmd.leaseMs > 500) {
            return Reject(cmd, nowMs, "invalid_input");
        }
        const auto& sample = samples[cmd.baseSeq % samples.size()];
        if (cmd.sceneEpoch != sceneEpoch || cmd.contextEpoch != contextEpoch ||
            sample.seq != cmd.baseSeq || cmd.baseSeq == 0 || sample.scene != sceneEpoch ||
            sample.context != contextEpoch || nowMs < sample.atMs || nowMs - sample.atMs > MaxStateAgeMs) {
            return Reject(cmd, nowMs, "stale_context_or_sample");
        }
        if (cmd.kind == InputKind::Renew && (cmd.ownerEpoch != ownerEpoch || !active)) {
            return Reject(cmd, nowMs, "nothing_to_renew");
        }
        if (cmd.kind == InputKind::Sequence) {
            if (cmd.steps.empty() || cmd.steps.size() > 8) return Reject(cmd, nowMs, "invalid_sequence");
            int total = 0;
            for (const auto& step : cmd.steps) {
                if (!step.pad.Valid() || step.ticks < 1 || step.ticks > 8) {
                    return Reject(cmd, nowMs, "invalid_sequence");
                }
                total += step.ticks;
            }
            if (total > 31) return Reject(cmd, nowMs, "sequence_too_long");
            if (actionSeq != 0 && cmd.ownerEpoch == ownerEpoch) return Reject(cmd, nowMs, "sequence_busy");
        }
        if (cmd.kind == InputKind::Cancel) {
            Release("cancelled_by_owner");
        }
        if (cmd.ownerEpoch > ownerEpoch) {
            Release("superseded_owner");
            ownerEpoch = cmd.ownerEpoch;
        }
        active = true;
        expiresAt = nowMs + cmd.leaseMs;
        lastCommandSeq = cmd.seq;
        auto& receipt = AddReceipt(cmd, nowMs);
        if (cmd.kind == InputKind::Renew) {
            receipt.status = ReceiptStatus::Completed;
            return true;
        }
        SupersedeSetpoint();
        desired = cmd.pad;
        setpointSeq = (cmd.kind == InputKind::Setpoint || cmd.kind == InputKind::Cancel) ? cmd.seq : 0;
        if (cmd.kind == InputKind::Sequence) {
            actionSeq = cmd.seq;
            steps = cmd.steps;
            // A requested press needs a real release edge when that button is already held.
            if ((previous.buttons & cmd.edgeButtons) != 0) {
                auto baseline = cmd.pad;
                baseline.buttons &= static_cast<uint16_t>(~cmd.edgeButtons);
                steps.insert(steps.begin(), {baseline, 1});
            }
            stepIndex = 0;
            ticksLeft = steps.front().ticks;
            actionDeadline = nowMs + 2000; // Renewals cannot extend action lifetime indefinitely.
        }
        return true;
    }

    ConsumedInput Consume(int64_t nowMs, PadState human = {}) {
        ++inputTick;
        Expire(nowMs);
        if (!active) {
            previous = human;
            return {};
        }
        ConsumedInput out;
        out.owned = true;
        out.pad = actionSeq ? steps[stepIndex].pad : desired;
        out.commandSeq = actionSeq ? actionSeq : setpointSeq;
        out.pressed = static_cast<uint16_t>((previous.buttons ^ out.pad.buttons) & out.pad.buttons);
        out.released = static_cast<uint16_t>((previous.buttons ^ out.pad.buttons) & previous.buttons);
        previous = out.pad;
        if (out.commandSeq) {
            lastAppliedSeq = out.commandSeq;
            if (auto* row = MutableReceipt(out.commandSeq)) {
                if (row->firstTick == 0) {
                    row->firstTick = inputTick;
                    row->applyLatencyMs = std::max<int64_t>(0, nowMs - row->receivedMs);
                }
                row->lastTick = inputTick;
                row->pressed |= out.pressed;
                row->released |= out.released;
                row->status = actionSeq ? ReceiptStatus::Consumed : ReceiptStatus::Completed;
            }
        }
        if (actionSeq && --ticksLeft == 0) {
            if (++stepIndex == steps.size()) {
                if (auto* row = MutableReceipt(actionSeq)) row->status = ReceiptStatus::Completed;
                actionSeq = 0;
                steps.clear();
            } else {
                ticksLeft = steps[stepIndex].ticks;
            }
        }
        return out;
    }

    void Release(const char* reason = "released") {
        if (actionSeq) {
            if (auto* row = MutableReceipt(actionSeq)) {
                row->status = ReceiptStatus::Cancelled;
                row->reason = reason;
            }
        }
        if (auto* row = MutableReceipt(setpointSeq); row && row->firstTick == 0) {
            row->status = ReceiptStatus::Cancelled;
            row->reason = reason;
        }
        active = false;
        actionSeq = setpointSeq = 0;
        desired = {};
        steps.clear();
    }

  private:
    struct Sample { uint64_t seq = 0, scene = 0, context = 0; int64_t atMs = 0; };
    std::array<Sample, 128> samples{};
    uint64_t sceneEpoch = 0, contextEpoch = 0, actionSeq = 0, setpointSeq = 0;
    bool active = false;
    int64_t expiresAt = 0, actionDeadline = 0;
    PadState desired{}, previous{};
    std::vector<InputStep> steps;
    size_t stepIndex = 0;
    int ticksLeft = 0;
    std::deque<InputReceipt> receipts;

    InputReceipt* MutableReceipt(uint64_t seq) {
        for (auto& row : receipts) if (row.seq == seq) return &row;
        return nullptr;
    }
    InputReceipt& AddReceipt(const ScheduledInput& cmd, int64_t nowMs) {
        // Do not evict a live sequence receipt during frequent renewals.
        if (receipts.size() >= MaxReceipts) {
            auto victim = std::find_if(receipts.begin(), receipts.end(), [&](const InputReceipt& row) {
                return row.seq != actionSeq && row.seq != setpointSeq;
            });
            if (victim != receipts.end()) receipts.erase(victim);
        }
        receipts.push_back({cmd.seq, cmd.ownerEpoch, 0, 0, ReceiptStatus::Accepted, 0, 0, nowMs, -1, ""});
        return receipts.back();
    }
    bool Reject(const ScheduledInput& cmd, int64_t nowMs, const char* reason) {
        auto& row = AddReceipt(cmd, nowMs);
        row.status = ReceiptStatus::Rejected;
        row.reason = reason;
        return false;
    }
    void Expire(int64_t nowMs) {
        if (active && nowMs >= expiresAt) Release("watchdog_expired");
        else if (actionSeq && nowMs >= actionDeadline) Release("sequence_timeout");
    }
    void SupersedeSetpoint() {
        if (auto* row = MutableReceipt(setpointSeq); row && row->firstTick == 0) {
            row->status = ReceiptStatus::Superseded;
            row->reason = "newer_setpoint";
        }
    }
};
} // namespace zelda_ai
