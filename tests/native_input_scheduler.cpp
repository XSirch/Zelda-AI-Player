#include "InputScheduler.hpp"
#include <cassert>
#include <iostream>
#include <string>
using namespace zelda_ai;
constexpr uint16_t A = 0x8000, B = 0x4000, Z = 0x2000;
InputScheduler ready() {
    InputScheduler out;
    out.SetContext(1, 1);
    out.ObserveSample(1, 0);
    return out;
}
ScheduledInput input(uint64_t seq = 1, uint64_t owner = 1) {
    ScheduledInput c;
    c.seq = seq; c.ownerEpoch = owner; c.sceneEpoch = 1; c.contextEpoch = 1; c.baseSeq = 1;
    return c;
}
ScheduledInput tap(uint64_t seq = 1, uint64_t owner = 1) {
    auto c = input(seq, owner);
    c.kind = InputKind::Sequence; c.edgeButtons = A;
    c.pad = {Z, 0, 0}; c.steps = {{{uint16_t(Z | A), 0, 0}, 1}, {{Z, 0, 0}, 1}};
    return c;
}
void button_width() {
    const auto pad = ToN64PadState(0x1A000u, -80, 80);
    assert(pad.buttons == A);
    assert(pad.stickX == -80 && pad.stickY == 80);
}
void end_to_end_latency() {
    auto s = ready(); auto c = tap();
    c.clientSentUs = 1000000;
    assert(s.Accept(c, 1005, 1005000));
    assert(s.Consume(1012, {}, 1012750).pressed & A);
    assert(s.Receipt(1)->applyLatencyMs == 7.75);
    assert(s.Receipt(1)->clientToConsumeMs == 12.75);
}
void one_tap() {
    auto s = ready(); auto c = tap();
    assert(s.Accept(c, 0)); assert(s.Receipt(1)->firstTick == 0);
    auto a = s.Consume(10); assert(a.pressed & A); assert(a.commandSeq == 1);
    auto b = s.Consume(60); assert((b.pad.buttons & A) == 0); assert(b.released & A);
    assert(s.Receipt(1)->status == ReceiptStatus::Completed);
    assert(s.Receipt(1)->firstTick == 1 && s.Receipt(1)->lastTick == 2);
    assert(s.Receipt(1)->applyLatencyMs == 10);
}
void repeated_press() {
    auto s = ready(); auto c = input(); c.pad = {uint16_t(Z | A), 0, 0};
    assert(s.Accept(c, 0)); assert(s.Consume(5).pressed & A);
    assert(s.Accept(tap(2), 10));
    assert((s.Consume(20).pad.buttons & A) == 0);
    assert(s.Consume(70).pressed & A);
    assert(s.Consume(120).released & A);
    assert(s.Receipt(2)->status == ReceiptStatus::Completed);
}
void release_before_consume() {
    auto s = ready(); assert(s.Accept(tap(), 0));
    auto c = input(2); c.kind = InputKind::Release;
    assert(s.Accept(c, 1)); assert(!s.Consume(2).owned);
    assert(s.Receipt(1)->status == ReceiptStatus::Cancelled && s.Receipt(1)->firstTick == 0);
}
void latest_setpoint() {
    auto s = ready(); auto a = input(); a.pad.stickY = 50;
    auto b = input(2); b.pad.stickX = -50;
    assert(s.Accept(a, 0)); assert(s.Accept(b, 0));
    auto out = s.Consume(1); assert(out.pad.stickX == -50 && out.pad.stickY == 0);
    assert(s.Receipt(1)->status == ReceiptStatus::Superseded);
    assert(s.Receipt(1)->firstTick == 0 && s.Receipt(2)->firstTick != 0);
}
void duplicate_once() {
    auto s = ready(); auto c = tap(); assert(s.Accept(c, 0));
    assert(s.Consume(10).pressed & A); assert(s.Accept(c, 15));
    assert(s.Consume(60).released & A); assert(s.Accept(c, 70));
    assert((s.Consume(110).pressed & A) == 0);
    assert(s.Receipt(1)->lastTick == 2);
}
void duplicate_does_not_renew() {
    auto s = ready(); auto c = input(); c.pad.buttons = B;
    assert(s.Accept(c, 0)); assert(s.Accept(c, 299));
    assert(!s.Consume(300).owned);
}
void stale_owner() {
    auto s = ready(); assert(s.Accept(tap(), 0));
    auto c = input(2, 2); c.pad.stickY = 70; assert(s.Accept(c, 5));
    auto old = input(3, 1); old.kind = InputKind::Release;
    assert(!s.Accept(old, 6)); assert(s.Consume(10).pad.stickY == 70);
    assert(s.Receipt(1)->status == ReceiptStatus::Cancelled);
}
void scene_change() {
    auto s = ready(); assert(s.Accept(tap(), 0)); s.SetContext(2, 2);
    assert(!s.Consume(1).owned); assert(!s.Accept(input(2), 2));
    s.ObserveSample(2, 3); auto c = input(3, 2);
    c.sceneEpoch = 2; c.contextEpoch = 2; c.baseSeq = 2;
    assert(s.Accept(c, 4)); assert(s.Consume(5).owned);
}
void context_change() {
    auto s = ready(); assert(s.Accept(tap(), 0)); s.SetContext(1, 2);
    assert(!s.Consume(1).owned); assert(!s.Accept(input(2), 2));
}
void watchdog_not_frames() {
    auto s = ready(); assert(s.Accept(tap(), 0)); assert(s.inputTick == 0);
    // No game/input frames have elapsed, but the wall-clock lease has.
    assert(!s.Consume(301).owned);
    assert(s.Receipt(1)->status == ReceiptStatus::Cancelled);
}
void old_sample() {
    auto s = ready(); assert(!s.Accept(input(), 301));
    assert(s.Receipt(1)->status == ReceiptStatus::Rejected);
    auto c = input(2); c.baseSeq = 20; assert(!s.Accept(c, 1));
}
void emergency_stale_state() {
    auto s = ready(); assert(s.Accept(tap(), 0));
    auto c = input(2, 2); c.kind = InputKind::Release; c.sceneEpoch = 999; c.baseSeq = 999;
    assert(s.Accept(c, 5)); assert(!s.Consume(6).owned);
}
void bad_values() {
    auto s = ready(); auto c = input(); c.pad.stickY = 81; assert(!s.Accept(c, 0));
    c = input(2); c.leaseMs = 501; assert(!s.Accept(c, 0));
    c = tap(3); c.steps[0].ticks = 0; assert(!s.Accept(c, 0));
    c = tap(4); c.steps.resize(9); assert(!s.Accept(c, 0));
    c = tap(5); c.steps = std::vector<InputStep>(8, {{}, 8}); assert(!s.Accept(c, 0));
    c = input(6); c.kind = InputKind::Renew; assert(!s.Accept(c, 0));
}
void busy_does_not_drop_action() {
    auto s = ready(); assert(s.Accept(tap(), 0)); assert(!s.Accept(tap(2), 0));
    assert(s.Consume(1).pressed & A); assert(s.Consume(51).released & A);
    assert(s.Receipt(1)->status == ReceiptStatus::Completed);
}
void bounded_receipts() {
    auto s = ready(); assert(s.Accept(tap(), 0));
    for (int i = 2; i < 200; ++i) { auto c = input(i); c.kind = InputKind::Renew; assert(s.Accept(c, 1)); }
    assert(s.Receipts().size() <= InputScheduler::MaxReceipts);
    assert(s.Receipt(1) != nullptr); assert(s.Consume(2).pressed & A);
    assert(s.Consume(52).released & A);
}
void renewed_sequence_deadline() {
    auto s = ready(); auto c = tap(); c.steps = {{{A, 0, 0}, 8}, {{A, 0, 0}, 8}, {{}, 8}};
    assert(s.Accept(c, 0));
    for (int i = 1; i < 20; ++i) {
        s.ObserveSample(i + 1, i * 100); auto r = input(i + 1); r.baseSeq = i + 1;
        r.kind = InputKind::Renew; assert(s.Accept(r, i * 100));
    }
    assert(!s.Consume(2000).owned);
    assert(std::string(s.Receipt(1)->reason) == "sequence_timeout");
}
int main(int argc, char** argv) {
    assert(argc == 2); std::string name = argv[1];
    #define CASE(n) if (name == #n) { n(); std::cout << #n << " OK\n"; return 0; }
    CASE(button_width) CASE(end_to_end_latency) CASE(one_tap) CASE(repeated_press) CASE(release_before_consume) CASE(latest_setpoint)
    CASE(duplicate_once) CASE(duplicate_does_not_renew) CASE(stale_owner) CASE(scene_change)
    CASE(context_change) CASE(watchdog_not_frames) CASE(old_sample) CASE(emergency_stale_state)
    CASE(bad_values) CASE(busy_does_not_drop_action) CASE(bounded_receipts) CASE(renewed_sequence_deadline)
    return 2;
}
