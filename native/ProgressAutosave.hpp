#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>

namespace zelda_ai {
enum AutosaveReason : uint8_t { SaveScene = 1, SaveResource = 2, SaveItem = 4, SaveChest = 8 };

struct SaveProgress {
    int rupees{}, health{}, magic{}, maxHealth{}, skullTokens{}, heartPieces{};
    uint16_t equipment{};
    uint32_t quests{};
    uint8_t abilities{};
    std::array<uint8_t, 24> items = [] { std::array<uint8_t, 24> a{}; a.fill(255); return a; }();
    std::array<int, 16> ammo{};
    std::array<int, 8> upgrades{};
    std::array<uint8_t, 20> dungeonItems{};
    std::array<int, 19> keys{};
};

class ProgressAutosave {
public:
    bool pending{}, initialized{}, inFlight{};
    int scene = -1, file = -1, inFlightScene = -1, inFlightFile = -1;
    uint8_t reasons{}, inFlightReasons{};
    uint64_t scheduledPeerSeq{};
    int64_t scheduledAt{}, lastGainAt{}, lastSubmittedAt{};
    SaveProgress previous{};

    void Reset() { *this = ProgressAutosave{}; }

    void Schedule(uint8_t reason, int nextScene, int nextFile, int64_t now, uint64_t peerSeq) {
        if (nextFile < 0 || nextFile > 2 || nextScene < 0) return;
        if (!pending) {
            pending = true;
            scheduledAt = now;
            scheduledPeerSeq = peerSeq;
        }
        reasons = static_cast<uint8_t>(reasons | reason);
        scene = nextScene;
        file = nextFile;
        if (reason & (SaveResource | SaveItem | SaveChest)) lastGainAt = now;
    }

    void Observe(const SaveProgress& current, int currentScene, int currentFile,
                 int64_t now, uint64_t peerSeq) {
        if (currentFile < 0 || currentFile > 2 || currentScene < 0) { Reset(); return; }
        if (initialized && file != currentFile) Reset();
        // First loaded values are a baseline, never an acquisition.
        uint8_t gain = 0;
        if (initialized) {
            if (current.rupees > previous.rupees || current.health > previous.health || current.magic > previous.magic)
                gain = SaveResource;
            for (size_t i = 0; i < current.ammo.size(); ++i)
                if (current.ammo[i] > previous.ammo[i]) gain = static_cast<uint8_t>(gain | SaveResource);
            if ((current.equipment & ~previous.equipment) || (current.quests & ~previous.quests)
                    || (current.abilities & ~previous.abilities) || current.maxHealth > previous.maxHealth
                    || current.skullTokens > previous.skullTokens || current.heartPieces > previous.heartPieces)
                gain = static_cast<uint8_t>(gain | SaveItem);
            for (size_t i = 0; i < current.items.size(); ++i)
                if (current.items[i] < 254 && current.items[i] != previous.items[i]) gain = static_cast<uint8_t>(gain | SaveItem);
            for (size_t i = 0; i < current.upgrades.size(); ++i)
                if (current.upgrades[i] > previous.upgrades[i]) gain = static_cast<uint8_t>(gain | SaveItem);
            for (size_t i = 0; i < current.dungeonItems.size(); ++i)
                if (current.dungeonItems[i] & ~previous.dungeonItems[i]) gain = static_cast<uint8_t>(gain | SaveItem);
            for (size_t i = 0; i < current.keys.size(); ++i)
                if (current.keys[i] > previous.keys[i]) gain = static_cast<uint8_t>(gain | SaveItem);
        }
        previous = current;
        initialized = true;
        scene = currentScene;
        file = currentFile;
        if (gain) Schedule(gain, currentScene, currentFile, now, peerSeq);
    }

    bool PeerExpired(int64_t now, uint64_t peerSeq, int64_t lastPeerAt) const {
        const bool confirmed = peerSeq > scheduledPeerSeq;
        const bool fresh = lastPeerAt > 0 && now - lastPeerAt <= 15000;
        return pending && (!confirmed || !fresh) && now - scheduledAt > 5000;
    }

    bool Ready(int64_t now, uint64_t peerSeq, int64_t lastPeerAt, bool safe,
               int currentScene, int currentFile) const {
        if (!pending || inFlight || !safe || currentScene != scene || currentFile != file
                || peerSeq <= scheduledPeerSeq || lastPeerAt <= 0 || now - lastPeerAt > 15000) return false;
        if (lastSubmittedAt > 0 && now - lastSubmittedAt < 1500) return false;
        return !(reasons & (SaveResource | SaveItem | SaveChest))
            || now - lastGainAt >= 750 || now - scheduledAt >= 5000;
    }

    void CancelPending() { pending = false; reasons = 0; scheduledAt = lastGainAt = 0; }

    uint8_t Submit(int64_t now) {
        inFlightReasons = reasons;
        inFlightScene = scene;
        inFlightFile = file;
        inFlight = true;
        lastSubmittedAt = now;
        CancelPending();
        return inFlightReasons;
    }

    uint8_t Confirm(int savedFile) {
        if (!inFlight || inFlightFile != savedFile) return 0;
        const uint8_t result = inFlightReasons;
        inFlight = false;
        inFlightReasons = 0;
        return result;
    }
};
}
