// Integration target: HarbourMasters/Shipwright d30fc192f2eb01ceea45bd1e12de61636cafbf86.
// Compile with SoH; this is not a DLL for an unmodified release executable.
#include "ZeldaAiBridge.h"
#include "InputLease.hpp"
#include <SDL2/SDL_net.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include "soh/Enhancements/game-interactor/GameInteractor.h"
#include "soh/ShipInit.hpp"
extern "C" {
#include "global.h"
extern PlayState* gPlayState;
extern SaveContext gSaveContext;
}

namespace {
using json = nlohmann::json;
constexpr const char* REVISION = "d30fc192f2eb01ceea45bd1e12de61636cafbf86";
int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct BridgeData {
    std::mutex mutex;
    bool initialized = false;
    UDPsocket socket = nullptr;
    UDPpacket* packet = nullptr;
    IPaddress destination{};
    std::string token;
    std::string instance;
    uint64_t seq = 0;
    uint64_t sceneEpoch = 0;
    uint64_t eventSeq = 0;
    int64_t lastSent = 0;
    bool playable = false;
    std::deque<json> events;
    zelda_ai::InputLease lease;

    void Init() {
        if (initialized) return;
        initialized = true;
        const char* value = std::getenv("ZELDA_BRIDGE_TOKEN");
        if (!value || std::string(value).size() < 24) return;
        token = value;
        int port = 8766;
        if (const char* raw = std::getenv("ZELDA_BRIDGE_PORT")) {
            try { port = std::stoi(raw); } catch (...) { return; }
        }
        if (port < 1024 || port > 65535 || SDLNet_Init() < 0) return;
        if (SDLNet_ResolveHost(&destination, "127.0.0.1", static_cast<Uint16>(port)) < 0) return;
        socket = SDLNet_UDP_Open(0);
        packet = SDLNet_AllocPacket(60000);
        if (!socket || !packet) return;
        instance = std::to_string(std::random_device{}()) + "-" + std::to_string(NowMs());
    }
    void Poll() {
        if (!socket || !packet) return;
        for (int count = 0; count < 16 && SDLNet_UDP_Recv(socket, packet) > 0; ++count) {
            if (packet->address.host != destination.host || packet->address.port != destination.port ||
                packet->len > 4096) continue;
            try {
                auto data = json::parse(packet->data, packet->data + packet->len);
                if (data.at("protocol") != 1 || data.at("token") != token || data.at("instance_id") != instance)
                    continue;
                zelda_ai::InputCommand command;
                command.seq = data.at("seq").get<uint64_t>();
                command.sceneEpoch = data.at("scene_epoch").get<uint64_t>();
                command.baseSeq = data.at("base_seq").get<uint64_t>();
                int buttonValue = data.at("buttons").get<int>();
                if (buttonValue < 0 || buttonValue > 65535) continue;
                command.buttons = static_cast<uint16_t>(buttonValue);
                command.stickX = data.at("stick_x").get<int>();
                command.stickY = data.at("stick_y").get<int>();
                command.leaseMs = data.at("lease_ms").get<int>();
                command.active = data.at("active").get<bool>() && playable;
                lease.Apply(command, sceneEpoch, seq, NowMs());
            } catch (const json::exception&) { /* Invalid datagrams cannot affect the game. */ }
        }
    }
};
// Deliberately process-lifetime; avoids shutdown-order calls after SDL has stopped.
BridgeData& Data() { static BridgeData data; return data; }

void Event(const char* kind, const std::string& detail) {
    auto& bridge = Data();
    std::scoped_lock lock(bridge.mutex);
    bridge.events.push_back({{"id", std::to_string(++bridge.eventSeq)}, {"kind", kind}, {"detail", detail}});
    if (bridge.events.size() > 16) bridge.events.pop_front();
}

void Snapshot() {
    auto& bridge = Data();
    std::scoped_lock lock(bridge.mutex);
    bridge.Init();
    if (!bridge.socket || !bridge.packet || NowMs() - bridge.lastSent < 200) return;
    bridge.lastSent = NowMs();
    bridge.playable = GameInteractor::IsSaveLoaded(true) && gPlayState != nullptr;
    if (!bridge.playable) bridge.lease.Release();
    json state = {{"protocol", 1}, {"token", bridge.token}, {"source", "soh"},
        {"instance_id", bridge.instance}, {"seq", ++bridge.seq}, {"scene_epoch", bridge.sceneEpoch},
        {"scene", -1}, {"room", -1}, {"in_game", bridge.playable}, {"player", nullptr},
        {"paused", false}, {"events", bridge.events}, {"last_command_seq", bridge.lease.lastCommandSeq},
        {"upstream_revision", REVISION}};
    if (bridge.playable) {
        Player* player = GET_PLAYER(gPlayState);
        if (player) {
            auto& pos = player->actor.world.pos;
            state["scene"] = gPlayState->sceneNum;
            state["room"] = gPlayState->roomCtx.curRoom.num;
            state["paused"] = gPlayState->pauseCtx.state != 0;
            state["message_id"] = gPlayState->msgCtx.textId;
            state["player"] = {{"position", {pos.x, pos.y, pos.z}}, {"yaw", player->actor.shape.rot.y},
                {"health", gSaveContext.health}, {"max_health", gSaveContext.healthCapacity},
                {"rupees", gSaveContext.rupees}, {"magic", gSaveContext.magic},
                {"age", gSaveContext.linkAge == 0 ? "adult" : "child"}};
            state["camera_eye"] = {gPlayState->view.eye.x, gPlayState->view.eye.y, gPlayState->view.eye.z};
            state["camera_at"] = {gPlayState->view.at.x, gPlayState->view.at.y, gPlayState->view.at.z};
            state["inventory"] = json::array();
            state["equipped"] = json::array();
            for (auto item : gSaveContext.inventory.items) state["inventory"].push_back(item);
            for (auto item : gSaveContext.equips.buttonItems) state["equipped"].push_back(item);
        }
    }
    std::string serialized = state.dump();
    if (serialized.size() > 59000) return;
    bridge.packet->address = bridge.destination;
    bridge.packet->len = static_cast<int>(serialized.size());
    std::copy(serialized.begin(), serialized.end(), bridge.packet->data);
    SDLNet_UDP_Send(bridge.socket, -1, bridge.packet);
}

void RegisterZeldaAiBridge() {
    static bool registered = false;
    if (registered) return;
    registered = true;
    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnGameStateMainStart>(Snapshot);
    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnSceneInit>([](int16_t scene) {
        {
            auto& bridge = Data();
            std::scoped_lock lock(bridge.mutex);
            ++bridge.sceneEpoch;
            bridge.lease.Release();
        }
        Event("scene_changed", std::to_string(scene));
    });
    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnEnemyDefeat>([](void* rawActor) {
        auto* actor = static_cast<Actor*>(rawActor);
        if (actor) Event("enemy_defeated", std::to_string(actor->id));
    });
    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnBossDefeat>([](void* rawActor) {
        auto* actor = static_cast<Actor*>(rawActor);
        if (actor) Event("boss_defeated", std::to_string(actor->id));
    });
}
static RegisterShipInitFunc registration(RegisterZeldaAiBridge);
} // namespace

extern "C" void ZeldaAiBridge_OverrideInput(int32_t controller, uint16_t* buttons, int8_t* stickX, int8_t* stickY) {
    if (controller != 0 || !buttons || !stickX || !stickY) return;
    auto& bridge = Data();
    std::scoped_lock lock(bridge.mutex);
    bridge.Poll();
    bridge.lease.Read(NowMs(), *buttons, *stickX, *stickY);
}
