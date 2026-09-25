// Integration target: HarbourMasters/Shipwright d30fc192f2eb01ceea45bd1e12de61636cafbf86.
// Compile with SoH; this is not a DLL for an unmodified release executable.
#include "ZeldaAiBridge.h"
#include "InputScheduler.hpp"
#include "ActorRegistry.hpp"
#include <SDL2/SDL_net.h>
#include <nlohmann/json.hpp>
#include <libultraship/bridge/consolevariablebridge.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>
#include "soh/Enhancements/game-interactor/GameInteractor.h"
#include "soh/Enhancements/item-tables/ItemTableTypes.h"
#include "soh/ShipInit.hpp"
#include "soh/cvar_prefixes.h"
#include "soh/ActorDB.h"
#include "soh/util.h"
extern "C" {
#include "global.h"
#include "message_data_fmt.h"
extern PlayState* gPlayState;
extern SaveContext gSaveContext;
}

// Shipwright's accessibility module already owns the language-specific character remapping.
// Reuse it instead of maintaining a second message table or OCR path.
std::string Message_TTS_Decode(uint8_t* sourceBuf, uint16_t startOffset, uint16_t size);

namespace {
using json = nlohmann::json;
constexpr const char* REVISION = "d30fc192f2eb01ceea45bd1e12de61636cafbf86";
constexpr size_t MAX_EVENTS = 64;
constexpr const char* BRIDGE_BUILD = "rt-input-v2.4";
constexpr size_t MAX_NEARBY_ACTORS = 24;
constexpr size_t MAX_ROOM_ACTORS = 64;
constexpr float MAX_NEARBY_ACTOR_DISTANCE = 1400.0f;

int64_t NowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int64_t NowMs() {
    return NowUs() / 1000;
}

const char* DoActionName(uint16_t action) {
    switch (action) {
        case DO_ACTION_ATTACK: return "attack";
        case DO_ACTION_CHECK: return "check";
        case DO_ACTION_ENTER: return "enter";
        case DO_ACTION_RETURN: return "return";
        case DO_ACTION_OPEN: return "open";
        case DO_ACTION_JUMP: return "jump";
        case DO_ACTION_DECIDE: return "decide";
        case DO_ACTION_DIVE: return "dive";
        case DO_ACTION_FASTER: return "faster";
        case DO_ACTION_THROW: return "throw";
        case DO_ACTION_CLIMB: return "climb";
        case DO_ACTION_DROP: return "drop";
        case DO_ACTION_DOWN: return "down";
        case DO_ACTION_SAVE: return "save";
        case DO_ACTION_SPEAK: return "speak";
        case DO_ACTION_NEXT: return "next";
        case DO_ACTION_GRAB: return "grab";
        case DO_ACTION_STOP: return "stop";
        case DO_ACTION_PUTAWAY: return "putaway";
        case DO_ACTION_REEL: return "reel";
        case DO_ACTION_NONE: return "none";
        default: return "other";
    }
}

const char* TextStateName(uint8_t state) {
    switch (state) {
        case TEXT_STATE_NONE: return "none";
        case TEXT_STATE_DONE_HAS_NEXT: return "done_has_next";
        case TEXT_STATE_CLOSING: return "closing";
        case TEXT_STATE_DONE_FADING: return "displaying";
        case TEXT_STATE_CHOICE: return "choice";
        case TEXT_STATE_EVENT: return "event";
        case TEXT_STATE_DONE: return "done";
        case TEXT_STATE_SONG_DEMO_DONE: return "song_demo_done";
        case TEXT_STATE_8: return "ocarina_result";
        case TEXT_STATE_9: return "ocarina_input";
        case TEXT_STATE_AWAITING_NEXT: return "awaiting_next";
        default: return "unknown";
    }
}

bool TextCanAdvance(uint8_t state) {
    return state == TEXT_STATE_DONE_HAS_NEXT || state == TEXT_STATE_CHOICE ||
           state == TEXT_STATE_EVENT || state == TEXT_STATE_DONE ||
           state == TEXT_STATE_SONG_DEMO_DONE || state == TEXT_STATE_AWAITING_NEXT;
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
    uint64_t fullSeq = 0, captureTick = 0, contextEpoch = 0, eventAck = 0;
    int previousMode = -1;
    bool forceFull = true;
    OSContPad lastDelivered{};
    bool wasOwned = false;
    zelda_ai::ActorRegistry actors;
    bool playable = false;
    int16_t lastScene = -1;
    int16_t lastRoom = -1;
    uint16_t doAction = DO_ACTION_NONE;
    std::deque<json> events;
    zelda_ai::InputScheduler scheduler;

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
        int count = 0;
        for (; count < 64 && SDLNet_UDP_Recv(socket, packet) > 0; ++count) {
            if (packet->address.host != destination.host || packet->address.port != destination.port ||
                packet->len > 8192) continue;
            try {
                auto data = json::parse(packet->data, packet->data + packet->len);
                if (data.at("protocol") != 2 || data.at("token") != token || data.at("instance_id") != instance)
                    continue;
                auto number = [&](const char* name) -> uint64_t {
                    const auto& v = data.at(name);
                    if (!v.is_number_unsigned()) throw std::runtime_error("invalid sequence");
                    return v.get<uint64_t>();
                };
                const std::string kind = data.at("kind").get<std::string>();
                if (kind == "observe_ack") {
                    eventAck = std::max(eventAck, std::min(number("event_cursor"), eventSeq));
                    if (data.value("request_full", false)) forceFull = true;
                    continue;
                }
                zelda_ai::ScheduledInput command;
                command.seq = number("seq");
                command.ownerEpoch = number("owner_epoch");
                command.sceneEpoch = number("scene_epoch");
                command.contextEpoch = number("context_epoch");
                command.baseSeq = number("base_seq");
                if (kind == "release") command.kind = zelda_ai::InputKind::Release;
                else if (kind == "cancel") command.kind = zelda_ai::InputKind::Cancel;
                else if (kind == "renew") command.kind = zelda_ai::InputKind::Renew;
                else if (kind == "sequence") command.kind = zelda_ai::InputKind::Sequence;
                else if (kind == "setpoint") command.kind = zelda_ai::InputKind::Setpoint;
                else continue;
                auto pad = [](const json& row) -> zelda_ai::PadState {
                    const int buttons = row.at("buttons").get<int>();
                    if (buttons < 0 || buttons > 65535) throw std::runtime_error("invalid buttons");
                    return {static_cast<uint16_t>(buttons), row.at("stick_x").get<int>(),
                            row.at("stick_y").get<int>()};
                };
                command.pad = pad(data);
                command.leaseMs = data.at("lease_ms").get<int>();
                command.clientSentUs = data.value("client_sent_us", int64_t{-1});
                if (command.kind == zelda_ai::InputKind::Sequence) {
                    const auto& steps = data.at("steps");
                    if (!steps.is_array() || steps.size() > 8) continue;
                    const int edges = data.at("edge_buttons").get<int>();
                    if (edges < 0 || edges > 65535) continue;
                    command.edgeButtons = static_cast<uint16_t>(edges);
                    for (const auto& row : steps) command.steps.push_back({pad(row), row.at("ticks").get<int>()});
                }
                if (!playable && command.kind != zelda_ai::InputKind::Release) continue;
                const auto nowUs = NowUs();
                scheduler.Accept(command, nowUs / 1000, nowUs);
            } catch (const std::exception&) {
                // Malformed, unauthorized or out-of-context packets never drive the controller.
            }
        }
        if (count == 64) scheduler.Release("input_queue_overflow");
    }
};

// Deliberately process-lifetime; avoids shutdown-order calls after SDL has stopped.
BridgeData& Data() {
    static BridgeData data;
    return data;
}

void PushEventLocked(BridgeData& bridge, const char* kind, const std::string& detail) {
    bridge.events.push_back({
        {"id", std::to_string(++bridge.eventSeq)},
        {"kind", kind},
        {"detail", detail.substr(0, 1000)},
    });
    while (bridge.events.size() > MAX_EVENTS) bridge.events.pop_front();
}

void Event(const char* kind, const std::string& detail) {
    auto& bridge = Data();
    std::scoped_lock lock(bridge.mutex);
    PushEventLocked(bridge, kind, detail);
}

bool ItemUsesAmmo(int item) {
    switch (item) {
        case ITEM_STICK:
        case ITEM_NUT:
        case ITEM_BOMB:
        case ITEM_BOW:
        case ITEM_SLINGSHOT:
        case ITEM_BOMBCHU:
        case ITEM_BEAN:
            return true;
        default:
            return false;
    }
}

json ProgressJson() {
    json questItems = json::array();
    for (int quest = QUEST_MEDALLION_FOREST; quest <= QUEST_SKULL_TOKEN; ++quest) {
        if (CHECK_QUEST_ITEM(quest)) {
            questItems.push_back(SohUtils::GetQuestItemName(quest));
        }
    }

    json ownedEquipment = json::array();
    json equipment = json::array();
    struct EquipmentEntry {
        uint16_t flag;
        int item;
        int type;
        int value;
        const char* typeName;
    };
    const EquipmentEntry equipmentItems[] = {
        {EQUIP_FLAG_SWORD_KOKIRI, ITEM_SWORD_KOKIRI, EQUIP_TYPE_SWORD, EQUIP_VALUE_SWORD_KOKIRI, "sword"},
        {EQUIP_FLAG_SWORD_MASTER, ITEM_SWORD_MASTER, EQUIP_TYPE_SWORD, EQUIP_VALUE_SWORD_MASTER, "sword"},
        {EQUIP_FLAG_SWORD_BGS, ITEM_SWORD_BGS, EQUIP_TYPE_SWORD, EQUIP_VALUE_SWORD_BIGGORON, "sword"},
        {EQUIP_FLAG_SHIELD_DEKU, ITEM_SHIELD_DEKU, EQUIP_TYPE_SHIELD, EQUIP_VALUE_SHIELD_DEKU, "shield"},
        {EQUIP_FLAG_SHIELD_HYLIAN, ITEM_SHIELD_HYLIAN, EQUIP_TYPE_SHIELD, EQUIP_VALUE_SHIELD_HYLIAN, "shield"},
        {EQUIP_FLAG_SHIELD_MIRROR, ITEM_SHIELD_MIRROR, EQUIP_TYPE_SHIELD, EQUIP_VALUE_SHIELD_MIRROR, "shield"},
        {EQUIP_FLAG_TUNIC_KOKIRI, ITEM_TUNIC_KOKIRI, EQUIP_TYPE_TUNIC, EQUIP_VALUE_TUNIC_KOKIRI, "tunic"},
        {EQUIP_FLAG_TUNIC_GORON, ITEM_TUNIC_GORON, EQUIP_TYPE_TUNIC, EQUIP_VALUE_TUNIC_GORON, "tunic"},
        {EQUIP_FLAG_TUNIC_ZORA, ITEM_TUNIC_ZORA, EQUIP_TYPE_TUNIC, EQUIP_VALUE_TUNIC_ZORA, "tunic"},
        {EQUIP_FLAG_BOOTS_KOKIRI, ITEM_BOOTS_KOKIRI, EQUIP_TYPE_BOOTS, EQUIP_VALUE_BOOTS_KOKIRI, "boots"},
        {EQUIP_FLAG_BOOTS_IRON, ITEM_BOOTS_IRON, EQUIP_TYPE_BOOTS, EQUIP_VALUE_BOOTS_IRON, "boots"},
        {EQUIP_FLAG_BOOTS_HOVER, ITEM_BOOTS_HOVER, EQUIP_TYPE_BOOTS, EQUIP_VALUE_BOOTS_HOVER, "boots"},
    };
    for (const auto& row : equipmentItems) {
        if (gSaveContext.inventory.equipment & row.flag) {
            const auto& name = SohUtils::GetItemName(row.item);
            ownedEquipment.push_back(name);
            equipment.push_back({
                {"item_id", row.item},
                {"name", name},
                {"equipment_type", row.typeName},
                {"value", row.value},
                {"equipped", CUR_EQUIP_VALUE(row.type) == row.value},
            });
        }
    }

    json dungeonItems = json::array();
    const int mapIndex = gSaveContext.mapIndex;
    int smallKeys = 0;
    if (mapIndex < static_cast<int>(ARRAY_COUNT(gSaveContext.inventory.dungeonItems))) {
        if (CHECK_DUNGEON_ITEM(DUNGEON_KEY_BOSS, mapIndex)) dungeonItems.push_back("Boss Key");
        if (CHECK_DUNGEON_ITEM(DUNGEON_COMPASS, mapIndex)) dungeonItems.push_back("Compass");
        if (CHECK_DUNGEON_ITEM(DUNGEON_MAP, mapIndex)) dungeonItems.push_back("Dungeon Map");
    }
    if (mapIndex < static_cast<int>(ARRAY_COUNT(gSaveContext.inventory.dungeonKeys))) {
        smallKeys = std::max<int>(gSaveContext.inventory.dungeonKeys[mapIndex], 0);
    }

    return {
        {"quest_items", questItems},
        {"owned_equipment", ownedEquipment},
        {"equipment", equipment},
        {"upgrade_levels", {
            {"quiver", CUR_UPG_VALUE(UPG_QUIVER)},
            {"bomb_bag", CUR_UPG_VALUE(UPG_BOMB_BAG)},
            {"strength", CUR_UPG_VALUE(UPG_STRENGTH)},
            {"scale", CUR_UPG_VALUE(UPG_SCALE)},
            {"wallet", CUR_UPG_VALUE(UPG_WALLET)},
            {"bullet_bag", CUR_UPG_VALUE(UPG_BULLET_BAG)},
            {"sticks", CUR_UPG_VALUE(UPG_STICKS)},
            {"nuts", CUR_UPG_VALUE(UPG_NUTS)},
        }},
        {"heart_pieces", static_cast<int>((gSaveContext.inventory.questItems >> 28) & 0xF)},
        {"skull_tokens", std::max<int>(gSaveContext.inventory.gsTokens, 0)},
        {"magic_acquired", gSaveContext.isMagicAcquired != 0},
        {"double_magic", gSaveContext.isDoubleMagicAcquired != 0},
        {"double_defense", gSaveContext.isDoubleDefenseAcquired != 0},
        {"map_index", mapIndex},
        {"dungeon_items", dungeonItems},
        {"small_keys", smallKeys},
    };
}

const char* ActorCategoryName(uint8_t category) {
    switch (category) {
        case ACTORCAT_SWITCH: return "switch";
        case ACTORCAT_BG: return "background";
        case ACTORCAT_PLAYER: return "player";
        case ACTORCAT_EXPLOSIVE: return "explosive";
        case ACTORCAT_NPC: return "npc";
        case ACTORCAT_ENEMY: return "enemy";
        case ACTORCAT_PROP: return "prop";
        case ACTORCAT_ITEMACTION: return "item_action";
        case ACTORCAT_MISC: return "misc";
        case ACTORCAT_BOSS: return "boss";
        case ACTORCAT_DOOR: return "door";
        case ACTORCAT_CHEST: return "chest";
        default: return "unknown";
    }
}

json NavigationProbes(Player* player) {
    static const char* names[] = {
        "forward", "forward_right", "right", "back_right",
        "back", "back_left", "left", "forward_left",
    };
    static const int16_t offsets[] = {
        0x0000, 0x2000, 0x4000, 0x6000,
        static_cast<int16_t>(0x8000), static_cast<int16_t>(0xA000),
        static_cast<int16_t>(0xC000), static_cast<int16_t>(0xE000),
    };
    static const float distances[] = { 70.0f, 140.0f };
    json result = json::array();
    if (!player) return result;

    const float baseFloor = player->actor.floorHeight;
    for (float probeDistance : distances) {
        for (size_t i = 0; i < ARRAY_COUNT(offsets); ++i) {
            const int16_t yaw = static_cast<int16_t>(player->actor.shape.rot.y + offsets[i]);
            const float radians = static_cast<float>(yaw) * 3.14159265358979323846f / 32768.0f;
            const float dirX = std::sin(radians);
            const float dirZ = std::cos(radians);

            Vec3f floorPos = player->actor.world.pos;
            floorPos.x += dirX * probeDistance;
            floorPos.z += dirZ * probeDistance;
            floorPos.y += 180.0f;
            CollisionPoly* floorPoly = nullptr;
            s32 floorBgId = BGCHECK_SCENE;
            const float floorY = BgCheck_EntityRaycastFloor3(
                &gPlayState->colCtx, &floorPoly, &floorBgId, &floorPos);
            const bool floorFound = floorPoly != nullptr && floorY > BGCHECK_Y_MIN + 1.0f;

            Vec3f wallStart = player->actor.world.pos;
            wallStart.y += 26.0f;
            Vec3f wallEnd = wallStart;
            wallEnd.x += dirX * probeDistance;
            wallEnd.z += dirZ * probeDistance;
            Vec3f wallHitPos{};
            CollisionPoly* wallPoly = nullptr;
            s32 wallBgId = BGCHECK_SCENE;
            const bool wallHit = BgCheck_EntityLineTest1(
                &gPlayState->colCtx, &wallStart, &wallEnd, &wallHitPos, &wallPoly,
                true, false, false, true, &wallBgId) != 0;
            const float wallDistance = wallHit
                ? std::sqrt(
                    (wallHitPos.x - wallStart.x) * (wallHitPos.x - wallStart.x) +
                    (wallHitPos.y - wallStart.y) * (wallHitPos.y - wallStart.y) +
                    (wallHitPos.z - wallStart.z) * (wallHitPos.z - wallStart.z))
                : 0.0f;
            const int wallFlags = wallHit && wallPoly
                ? SurfaceType_GetWallFlags(&gPlayState->colCtx, wallPoly, wallBgId) : 0;

            result.push_back({
                {"direction", names[i]},
                {"distance", probeDistance},
                {"floor_found", floorFound},
                {"floor_y", floorFound ? json(floorY) : json(nullptr)},
                {"delta_y", floorFound ? json(floorY - baseFloor) : json(nullptr)},
                {"floor_type", floorFound
                    ? json(SurfaceType_GetFloorType(&gPlayState->colCtx, floorPoly, floorBgId))
                    : json(nullptr)},
                {"wall_hit", wallHit},
                {"wall_distance", wallHit ? json(wallDistance) : json(nullptr)},
                {"wall_flags", wallFlags},
            });
        }
    }
    return result;
}

json ActorJson(Actor* actor, Player* player, bool metadata = true) {
    if (!actor || !player) return nullptr;
    std::string actorName;
    std::string actorDescription;
    if (metadata && ActorDB::Instance != nullptr) {
        const auto& entry = ActorDB::Instance->RetrieveEntry(actor->id);
        if (entry.entry.valid) {
            actorName = entry.name.substr(0, 96);
            actorDescription = entry.desc.substr(0, 200);
        }
    }
    const auto& a = actor->world.pos;
    const auto& p = player->actor.world.pos;
    const float dx = a.x - p.x;
    const float dy = a.y - p.y;
    const float dz = a.z - p.z;
    const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    return {
        {"actor_id", actor->id},
        {"actor_uid", Data().actors.Get(actor)},
        {"yaw", actor->shape.rot.y},
        {"name", actorName},
        {"description", actorDescription},
        {"category", actor->category},
        {"category_name", ActorCategoryName(actor->category)},
        {"room", actor->room},
        {"params", actor->params},
        {"position", {a.x, a.y, a.z}},
        {"focus_position", {actor->focus.pos.x, actor->focus.pos.y, actor->focus.pos.z}},
        {"distance", distance},
        {"targeted", actor->isTargeted != 0},
        {"drawn", actor->isDrawn != 0},
        {"text_id", actor->textId},
    };
}

Actor* ContextActor(Player* player, uint16_t doAction) {
    if (!player) return nullptr;
    switch (doAction) {
        case DO_ACTION_OPEN:
        case DO_ACTION_ENTER:
            if (player->doorActor) return player->doorActor;
            break;
        case DO_ACTION_GRAB:
        case DO_ACTION_THROW:
        case DO_ACTION_DROP:
            if (player->interactRangeActor) return player->interactRangeActor;
            if (player->heldActor) return player->heldActor;
            break;
        case DO_ACTION_SPEAK:
        case DO_ACTION_CHECK:
            if (player->focusActor) return player->focusActor;
            break;
        default:
            break;
    }
    if (player->interactRangeActor && player->interactRangeActor != &player->actor) {
        return player->interactRangeActor;
    }
    if (player->focusActor && player->focusActor != &player->actor) {
        return player->focusActor;
    }
    if (player->doorActor && player->doorActor != &player->actor) {
        return player->doorActor;
    }
    return nullptr;
}

int ActorObservationPriority(Actor* actor, Actor* contextActor) {
    if (!actor) return 99;
    if (actor == contextActor || actor->isTargeted) return 0;
    // Exits must survive observation caps even when they are off camera or far across the room.
    if (actor->category == ACTORCAT_DOOR) return 1;
    if (actor->textId != 0 || actor->category == ACTORCAT_NPC || actor->category == ACTORCAT_BOSS ||
        actor->category == ACTORCAT_CHEST) return 2;
    if (actor->category == ACTORCAT_ENEMY) return 3;
    if (actor->category == ACTORCAT_SWITCH || actor->category == ACTORCAT_BG ||
        actor->category == ACTORCAT_PROP || actor->category == ACTORCAT_ITEMACTION) return 4;
    return 5;
}

json NearbyActors(Player* player) {
    struct Candidate {
        int priority;
        float distance;
        Actor* actor;
    };
    std::vector<Candidate> candidates;
    const int room = gPlayState->roomCtx.curRoom.num;
    Actor* contextActor = ContextActor(player, Data().doAction);
    for (int category = 0; category < ACTORCAT_MAX; ++category) {
        for (Actor* actor = gPlayState->actorCtx.actorLists[category].head; actor != nullptr; actor = actor->next) {
            if (actor == &player->actor || !actor->isDrawn || (actor->room != -1 && actor->room != room)) continue;
            const auto& a = actor->world.pos;
            const auto& p = player->actor.world.pos;
            const float dx = a.x - p.x;
            const float dy = a.y - p.y;
            const float dz = a.z - p.z;
            const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (std::isfinite(distance) && distance <= MAX_NEARBY_ACTOR_DISTANCE) {
                candidates.push_back({ActorObservationPriority(actor, contextActor), distance, actor});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.priority != rhs.priority) return lhs.priority < rhs.priority;
        return lhs.distance < rhs.distance;
    });
    json result = json::array();
    for (size_t i = 0; i < candidates.size() && i < MAX_NEARBY_ACTORS; ++i) {
        result.push_back(ActorJson(candidates[i].actor, player));
    }
    return result;
}

json RoomActors(Player* player, bool metadata = true) {
    struct Candidate {
        int priority;
        float distance;
        Actor* actor;
    };
    std::vector<Candidate> candidates;
    const int room = gPlayState->roomCtx.curRoom.num;
    Actor* contextActor = ContextActor(player, Data().doAction);
    for (int category = 0; category < ACTORCAT_MAX; ++category) {
        for (Actor* actor = gPlayState->actorCtx.actorLists[category].head; actor != nullptr; actor = actor->next) {
            if (actor == &player->actor || (actor->room != -1 && actor->room != room)) continue;
            const auto& a = actor->world.pos;
            const auto& p = player->actor.world.pos;
            const float dx = a.x - p.x;
            const float dy = a.y - p.y;
            const float dz = a.z - p.z;
            const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (!std::isfinite(distance)) continue;
            candidates.push_back({ActorObservationPriority(actor, contextActor), distance, actor});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.priority != rhs.priority) return lhs.priority < rhs.priority;
        return lhs.distance < rhs.distance;
    });

    json actors = json::array();
    for (size_t i = 0; i < candidates.size() && i < MAX_ROOM_ACTORS; ++i) {
        actors.push_back(ActorJson(candidates[i].actor, player, metadata));
    }
    return {
        {"actors", actors},
        {"count", candidates.size()},
        {"truncated", candidates.size() > MAX_ROOM_ACTORS},
    };
}

std::vector<std::string> DecodeChoices(MessageContext* msgCtx, int count) {
    std::vector<std::string> choices;
    if (!msgCtx || count <= 0 || msgCtx->decodedTextLen == 0) return choices;
    const uint16_t length = std::min<uint16_t>(msgCtx->decodedTextLen, sizeof(msgCtx->msgBufDecoded));
    uint16_t cursor = 0;
    while (cursor < length && msgCtx->msgBufDecoded[cursor] != MESSAGE_TWO_CHOICE &&
           msgCtx->msgBufDecoded[cursor] != MESSAGE_THREE_CHOICE) {
        ++cursor;
    }
    if (cursor >= length) return choices;
    ++cursor;
    for (int option = 0; option < count && cursor < length; ++option) {
        uint16_t end = cursor;
        while (end < length && msgCtx->msgBufDecoded[end] != MESSAGE_NEWLINE) ++end;
        if (end > cursor) {
            choices.push_back(Message_TTS_Decode(msgCtx->msgBufDecoded, cursor, end - cursor));
        } else {
            choices.emplace_back("");
        }
        cursor = end < length ? end + 1 : end;
    }
    return choices;
}

json DialogueJson(Player* player) {
    MessageContext* msgCtx = &gPlayState->msgCtx;
    const bool active = msgCtx->msgLength != 0;
    const uint8_t stateCode = Message_GetState(msgCtx);
    std::string text;
    if (active && msgCtx->decodedTextLen > 0) {
        const uint16_t length = std::min<uint16_t>(msgCtx->decodedTextLen, sizeof(msgCtx->msgBufDecoded));
        text = Message_TTS_Decode(msgCtx->msgBufDecoded, 0, length);
        if (text.size() > 4096) text.resize(4096);
    }
    int choiceCount = 0;
    if (active && stateCode == TEXT_STATE_CHOICE) {
        choiceCount = std::clamp<int>(msgCtx->choiceNum, 0, 3);
    }
    auto choices = DecodeChoices(msgCtx, choiceCount);
    json speaker = nullptr;
    if (active && msgCtx->talkActor) speaker = ActorJson(msgCtx->talkActor, player);
    return {
        {"active", active},
        {"text_id", active ? json(msgCtx->textId) : json(nullptr)},
        {"text", text},
        {"state", TextStateName(stateCode)},
        {"state_code", stateCode},
        {"message_mode", msgCtx->msgMode},
        {"can_advance", active && TextCanAdvance(stateCode)},
        {"choice_count", choiceCount},
        {"choice_index", choiceCount > 0 ? std::min<int>(msgCtx->choiceIndex, choiceCount - 1) : 0},
        {"choices", choices},
        {"speaker", speaker},
    };
}

void Snapshot() {
    auto& bridge = Data();
    std::unique_lock lock(bridge.mutex);
    bridge.Init();
    if (!bridge.socket || !bridge.packet) return;
    const auto now = NowMs();
    bridge.playable = GameInteractor::IsSaveLoaded(true) && gPlayState != nullptr;
    int mode = 0;
    if (bridge.playable) {
        const int16_t scene = gPlayState->sceneNum;
        const int16_t room = gPlayState->roomCtx.curRoom.num;
        if (bridge.lastScene != scene || bridge.lastRoom != room) {
            if (bridge.lastScene >= 0 && bridge.lastScene != scene)
                PushEventLocked(bridge, "scene_changed", std::to_string(bridge.lastScene) + "->" + std::to_string(scene));
            else if (bridge.lastRoom >= 0 && bridge.lastRoom != room)
                PushEventLocked(bridge, "room_changed", std::to_string(bridge.lastRoom) + "->" + std::to_string(room));
            ++bridge.sceneEpoch;
            bridge.lastScene = scene;
            bridge.lastRoom = room;
            bridge.forceFull = true;
        }
        mode = 1 | (gPlayState->pauseCtx.state != 0 ? 2 : 0)
            | (gPlayState->msgCtx.msgLength != 0 ? 4 : 0)
            | ((gPlayState->csCtx.state != CS_STATE_IDLE || Player_InCsMode(gPlayState) != 0) ? 8 : 0)
            | (gPlayState->gameOverCtx.state != 0 ? 16 : 0)
            | (gPlayState->msgCtx.ocarinaMode != 0 ? 32 : 0);
    } else {
        bridge.scheduler.Release("game_not_ready");
        bridge.lastScene = bridge.lastRoom = -1;
    }
    if (mode != bridge.previousMode) {
        ++bridge.contextEpoch;
        bridge.previousMode = mode;
        bridge.forceFull = true;
    }
    bridge.scheduler.SetContext(bridge.sceneEpoch, bridge.contextEpoch);
    bridge.Poll();
    const bool full = bridge.forceFull || now - bridge.lastSent >= 200 || !bridge.fullSeq;
    if (!bridge.playable && !full) return;
    ++bridge.captureTick;
    const uint64_t sampleSeq = ++bridge.seq;
    if (full) {
        bridge.lastSent = now;
        bridge.fullSeq = sampleSeq;
        bridge.forceFull = false;
    }
    bridge.scheduler.ObserveSample(sampleSeq, now);
    json receipts = json::array();
    for (const auto& r : bridge.scheduler.Receipts()) {
        receipts.push_back({{"seq", r.seq}, {"owner_epoch", r.ownerEpoch},
            {"status", zelda_ai::ReceiptStatusName(r.status)}, {"first_tick", r.firstTick},
            {"last_tick", r.lastTick}, {"pressed", r.pressed}, {"released", r.released},
            {"apply_latency_ms", r.applyLatencyMs < 0 ? json(nullptr) : json(r.applyLatencyMs)},
            {"client_to_consume_ms", r.clientToConsumeMs < 0 ? json(nullptr) : json(r.clientToConsumeMs)},
            {"reason", r.reason}});
    }
    json pendingEvents = json::array();
    for (const auto& event : bridge.events) {
        if (std::stoull(event.at("id").get<std::string>()) > bridge.eventAck && pendingEvents.size() < 16)
            pendingEvents.push_back(event);
    }
    const uint64_t eventFloor = bridge.events.empty() ? bridge.eventSeq + 1
        : std::stoull(bridge.events.front().at("id").get<std::string>());

    json state = {
        {"protocol", 2},
        {"kind", full ? "full" : "fast"},
        {"full_seq", bridge.fullSeq},
        {"capture_tick", bridge.captureTick},
        {"context_epoch", bridge.contextEpoch},
        {"input_tick", bridge.scheduler.inputTick},
        {"owner_epoch", bridge.scheduler.ownerEpoch},
        {"last_received_seq", bridge.scheduler.lastReceivedSeq},
        {"last_applied_command_seq", bridge.scheduler.lastAppliedSeq},
        {"input_receipts", receipts},
        {"event_floor", eventFloor},
        {"event_seq", bridge.eventSeq},
        {"bridge_build", BRIDGE_BUILD},
        {"capabilities", {"fast_state", "input_sequence", "consumed_receipts", "client_to_consume_latency",
                          "player_relative_dodge_state", "control_stick_direction", "actor_uid", "event_cursor"}},
        {"token", bridge.token},
        {"source", "soh"},
        {"instance_id", bridge.instance},
        {"seq", sampleSeq},
        {"scene_epoch", bridge.sceneEpoch},
        {"scene", -1},
        {"scene_name", ""},
        {"room", -1},
        {"entrance_index", -1},
        {"day_time", 0},
        {"is_night", false},
        {"in_game", bridge.playable},
        {"player", nullptr},
        {"inventory_named", json::array()},
        {"dialogue", json::object()},
        {"progress", json::object()},
        {"context_action", {{"code", bridge.doAction}, {"label", DoActionName(bridge.doAction)}}},
        {"context_actor", nullptr},
        {"pause_menu", {{"active", false}, {"ready", false}, {"state", 0}, {"transition_state", 0},
            {"page_index", 0}, {"cursor_special_pos", 0}, {"cursor_point", json::array()},
            {"cursor_item", json::array()}, {"cursor_slot", json::array()},
            {"named_item", nullptr}, {"prompt_choice", 0}}},
        {"game_over_state", 0},
        {"ocarina_mode", 0},
        {"ocarina_action", 0},
        {"last_played_song", 0},
        {"target_actor", nullptr},
        {"target_candidate", nullptr},
        {"camera_eye", nullptr},
        {"camera_at", nullptr},
        {"camera_input_yaw", nullptr},
        {"mirrored_world", CVarGetInteger(CVAR_ENHANCEMENT("MirroredWorld"), 0) != 0},
        {"nearby_actors", json::array()},
        {"room_actors", json::array()},
        {"room_actor_count", 0},
        {"room_actors_truncated", false},
        {"navigation_probes", json::array()},
        {"cutscene_active", false},
        {"paused", false},
        {"events", pendingEvents},
        {"last_command_seq", bridge.scheduler.lastCommandSeq},
        {"upstream_revision", REVISION},
    };

    const bool playable = bridge.playable;
    lock.unlock();

    if (playable) {
        Player* player = GET_PLAYER(gPlayState);
        if (player) {
            const int16_t scene = gPlayState->sceneNum;
            const int16_t room = gPlayState->roomCtx.curRoom.num;

            auto& pos = player->actor.world.pos;
            state["scene_epoch"] = bridge.sceneEpoch;
            state["scene"] = scene;
            state["scene_name"] = SohUtils::GetSceneName(scene);
            state["room"] = room;
            state["entrance_index"] = gSaveContext.entranceIndex;
            state["day_time"] = gSaveContext.dayTime;
            state["is_night"] = gSaveContext.nightFlag != 0;
            state["paused"] = gPlayState->pauseCtx.state != 0;
            state["pause_menu"] = {
                {"active", gPlayState->pauseCtx.state != 0},
                {"ready", gPlayState->pauseCtx.state == 6 && gPlayState->pauseCtx.unk_1E4 == 0},
                {"state", gPlayState->pauseCtx.state},
                {"transition_state", gPlayState->pauseCtx.unk_1E4},
                {"page_index", std::min<int>(gPlayState->pauseCtx.pageIndex, 4)},
                {"cursor_special_pos", gPlayState->pauseCtx.cursorSpecialPos},
                {"cursor_point", json::array()},
                {"cursor_item", json::array()},
                {"cursor_slot", json::array()},
                {"named_item", gPlayState->pauseCtx.namedItem == PAUSE_ITEM_NONE
                    ? json(nullptr) : json(gPlayState->pauseCtx.namedItem)},
                {"prompt_choice", gPlayState->pauseCtx.promptChoice},
            };
            for (auto value : gPlayState->pauseCtx.cursorPoint) state["pause_menu"]["cursor_point"].push_back(value);
            for (auto value : gPlayState->pauseCtx.cursorItem) state["pause_menu"]["cursor_item"].push_back(value);
            for (auto value : gPlayState->pauseCtx.cursorSlot) state["pause_menu"]["cursor_slot"].push_back(value);
            state["game_over_state"] = gPlayState->gameOverCtx.state;
            state["ocarina_mode"] = gPlayState->msgCtx.ocarinaMode;
            state["ocarina_action"] = gPlayState->msgCtx.ocarinaAction;
            state["last_played_song"] = gPlayState->msgCtx.lastPlayedSong;
            state["cutscene_active"] =
                (gPlayState->csCtx.state != CS_STATE_IDLE) || (Player_InCsMode(gPlayState) != 0);
            state["dialogue"] = DialogueJson(player);
            if (full) state["progress"] = ProgressJson();
            state["message_id"] =
                state["dialogue"]["active"].get<bool>() ? state["dialogue"]["text_id"] : json(nullptr);
            state["player"] = {
                {"position", {pos.x, pos.y, pos.z}},
                {"yaw", player->actor.shape.rot.y},
                {"speed_xz", player->actor.speedXZ},
                {"floor_height", player->actor.floorHeight},
                {"wall_yaw", player->actor.wallYaw},
                {"bg_check_flags", player->actor.bgCheckFlags},
                {"wall_flags", player->actor.wallPoly
                    ? SurfaceType_GetWallFlags(&gPlayState->colCtx, player->actor.wallPoly, player->actor.wallBgId) : 0},
                {"state_flags_1", player->stateFlags1},
                {"state_flags_2", player->stateFlags2},
                {"control_stick_direction", player->controlStickDirections[player->controlStickDataIndex]},
                {"hop_direction", (player->stateFlags2 & PLAYER_STATE2_HOPPING)
                    ? json(player->av1.actionVar1) : json(nullptr)},
                {"climbing_ladder", (player->stateFlags1 & PLAYER_STATE1_CLIMBING_LADDER) != 0},
                {"hanging_ledge", (player->stateFlags1 & PLAYER_STATE1_HANGING_OFF_LEDGE) != 0},
                {"climbing_ledge", (player->stateFlags1 & PLAYER_STATE1_CLIMBING_LEDGE) != 0},
                {"can_climb", (player->stateFlags2 & PLAYER_STATE2_DO_ACTION_CLIMB) != 0},
                {"can_down", (player->stateFlags2 & PLAYER_STATE2_DO_ACTION_DOWN) != 0 ||
                    (player->stateFlags1 & (PLAYER_STATE1_HANGING_OFF_LEDGE | PLAYER_STATE1_CLIMBING_LADDER)) != 0},
                {"y_dist_to_water", player->actor.yDistToWater},
                {"health", gSaveContext.health},
                {"max_health", gSaveContext.healthCapacity},
                {"rupees", gSaveContext.rupees},
                {"magic", gSaveContext.magic},
                {"age", gSaveContext.linkAge == 0 ? "adult" : "child"},
            };
            state["camera_eye"] = {
                gPlayState->view.eye.x, gPlayState->view.eye.y, gPlayState->view.eye.z,
            };
            state["camera_at"] = {
                gPlayState->view.lookAt.x, gPlayState->view.lookAt.y, gPlayState->view.lookAt.z,
            };
            state["camera_input_yaw"] = Camera_GetInputDirYaw(GET_ACTIVE_CAM(gPlayState));
            state["mirrored_world"] = CVarGetInteger(CVAR_ENHANCEMENT("MirroredWorld"), 0) != 0;
            Actor* target = gPlayState->actorCtx.targetCtx.targetedActor;
            if (target) state["target_actor"] = ActorJson(target, player, full);
            Actor* candidate = gPlayState->actorCtx.targetCtx.arrowPointedActor;
            if (candidate) state["target_candidate"] = ActorJson(candidate, player, full);
            Actor* contextActor = ContextActor(player, bridge.doAction);
            if (contextActor) state["context_actor"] = ActorJson(contextActor, player, full);
            if (full) state["nearby_actors"] = NearbyActors(player);
            const auto roomActors = RoomActors(player, full);
            state["room_actors"] = roomActors["actors"];
            state["room_actor_count"] = roomActors["count"];
            state["room_actors_truncated"] = roomActors["truncated"];
            state["navigation_probes"] = NavigationProbes(player);
            state["inventory"] = json::array();
            state["inventory_named"] = json::array();
            state["equipped"] = json::array();
            if (full) for (size_t slot = 0; slot < ARRAY_COUNT(gSaveContext.inventory.items); ++slot) {
                const auto item = gSaveContext.inventory.items[slot];
                state["inventory"].push_back(item);
                if (item != ITEM_NONE && item != ITEM_NONE_FE) {
                    state["inventory_named"].push_back({
                        {"slot", slot},
                        {"item_id", item},
                        {"name", SohUtils::GetItemName(item)},
                        {"ammo", ItemUsesAmmo(item) && slot < ARRAY_COUNT(gSaveContext.inventory.ammo)
                            ? json(std::max<int>(gSaveContext.inventory.ammo[slot], 0)) : json(nullptr)},
                    });
                }
            }
            if (full) for (auto item : gSaveContext.equips.buttonItems) state["equipped"].push_back(item);
        }
    }

    if (!full) {
        static const char* slowFields[] = {"bridge_build", "capabilities", "upstream_revision",
            "scene_name", "entrance_index", "day_time", "is_night", "inventory", "inventory_named",
            "equipped", "progress", "pause_menu", "message_id", "ocarina_action", "last_played_song",
            "nearby_actors"};
        for (const char* field : slowFields) state.erase(field);
    }
    std::string serialized = state.dump();
    if (full && serialized.size() > 59000) {
        // room_actors is the authoritative actor observation. nearby_actors is redundant,
        // so drop that compact compatibility subset first under packet pressure.
        state["nearby_actors"] = json::array();
        serialized = state.dump();
    }
    while (serialized.size() > 59000 && state["room_actors"].is_array() && !state["room_actors"].empty()) {
        state["room_actors"].erase(state["room_actors"].end() - 1);
        state["room_actors_truncated"] = true;
        serialized = state.dump();
    }
    if (serialized.size() > 59000) return;
    lock.lock();
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
        auto& bridge = Data();
        std::scoped_lock lock(bridge.mutex);
        const int16_t previous = bridge.lastScene;
        ++bridge.sceneEpoch;
        ++bridge.contextEpoch;
        bridge.scheduler.SetContext(bridge.sceneEpoch, bridge.contextEpoch);
        bridge.forceFull = true;
        bridge.actors.Clear();
        bridge.lastScene = scene;
        bridge.lastRoom = -1;
        PushEventLocked(bridge, "scene_changed",
            std::to_string(previous) + "->" + std::to_string(scene));
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnActorInit>([](void* actor) {
        Data().actors.Spawn(actor);
    });
    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnActorDestroy>([](void* actor) {
        Data().actors.Destroy(actor);
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnTransitionEnd>([](int16_t scene) {
        Event("transition_end", std::to_string(scene));
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnOpenText>([](uint16_t* textId, bool*) {
        if (textId) Event("dialogue_opened", std::to_string(*textId));
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnSetDoAction>([](uint16_t action) {
        auto& bridge = Data();
        std::scoped_lock lock(bridge.mutex);
        if (bridge.doAction == action) return;
        bridge.doAction = action;
        PushEventLocked(bridge, "context_action_changed",
            std::to_string(action) + ":" + DoActionName(action));
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnItemReceive>([](GetItemEntry itemEntry) {
        std::string detail = std::to_string(itemEntry.itemId);
        // MOD_NONE is 0 in the pinned Shipwright revision; avoid an extra randomizer-enum include here.
        if (itemEntry.modIndex == 0 && itemEntry.itemId <= ITEM_ROCS_FEATHER) {
            detail += ":" + SohUtils::GetItemName(itemEntry.itemId);
        }
        Event("item_received", detail);
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnSceneFlagSet>(
        [](int16_t scene, int16_t flagType, int16_t flag) {
            Event("scene_flag_set", std::to_string(scene) + ":" + std::to_string(flagType) + ":" + std::to_string(flag));
        });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnSceneFlagUnset>(
        [](int16_t scene, int16_t flagType, int16_t flag) {
            Event("scene_flag_unset", std::to_string(scene) + ":" + std::to_string(flagType) + ":" + std::to_string(flag));
        });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnEnemyDefeat>([](void* rawActor) {
        auto* actor = static_cast<Actor*>(rawActor);
        if (!actor) return;
        auto& bridge = Data();
        std::scoped_lock lock(bridge.mutex);
        PushEventLocked(bridge, "enemy_defeated", std::to_string(actor->id));
        bridge.events.back()["actor_uid"] = bridge.actors.Get(actor);
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnBossDefeat>([](void* rawActor) {
        auto* actor = static_cast<Actor*>(rawActor);
        if (!actor) return;
        auto& bridge = Data();
        std::scoped_lock lock(bridge.mutex);
        PushEventLocked(bridge, "boss_defeated", std::to_string(actor->id));
        bridge.events.back()["actor_uid"] = bridge.actors.Get(actor);
        if (actor->id == ACTOR_BOSS_GANON2) {
            PushEventLocked(bridge, "game_completed", "final_ganon_defeated");
        }
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnPlayerHealthChange>([](int16_t amount) {
        Event("health_changed", std::to_string(amount));
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnOcarinaSongAction>([]() {
        if (gPlayState) Event("ocarina_song_action", std::to_string(gPlayState->msgCtx.lastPlayedSong));
    });
}

static RegisterShipInitFunc registration(RegisterZeldaAiBridge);
} // namespace

extern "C" void ZeldaAiBridge_ConsumeInput(int32_t controller, void* rawInput, int32_t mode) {
    if (controller != 0 || !rawInput || mode == 0) return;
    // GameState_ReqPadData uses mode=1. Non-consuming mode=0 reads do not advance actions.
    auto* input = static_cast<Input*>(rawInput);
    auto& bridge = Data();
    std::scoped_lock lock(bridge.mutex);
    bridge.Poll();
    const auto humanInput = zelda_ai::ToN64PadState(
        static_cast<uint32_t>(input->cur.button), input->cur.stick_x, input->cur.stick_y);
    const auto nowUs = NowUs();
    const auto delivery = bridge.scheduler.Consume(nowUs / 1000, humanInput, nowUs);
    if (delivery.owned || bridge.wasOwned) {
        input->prev = bridge.lastDelivered;
        if (delivery.owned) {
            input->cur.button = delivery.pad.buttons;
            input->cur.stick_x = static_cast<int8_t>(delivery.pad.stickX);
            input->cur.stick_y = static_cast<int8_t>(delivery.pad.stickY);
            input->cur.right_stick_x = input->cur.right_stick_y = 0;
        }
        const auto previousButtons = zelda_ai::ToN64PadState(
            static_cast<uint32_t>(input->prev.button), 0, 0).buttons;
        const auto currentButtons = zelda_ai::ToN64PadState(
            static_cast<uint32_t>(input->cur.button), 0, 0).buttons;
        const uint16_t changed = static_cast<uint16_t>(previousButtons ^ currentButtons);
        input->press.button = changed & currentButtons;
        input->rel.button = changed & previousButtons;
        PadUtils_UpdateRelXY(input);
        PadUtils_UpdateRelRXY(input);
        input->press.stick_x = static_cast<int8_t>(input->cur.stick_x - input->prev.stick_x);
        input->press.stick_y = static_cast<int8_t>(input->cur.stick_y - input->prev.stick_y);
        input->press.right_stick_x = static_cast<int8_t>(input->cur.right_stick_x - input->prev.right_stick_x);
        input->press.right_stick_y = static_cast<int8_t>(input->cur.right_stick_y - input->prev.right_stick_y);
    }
    bridge.wasOwned = delivery.owned;
    bridge.lastDelivered = input->cur;
}
