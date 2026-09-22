// Integration target: HarbourMasters/Shipwright d30fc192f2eb01ceea45bd1e12de61636cafbf86.
// Compile with SoH; this is not a DLL for an unmodified release executable.
#include "ZeldaAiBridge.h"
#include "InputLease.hpp"
#include <SDL2/SDL_net.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "soh/Enhancements/game-interactor/GameInteractor.h"
#include "soh/Enhancements/item-tables/ItemTableTypes.h"
#include "soh/ShipInit.hpp"
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
constexpr size_t MAX_EVENTS = 16;
constexpr size_t MAX_ACTORS = 24;
constexpr float MAX_ACTOR_DISTANCE = 1400.0f;

int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
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
    bool playable = false;
    int16_t lastScene = -1;
    int16_t lastRoom = -1;
    uint16_t doAction = DO_ACTION_NONE;
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
            } catch (const json::exception&) {
                // Invalid datagrams cannot affect the game.
            }
        }
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

json ActorJson(Actor* actor, Player* player) {
    if (!actor || !player) return nullptr;
    std::string actorName;
    std::string actorDescription;
    if (ActorDB::Instance != nullptr) {
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
        {"name", actorName},
        {"description", actorDescription},
        {"category", actor->category},
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
    if (actor->textId != 0 || actor->category == ACTORCAT_NPC || actor->category == ACTORCAT_BOSS ||
        actor->category == ACTORCAT_DOOR || actor->category == ACTORCAT_CHEST) return 1;
    if (actor->category == ACTORCAT_ENEMY) return 2;
    if (actor->category == ACTORCAT_SWITCH || actor->category == ACTORCAT_BG ||
        actor->category == ACTORCAT_PROP || actor->category == ACTORCAT_ITEMACTION) return 3;
    return 4;
}

json DrawnActors(Player* player) {
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
            if (std::isfinite(distance) && distance <= MAX_ACTOR_DISTANCE) {
                candidates.push_back({ActorObservationPriority(actor, contextActor), distance, actor});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.priority != rhs.priority) return lhs.priority < rhs.priority;
        return lhs.distance < rhs.distance;
    });
    json result = json::array();
    for (size_t i = 0; i < candidates.size() && i < MAX_ACTORS; ++i) {
        result.push_back(ActorJson(candidates[i].actor, player));
    }
    return result;
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
    std::scoped_lock lock(bridge.mutex);
    bridge.Init();
    if (!bridge.socket || !bridge.packet || NowMs() - bridge.lastSent < 200) return;
    bridge.lastSent = NowMs();
    bridge.playable = GameInteractor::IsSaveLoaded(true) && gPlayState != nullptr;
    if (!bridge.playable) {
        bridge.lease.Release();
        bridge.lastScene = -1;
        bridge.lastRoom = -1;
    }

    json state = {
        {"protocol", 1},
        {"token", bridge.token},
        {"source", "soh"},
        {"instance_id", bridge.instance},
        {"seq", ++bridge.seq},
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
        {"nearby_actors", json::array()},
        {"cutscene_active", false},
        {"paused", false},
        {"events", bridge.events},
        {"last_command_seq", bridge.lease.lastCommandSeq},
        {"upstream_revision", REVISION},
    };

    if (bridge.playable) {
        Player* player = GET_PLAYER(gPlayState);
        if (player) {
            const int16_t scene = gPlayState->sceneNum;
            const int16_t room = gPlayState->roomCtx.curRoom.num;

            if (bridge.lastScene == -1) {
                bridge.lastScene = scene;
                bridge.lastRoom = room;
            } else if (scene != bridge.lastScene) {
                const int16_t previous = bridge.lastScene;
                ++bridge.sceneEpoch;
                bridge.lease.Release();
                bridge.lastScene = scene;
                bridge.lastRoom = room;
                PushEventLocked(bridge, "scene_changed",
                    std::to_string(previous) + "->" + std::to_string(scene));
            } else if (bridge.lastRoom == -1) {
                bridge.lastRoom = room;
            } else if (room != bridge.lastRoom) {
                const int16_t previous = bridge.lastRoom;
                ++bridge.sceneEpoch;
                bridge.lease.Release();
                bridge.lastRoom = room;
                PushEventLocked(bridge, "room_changed",
                    std::to_string(previous) + "->" + std::to_string(room));
            }

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
            state["progress"] = ProgressJson();
            state["message_id"] =
                state["dialogue"]["active"].get<bool>() ? state["dialogue"]["text_id"] : json(nullptr);
            state["player"] = {
                {"position", {pos.x, pos.y, pos.z}},
                {"yaw", player->actor.shape.rot.y},
                {"speed_xz", player->actor.speedXZ},
                {"floor_height", player->actor.floorHeight},
                {"wall_yaw", player->actor.wallYaw},
                {"bg_check_flags", player->actor.bgCheckFlags},
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
            Actor* target = gPlayState->actorCtx.targetCtx.targetedActor;
            if (!target) target = gPlayState->actorCtx.targetCtx.arrowPointedActor;
            if (target) state["target_actor"] = ActorJson(target, player);
            Actor* contextActor = ContextActor(player, bridge.doAction);
            if (contextActor) state["context_actor"] = ActorJson(contextActor, player);
            state["nearby_actors"] = DrawnActors(player);
            state["inventory"] = json::array();
            state["inventory_named"] = json::array();
            state["equipped"] = json::array();
            for (size_t slot = 0; slot < ARRAY_COUNT(gSaveContext.inventory.items); ++slot) {
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
            for (auto item : gSaveContext.equips.buttonItems) state["equipped"].push_back(item);
        }
    }

    // Events may have been appended while producing this sample.
    state["events"] = bridge.events;
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
        auto& bridge = Data();
        std::scoped_lock lock(bridge.mutex);
        const int16_t previous = bridge.lastScene;
        ++bridge.sceneEpoch;
        bridge.lease.Release();
        bridge.lastScene = scene;
        bridge.lastRoom = -1;
        PushEventLocked(bridge, "scene_changed",
            std::to_string(previous) + "->" + std::to_string(scene));
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
        if (actor) Event("enemy_defeated", std::to_string(actor->id));
    });

    GameInteractor::Instance->RegisterGameHook<GameInteractor::OnBossDefeat>([](void* rawActor) {
        auto* actor = static_cast<Actor*>(rawActor);
        if (!actor) return;
        Event("boss_defeated", std::to_string(actor->id));
        if (actor->id == ACTOR_BOSS_GANON2) {
            Event("game_completed", "final_ganon_defeated");
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

extern "C" void ZeldaAiBridge_OverrideInput(int32_t controller, uint16_t* buttons, int8_t* stickX, int8_t* stickY) {
    if (controller != 0 || !buttons || !stickX || !stickY) return;
    auto& bridge = Data();
    std::scoped_lock lock(bridge.mutex);
    bridge.Poll();
    bridge.lease.Read(NowMs(), *buttons, *stickX, *stickY);
}
