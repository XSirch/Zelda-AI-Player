"""Physical modes from the pinned SoH's current self-observation flags.

Water depth alone does not identify swimming: shallow wading can be grounded.
These modes grant no jump button, hidden landing or underwater reachability.
"""
from __future__ import annotations

GROUND = 1
JUMPING = 1 << 18
FREEFALL = 1 << 19
FIRST_PERSON = 1 << 20
IN_WATER = 1 << 27
UNDERWATER = 1 << 10
DIVING = 1 << 11


def camera_modal_active(player):
    """The pinned native first-person flag owns the analog as camera input."""
    return bool(player and player.state_flags_1 & FIRST_PERSON)


def water_active(player):
    return bool(player and (player.state_flags_1 & IN_WATER
                            or player.state_flags_2 & (UNDERWATER | DIVING)))


def grounded(player):
    """Settled dry floor contact, excluding retained ground bits in animations."""
    return bool(player and player.bg_check_flags & GROUND
                and abs(player.position[1] - player.floor_height) <= 4
                and not player.state_flags_1 & (JUMPING | FREEFALL)
                and not water_active(player)
                and not (player.climbing_ladder or player.climbing_ledge or player.hanging_ledge))


def locomotor_mode(game):
    player = game.player
    if not player:
        return "unknown"
    if player.climbing_ladder:
        return "ladder"
    if player.hanging_ledge:
        return "hanging_ledge"
    if player.climbing_ledge:
        return "climbing_ledge"
    if player.state_flags_2 & DIVING:
        return "diving"
    if player.state_flags_2 & UNDERWATER:
        contact = player.bg_check_flags & GROUND and abs(player.position[1] - player.floor_height) <= 4
        return "submerged_ground" if contact else "underwater"
    if player.state_flags_1 & IN_WATER:
        return "swimming"
    if player.state_flags_1 & (JUMPING | FREEFALL):
        return "airborne"
    if camera_modal_active(player):
        return "first_person"
    if grounded(player):
        return "ground"
    return "unknown"
