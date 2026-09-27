from pathlib import Path


def test_panel_is_minimal_autonomy_instrument():
    text = Path("web/src/main.tsx").read_text(encoding="utf-8")
    for label in [
        "AUTONOMOUS ML",
        "SHIP OF HARKINIAN",
        "BRIDGE REALTIME",
        "COGNIÇÃO IA",
        "PENSAMENTO OPERACIONAL",
        "CONTROLE AO VIVO",
        "INICIAR",
        "PARAR",
    ]:
        assert label in text

    # Legacy laboratory surfaces are intentionally gone from the user-facing panel.
    for legacy in [
        "BENCHMARKS",
        "EXECUÇÕES",
        "SKILLS",
        "MEMÓRIA",
        "NAVIGATION V2 / A*",
        "INTERVENÇÃO HUMANA",
        "DIAGNÓSTICO LOCAL",
    ]:
        assert legacy not in text


def test_panel_renders_raw_physical_controls_only():
    text = Path("web/src/main.tsx").read_text(encoding="utf-8")
    for button in ["A", "B", "Z", "R", "START", "C_UP", "C_LEFT", "C_DOWN", "C_RIGHT"]:
        assert button in text
    assert "stick_x" in text
    assert "stick_y" in text
    assert "button_names" in text

    # No skill selector or semantic movement macro is exposed.
    assert "navigate_to" not in text
    assert "fight_enemy" not in text
    assert "explore_area" not in text


def test_panel_uses_realtime_websocket_and_simple_start_stop():
    text = Path("web/src/main.tsx").read_text(encoding="utf-8")
    assert "new WebSocket" in text
    assert "/api/events" in text
    assert "socket.onmessage" in text
    assert "api<Snapshot>(path, {})" in text
    assert "'/start' | '/stop'" in text


def test_frontend_contract_is_compact():
    text = Path("web/src/types.ts").read_text(encoding="utf-8")
    assert "interface Snapshot" in text
    assert "interface InputState" in text
    assert "interface AgentIntent" in text
    assert "button_names: string[]" in text
    assert "stick_x: number" in text
    assert "stick_y: number" in text
    assert "GameState" not in text
    assert "Skill" not in text


def test_legacy_dashboard_components_are_removed():
    assert not Path("web/src/RealtimePanel.tsx").exists()
    assert not Path("web/src/RunBudgetFields.tsx").exists()
