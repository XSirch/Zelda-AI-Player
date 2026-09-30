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
    assert "'/start' | '/stop' | '/evaluate'" in text


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


def test_panel_keeps_tokens_cost_and_codex_quota_visible():
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    types = Path("web/src/types.ts").read_text(encoding="utf-8")
    for label in ["TEMPO RUN", "TEMPO FINAL", "TOKENS RUN", "CACHE / REASONING", "CUSTO API", "COTA RESTANTE"]:
        assert label in main
    assert "usage?.input_tokens" in main
    assert "usage?.output_tokens" in main
    assert "snapshot?.elapsed_s" in main
    assert "remaining_percent" in main
    assert "window_duration_mins" in main
    assert "UsageSnapshot" in types
    assert "QuotaWindow" in types


def test_panel_labels_simulator_and_stops_paused_runs():
    text = Path("web/src/main.tsx").read_text(encoding="utf-8")
    assert "SIMULADOR · NÃO É GAMEPLAY REAL" in text
    assert "MODO SIMULADOR" in text
    assert "snapshot?.status === 'paused'" in text
    assert "onClick={() => void control('/stop')}" in text
    assert "onClick={() => void control('/start')}" in text


def test_panel_shows_ml_learning_and_objective_achievements():
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    types = Path("web/src/types.ts").read_text(encoding="utf-8")
    for label in [
        "APRENDIZADO ML",
        "PONTOS DE CONQUISTA",
        "UPDATES PPO · RUN",
        "AMOSTRAS · RUN",
        "REWARD DA RUN",
        "CONQUISTAS DA RUN",
        "PASSOS COM PROGRESSO",
    ]:
        assert label in main
    assert "LearningAchievement" in types
    assert "objective_score" in types
    assert "achievements" in types
    assert "useful_progress_rate" in types
    assert "reward PPO" in main
    assert "sem expansão" in main
    assert "frontier" in main
    assert "COLETAS" in main
    assert "rupees" in main
    assert "baús" in main
    assert "resources?:" in types


def test_panel_exposes_sparse_cognition_telemetry():
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    types = Path("web/src/types.ts").read_text(encoding="utf-8")
    assert "ÚLTIMA CHAMADA IA" in main
    assert "chamadas" in main
    assert "usage?.calls" in main
    assert "trigger: string | null" in types
    assert "calls: number" in types


def test_panel_exposes_champion_evaluation():
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    types = Path("web/src/types.ts").read_text(encoding="utf-8")
    for label in [
        "AVALIAR CHAMPION",
        "AVALIAÇÃO · PESOS CONGELADOS",
        "CHAMPIONS",
        "LATEST",
        "BEST TIME",
        "SALVANDO CHAMPION",
    ]:
        assert label in main
    assert "'/evaluate'" in main
    assert "run_mode" in types
    assert "ChampionSummary" in types
    assert "ChampionCatalog" in types
    assert "capture_pending" in types
    assert "training_enabled" in types


def test_panel_shows_actionable_goal_guidance():
    main = Path("web/src/main.tsx").read_text(encoding="utf-8")
    types = Path("web/src/types.ts").read_text(encoding="utf-8")
    assert "GUIDANCE DO OBJETIVO" in main
    assert "thought?.guidance?.active" in main
    assert "MotorGuidance" in types
    assert "button_quiet" in types
    assert "distance: number | null" in types
    assert "blocked?: boolean" in types
    assert "detour?: string | null" in types
    assert "BLOQUEADO" in main
    assert "desvio local" in main
    assert "anti-loop" in main
    assert "rota aprendida" in main
    assert "route_active?: boolean" in types
    assert "route_memory?:" in types
    assert "ROTAS APRENDIDAS" in main
    assert "expected_button_count?: number" in types
    assert "guidance_mix?: number" in types
    assert "exploration_decay?: number" in types
    assert "botões esperados" in main
    assert "exploração" in main
    assert "frontier_active?: boolean" in types
    assert "frontier observado" in main
    assert "exit_active?: boolean" in types
    assert "saída observada" in main
    assert "interaction_learning?:" in types
    assert "INTERAÇÃO" in main
    assert "frontier_stable?: boolean" in types
    assert "frontier_age_s?: number | null" in types
    assert "mantido" in main
    assert "frontier_completed?: number" in types
    assert "frontier_abandoned?: number" in types
    assert "route_edge_failures?: number" in types
    assert "route_edge_abandoned?: number" in types
    assert "route_edges_cooling_down?: number" in types
    assert "aresta falhou" in main
    assert "em cooldown" in main
