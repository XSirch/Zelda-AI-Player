from pathlib import Path


def test_budget_ui_matches_backend_ranges():
    text = Path('web/src/RunBudgetFields.tsx').read_text(encoding='utf-8')
    for field in ['max_calls', 'max_tokens', 'max_runtime_s', 'max_cost_usd']:
        row = next(line for line in text.splitlines() if f"id: '{field}'" in line)
        assert 'min: 0' in row
    assert "id: 'max_output_tokens'" in text and 'min: 256' in text
    assert "text !== ''" in text
    assert 'Sem limite' in text


def test_stop_control_does_not_share_provider_busy():
    text = Path('web/src/main.tsx').read_text(encoding='utf-8')
    stop = next(line for line in text.splitlines() if '>Encerrar</button>' in line)
    assert 'disabled' not in stop
    assert 'action(' not in stop
    assert 'control_generation' in text


def test_realtime_panel_has_local_diagnostics_without_model_controls():
    text = Path('web/src/RealtimePanel.tsx').read_text(encoding='utf-8')
    assert 'DIAGNÓSTICO LOCAL' in text
    assert 'SEM MODELO · SEM BENCHMARK' in text
    for label in ['Frente 1 s', 'Ré 1 s', 'Backflip', 'A ×20', 'B ×20']:
        assert label in text


def test_diagnostic_panel_has_emergency_handoff():
    text = Path('web/src/RealtimePanel.tsx').read_text(encoding='utf-8')
    assert 'LIBERAR CONTROLE' in text
    main = Path('web/src/main.tsx').read_text(encoding='utf-8')
    assert '/diagnostics/release' in main
    assert 'diagnostic_active' in main


def test_live_dashboard_uses_tabs_instead_of_stacked_panels():
    text = Path('web/src/main.tsx').read_text(encoding='utf-8')
    for label in ['CONTROLE', 'COMBATE', 'TERRENO', 'ATORES', 'PROGRESSO', 'ESTADO', 'DECISÃO']:
        assert label in text
    assert 'className="live-tabs"' in text
    assert 'CombatLearningPanel' in text
    # Events and human intervention moved into the decision tab; the old always-visible bottom grid is gone.
    live_start = text.index("{tab === 'AO VIVO'")
    live_end = text.index("{(tab === 'BENCHMARKS'", live_start)
    assert 'bottom-grid' not in text[live_start:live_end]


def test_navigation_v2_dashboard_exposes_build_mesh_and_astar():
    text = Path('web/src/main.tsx').read_text(encoding='utf-8')
    for label in ['NAVIGATION V2 / A*', 'BRIDGE BUILD', 'NAVMESH', 'CELLS', 'RAIO LOCAL',
                  'PROBE YAW', 'WAYPOINT', 'CUSTO A*', 'NAVIGATION V2 NÃO CONFIRMADO']:
        assert label in text
    assert "game.bridge_build.startsWith('rt-input-v2.')" in text
    assert "capabilities.includes('local_navmesh')" in text
    assert "capabilities.includes('probe_yaw_v2')" in text
    assert "Conectado · ${game?.bridge_build" in text


def test_dashboard_types_include_navigation_debug_contract():
    text = Path('web/src/types.ts').read_text(encoding='utf-8')
    assert 'NavigationMeshSnapshot' in text
    assert 'NavigationDebugTelemetry' in text
    assert 'bridge_build: string' in text
    assert 'capabilities: string[]' in text
    assert 'navigation?: NavigationDebugTelemetry | null' in text
    assert 'SceneExitObservation' in text
    assert 'scene_exits: SceneExitObservation[]' in text



def test_terrain_panel_exposes_scene_exit_surfaces():
    text = Path('web/src/main.tsx').read_text(encoding='utf-8')
    assert 'SAÍDAS DE CENA OBSERVADAS' in text
    assert 'SCENE EXITS' in text
    assert 'use traverse_exit' in text
    assert 'exit.exit_index' in text
    assert 'exit.entrance_index' in text
    assert "capabilities.includes('scene_exit_surfaces')" in text
    assert 'SCENE EXIT SURFACES INDISPONÍVEIS' in text
    assert 'rt-input-v2.7' in text
    assert "(value & 0xFFFF).toString(16)" in text
    assert "hex16(exit.entrance_index)" in text

