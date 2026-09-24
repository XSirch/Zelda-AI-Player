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
