import type { InputDiagnosticAction, InputDiagnosticResult, Snapshot } from './types';

const actions: Array<[InputDiagnosticAction, string]> = [
  ['tap_a', 'A'],
  ['tap_b', 'B'],
  ['target', 'Z-target'],
  ['forward', 'Frente 1 s'],
  ['back', 'Ré 1 s'],
  ['backflip', 'Backflip'],
  ['stress_a', 'A ×20'],
  ['stress_b', 'B ×20'],
];

export function RealtimePanel({ bridge, runtimeStatus, busy, result, onDiagnostic }: {
  bridge: Snapshot['bridge'] | undefined;
  runtimeStatus?: string;
  busy: boolean;
  result: InputDiagnosticResult | null;
  onDiagnostic: (action: InputDiagnosticAction) => void;
}) {
  const rt = bridge?.realtime;
  const ms = (value?: number | null) => value == null ? '—' : `${value.toFixed(1)} ms`;
  const blocked = !rt?.enabled || ['starting', 'running', 'paused'].includes(runtimeStatus ?? '') || busy;
  const last = rt?.last_receipt;
  return <section className="panel realtime-panel">
    <div className="section-head"><span>CONTROLE / LATÊNCIA</span><span>{rt?.enabled ? 'BRIDGE V2' : 'LEGADO / SEM RT'}</span></div>
    <div className="telemetry">
      <div><span>ESTADO RECEBIDO</span><strong>{rt?.state_hz == null ? '—' : `${rt.state_hz.toFixed(1)} Hz`}</strong></div>
      <div><span>INTERVALO P95</span><strong>{ms(rt?.state_interval_p95_ms)}</strong></div>
      <div><span>ACEITO → CONSUMIDO</span><strong>P50 {ms(rt?.native_apply_p50_ms)} · P95 {ms(rt?.native_apply_p95_ms)} · P99 {ms(rt?.native_apply_p99_ms)}</strong></div>
      <div><span>DONO DO INPUT</span><strong>{rt?.owner ?? '—'}</strong></div>
      <div><span>LACUNAS DE EVENTOS</span><strong>{rt?.event_gaps ?? 0}</strong></div>
      <div><span>AMOSTRAS PERDIDAS</span><strong>{rt?.dropped_samples ?? 0}</strong></div>
    </div>
    {last && <div className="diagnostic-last"><span>ÚLTIMO CONSUMO</span><code>SEQ {last.seq} · {last.status} · {ms(last.apply_latency_ms)} · PRESS 0x{last.pressed.toString(16).toUpperCase().padStart(4, '0')} · RELEASE 0x{last.released.toString(16).toUpperCase().padStart(4, '0')}</code></div>}
    <div className="diagnostic-block">
      <div className="section-head"><span>DIAGNÓSTICO LOCAL</span><span>SEM MODELO · SEM BENCHMARK</span></div>
      <div className="diagnostic-actions">{actions.map(([action, label]) =>
        <button type="button" key={action} disabled={blocked} onClick={() => onDiagnostic(action)}>{busy ? 'AGUARDE' : label}</button>)}</div>
      <p className="muted">Frente, ré e backflip só executam quando o probe local comprova piso seguro. A/B podem interagir ou atacar: use em uma área segura. Pare a IA antes de testar.</p>
      {result && <div className="diagnostic-result">
        <div><span>TESTE</span><strong>{result.action}</strong></div>
        <div><span>RESULTADO</span><strong>{result.status} / {result.reason}</strong></div>
        <div><span>COMANDOS</span><strong>{result.commands} · consumidos {result.consumed} · perdidos {result.lost}</strong></div>
        <div><span>EDGES</span><strong>press {result.presses}/{result.expected_edges} · release {result.releases}/{result.expected_edges}</strong></div>
        <div><span>LATÊNCIA</span><strong>P50 {ms(result.latency_ms.p50)} · P95 {ms(result.latency_ms.p95)} · P99 {ms(result.latency_ms.p99)}</strong></div>
        <div><span>DESLOCAMENTO</span><strong>{result.distance == null ? '—' : `${result.distance.toFixed(1)} u`}</strong></div>
      </div>}
    </div>
    <p className="muted content">{rt?.enabled
      ? 'Consumo confirma entrega ao engine, não dano, esquiva ou sucesso da ação. Não inclui latência do vídeo nem inferência.'
      : 'Reintegre e recompile o SoH para habilitar sequências nativas e confirmação de consumo. O ACK legado confirma somente recebimento.'}</p>
  </section>;
}
