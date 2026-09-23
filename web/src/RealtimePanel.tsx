import type { Snapshot } from './types';

export function RealtimePanel({ bridge }: { bridge: Snapshot['bridge'] | undefined }) {
  const rt = bridge?.realtime;
  const ms = (value?: number | null) => value == null ? '—' : `${value.toFixed(1)} ms`;
  return <section className="panel">
    <div className="section-head"><span>CONTROLE / LATÊNCIA</span><span>{rt?.enabled ? 'BRIDGE V2' : 'LEGADO / SEM RT'}</span></div>
    <div className="telemetry">
      <div><span>ESTADO RECEBIDO</span><strong>{rt?.state_hz == null ? '—' : `${rt.state_hz.toFixed(1)} Hz`}</strong></div>
      <div><span>INTERVALO P95</span><strong>{ms(rt?.state_interval_p95_ms)}</strong></div>
      <div><span>ACEITO → CONSUMIDO P95</span><strong>{ms(rt?.native_apply_p95_ms)}</strong></div>
      <div><span>DONO DO INPUT</span><strong>{rt?.owner ?? '—'}</strong></div>
      <div><span>LACUNAS DE EVENTOS</span><strong>{rt?.event_gaps ?? 0}</strong></div>
      <div><span>AMOSTRAS PERDIDAS</span><strong>{rt?.dropped_samples ?? 0}</strong></div>
    </div>
    <p className="muted content">{rt?.enabled
      ? 'Consumo confirma entrega ao engine, não dano, esquiva ou sucesso da ação. Não inclui latência do vídeo nem inferência.'
      : 'Reintegre e recompile o SoH para habilitar sequências nativas e confirmação de consumo. O ACK legado confirma somente recebimento.'}</p>
  </section>;
}
