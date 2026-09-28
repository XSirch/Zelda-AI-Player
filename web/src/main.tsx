import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { api, bootstrap } from './api';
import type { Snapshot } from './types';
import './style.css';

const BUTTONS = ['A', 'B', 'Z', 'R', 'START', 'C_UP', 'C_LEFT', 'C_DOWN', 'C_RIGHT'];

function Led({ on }: { on: boolean }) {
  return <span className={on ? 'led on' : 'led'} aria-hidden="true" />;
}

function Connection({ label, on, detail }: { label: string; on: boolean; detail?: string }) {
  return <div className="connection-item">
    <div><Led on={on} /><span>{label}</span></div>
    <strong>{on ? 'CONECTADO' : 'OFFLINE'}</strong>
    {detail && <small>{detail}</small>}
  </div>;
}

function ControllerView({ snapshot }: { snapshot: Snapshot | null }) {
  const input = snapshot?.input ?? { buttons: 0, button_names: [], stick_x: 0, stick_y: 0, reason: 'idle' };
  const active = new Set(input.button_names);
  const left = 50 + (input.stick_x / 80) * 38;
  const top = 50 - (input.stick_y / 80) * 38;
  const pressed = (name: string) => active.has(name) ? 'pressed' : '';

  return <section className="controller-panel">
    <div className="section-label">CONTROLE AO VIVO</div>
    <div className="controller-grid">
      <div className="stick-block">
        <div className="stick-zone">
          <span className="axis horizontal" />
          <span className="axis vertical" />
          <span className="stick-dot" style={{ left: `${left}%`, top: `${top}%` }} />
        </div>
        <div className="stick-readout">
          <span>X <b>{input.stick_x >= 0 ? '+' : ''}{input.stick_x}</b></span>
          <span>Y <b>{input.stick_y >= 0 ? '+' : ''}{input.stick_y}</b></span>
        </div>
      </div>

      <div className="n64-buttons" aria-label="Botões físicos do controle">
        <div className="shoulders">
          <span className={`key shoulder ${pressed('Z')}`}>Z</span>
          <span className={`key shoulder ${pressed('R')}`}>R</span>
        </div>
        <div className="face">
          <span className={`key face-key b-key ${pressed('B')}`}>B</span>
          <span className={`key face-key a-key ${pressed('A')}`}>A</span>
        </div>
        <div className="start-row">
          <span className={`key start-key ${pressed('START')}`}>START</span>
        </div>
        <div className="c-cluster">
          <span className={`key c-key c-up ${pressed('C_UP')}`}>C↑</span>
          <span className={`key c-key c-left ${pressed('C_LEFT')}`}>C←</span>
          <span className={`key c-key c-right ${pressed('C_RIGHT')}`}>C→</span>
          <span className={`key c-key c-down ${pressed('C_DOWN')}`}>C↓</span>
        </div>
      </div>
    </div>
    <div className="raw-buttons">
      {BUTTONS.map(name => <span key={name} className={active.has(name) ? 'active' : ''}>{name}</span>)}
    </div>
  </section>;
}

function Thought({ snapshot }: { snapshot: Snapshot | null }) {
  const thought = snapshot?.thought;
  const intent = thought?.intent;
  const thinking = thought?.state === 'thinking';
  return <section className="thought-panel">
    <div className="section-label">
      PENSAMENTO OPERACIONAL
      <span className={thinking ? 'thinking live' : 'thinking'}>{thinking ? 'PENSANDO' : (thought?.state ?? 'IDLE').toUpperCase()}</span>
    </div>
    <div className="thought-copy">
      <p>{thought?.summary ?? 'Aguardando início.'}</p>
      {intent?.objective && <div className="objective"><span>OBJETIVO ATUAL</span><strong>{intent.objective}</strong></div>}
      {thought?.trigger && <div className="planner-line"><span>ÚLTIMA CHAMADA IA</span><strong>{thought.trigger}</strong></div>}
      {thought?.motor && <div className="motor-line"><span>MOTOR ML</span><strong>{thought.motor}</strong></div>}
    </div>
  </section>;
}


const compact = (value: number) => new Intl.NumberFormat('pt-BR', {
  notation: value >= 10_000 ? 'compact' : 'standard',
  maximumFractionDigits: 1,
}).format(value);

function durationLabel(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return '—';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours
    ? `${hours}h ${String(minutes).padStart(2, '0')}m`
    : `${minutes}m ${String(secs).padStart(2, '0')}s`;
}

function windowLabel(minutes: number | null) {
  if (minutes == null) return 'JANELA';
  if (minutes % 1440 === 0) return `${minutes / 1440}D`;
  if (minutes % 60 === 0) return `${minutes / 60}H`;
  return `${minutes}MIN`;
}

function UsageStrip({ snapshot }: { snapshot: Snapshot | null }) {
  const usage = snapshot?.usage;
  const windows = usage?.quota?.windows ?? [];
  const cost = usage?.cost_usd;
  return <section className="usage-strip">
    <div className="usage-cell">
      <span>TOKENS RUN</span>
      <strong>{compact(usage?.total_tokens ?? 0)}</strong>
      <small>{compact(usage?.calls ?? 0)} chamadas · {compact(usage?.input_tokens ?? 0)} in · {compact(usage?.output_tokens ?? 0)} out</small>
    </div>
    <div className="usage-cell">
      <span>CACHE / REASONING</span>
      <strong>{compact(usage?.cached_input_tokens ?? 0)} / {compact(usage?.reasoning_output_tokens ?? 0)}</strong>
      <small>tokens observados</small>
    </div>
    <div className="usage-cell">
      <span>CUSTO API</span>
      <strong>{cost == null ? '—' : `${cost.toFixed(4)}`}</strong>
      <small>{usage?.provider === 'codex' ? 'ChatGPT · sem custo USD reportado' : 'custo reportado pelo provider'}</small>
    </div>
    <div className="usage-cell quota-cell">
      <span>COTA RESTANTE</span>
      {windows.length ? <div className="quota-windows">{windows.slice(0, 3).map((window, index) =>
        <div className="quota-window" key={`${window.limit_id ?? 'default'}-${window.slot}-${index}`}>
          <strong>{window.remaining_percent}%</strong>
          <small>{window.limit_name || windowLabel(window.window_duration_mins)}</small>
        </div>
      )}</div> : <strong>—</strong>}
      {!windows.length && <small>{usage?.quota?.error || 'cota não anunciada pelo Codex'}</small>}
    </div>
  </section>;
}

function LearningPanel({ snapshot }: { snapshot: Snapshot | null }) {
  const learning = snapshot?.learning;
  const achievements = [...(learning?.achievements ?? [])].slice(-8).reverse();
  const lifetimeUpdates = learning?.updates ?? 0;
  const lifetimeSamples = learning?.samples_trained ?? 0;
  const updates = learning?.run_updates ?? 0;
  const samples = learning?.run_samples_trained ?? 0;
  const objectiveScore = learning?.objective_score ?? 0;
  const totalReward = learning?.total_reward ?? 0;
  const recentReward = learning?.recent_mean_reward ?? 0;
  const usefulProgressRate = learning?.useful_progress_rate ?? 0;
  const exploration = learning?.exploration;
  const rewardBreakdown = Object.entries(learning?.reward_breakdown ?? {})
    .filter(([, value]) => value !== 0)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 7);
  const evaluating = snapshot?.run_mode === 'evaluation';
  const trainingState = evaluating
    ? 'AVALIAÇÃO · PESOS CONGELADOS'
    : updates > 0
      ? 'TREINANDO'
      : snapshot?.status === 'running'
        ? 'COLETANDO EXPERIÊNCIA'
        : 'AGUARDANDO';
  const activeChampion = snapshot?.active_champion;
  const latestChampion = snapshot?.champions?.latest;
  const bestChampion = snapshot?.champions?.best_completion;

  return <section className="learning-panel">
    <div className="section-label">
      APRENDIZADO ML
      <span className={updates > 0 ? 'learning-state active' : 'learning-state'}>{trainingState}</span>
    </div>
    <div className="learning-metrics">
      <div><span>PONTOS DE CONQUISTA</span><strong>{compact(objectiveScore)}</strong><small>objetivos observados · não é reward PPO</small></div>
      <div><span>UPDATES PPO · RUN</span><strong>{compact(updates)}</strong><small>{evaluating ? 'congelado · sem updates' : `${compact(lifetimeUpdates)} no checkpoint`}</small></div>
      <div><span>AMOSTRAS · RUN</span><strong>{compact(samples)}</strong><small>{evaluating ? 'avaliação não treina' : `${compact(lifetimeSamples)} treinadas no total`}</small></div>
      <div><span>REWARD DA RUN</span><strong>{totalReward.toFixed(2)}</strong><small>última média: {recentReward.toFixed(3)}</small></div>
      <div><span>PASSOS COM PROGRESSO</span><strong>{Math.round(usefulProgressRate * 100)}%</strong><small>ignora curiosidade pura</small></div>
    </div>
    <div className="champion-strip">
      <span>CHAMPIONS <b>{snapshot?.champions?.count ?? 0}</b></span>
      <span>
        {activeChampion
          ? <>ATIVO <b>{activeChampion.id}</b> · {durationLabel(activeChampion.elapsed_s)}</>
          : latestChampion
            ? <>LATEST <b>{latestChampion.id}</b> · {durationLabel(latestChampion.elapsed_s)}</>
            : 'nenhum completion salvo'}
      </span>
      {bestChampion && <span>BEST TIME <b>{bestChampion.id}</b> · {durationLabel(bestChampion.elapsed_s)}</span>}
    </div>
    <div className="learning-body">
      <div className="achievements">
        <div className="learning-subhead">
          <span>CONQUISTAS DA RUN</span>
          <small>
            {exploration
              ? `${exploration.unique_spaces} regiões · ${exploration.unique_actors} atores · ${exploration.unique_transitions} transições`
              : 'sem exploração registrada'}
          </small>
        </div>
        {achievements.length ? achievements.map(achievement =>
          <div className="achievement-row" key={`${achievement.id}-${achievement.action_index}`}>
            <strong>+{achievement.points}</strong>
            <div>
              <b>{achievement.title}</b>
              <small>{achievement.detail}</small>
            </div>
            <span>reward PPO +{achievement.training_reward.toFixed(2)}</span>
          </div>
        ) : <div className="achievement-empty">
          Nenhuma conquista objetiva ainda. O ator pode estar coletando experiência antes do primeiro marco.
        </div>}
      </div>
      <div className="learning-note">
        <span>REWARD AGORA</span>
        <div className="reward-breakdown">
          {rewardBreakdown.length ? rewardBreakdown.map(([key, value]) =>
            <code key={key} className={value > 0 ? 'positive' : 'negative'}>
              {key} {value > 0 ? '+' : ''}{value.toFixed(3)}
            </code>
          ) : <code>sem sinal</code>}
        </div>
        <span>COMO LER</span>
        <p>
          Conquistas mostram resultados concretos no jogo. Updates e amostras confirmam que a rede foi treinada.
          Curiosidade pura não conta como progresso útil e reward crescente, sozinho, não prova melhora.
        </p>
      </div>
    </div>
  </section>;
}

function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [socketOnline, setSocketOnline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const token = useRef('');
  const reconnect = useRef<number | null>(null);

  useEffect(() => {
    let disposed = false;
    let socket: WebSocket | null = null;

    const connect = () => {
      if (disposed || !token.current) return;
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
      socket = new WebSocket(`${protocol}://${location.host}/api/events`);
      socket.onopen = () => {
        setSocketOnline(true);
        socket?.send(JSON.stringify({ token: token.current }));
      };
      socket.onmessage = event => {
        try { setSnapshot(JSON.parse(event.data) as Snapshot); } catch { /* ignore malformed local frame */ }
      };
      socket.onclose = () => {
        setSocketOnline(false);
        if (!disposed) reconnect.current = window.setTimeout(connect, 800);
      };
      socket.onerror = () => socket?.close();
    };

    void (async () => {
      try {
        token.current = await bootstrap();
        const initial = await api<Snapshot>('/status');
        if (!disposed) setSnapshot(initial);
        connect();
      } catch (err) {
        if (!disposed) setError(err instanceof Error ? err.message : String(err));
      }
    })();

    return () => {
      disposed = true;
      if (reconnect.current != null) window.clearTimeout(reconnect.current);
      socket?.close();
    };
  }, []);

  const activeRun = snapshot?.status === 'running' || snapshot?.status === 'paused' || snapshot?.status === 'starting';
  const simulator = snapshot?.connection.source === 'simulator';
  const championAvailable = !!snapshot?.champions?.latest;
  const connectionDetail = useMemo(() => {
    const hz = snapshot?.connection.state_hz;
    return hz ? `${hz.toFixed(1)} Hz` : undefined;
  }, [snapshot?.connection.state_hz]);

  async function control(path: '/start' | '/stop' | '/evaluate') {
    setBusy(true);
    setError('');
    try {
      setSnapshot(await api<Snapshot>(path, {}));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return <main className="shell">
    <header className="topbar">
      <div>
        <span className="kicker">ZELDA AI PLAYER</span>
        <h1>AUTONOMOUS ML</h1>
      </div>
      <div className="run-actions">
        {activeRun ? <button
          className="stop"
          disabled={busy}
          onClick={() => void control('/stop')}
        >
          {busy ? '…' : 'PARAR'}
        </button> : <>
          <button
            className="start"
            disabled={busy || !snapshot?.connection.game}
            onClick={() => void control('/start')}
          >
            {busy ? '…' : 'INICIAR'}
          </button>
          <button
            className="evaluate"
            disabled={busy || !snapshot?.connection.game || !championAvailable}
            onClick={() => void control('/evaluate')}
            title={championAvailable ? 'Carrega o champion mais recente com pesos congelados' : 'Nenhum champion concluído disponível'}
          >
            AVALIAR CHAMPION
          </button>
        </>}
      </div>
    </header>

    <section className="connections">
      <Connection label={simulator ? 'SIMULADOR · NÃO É GAMEPLAY REAL' : 'SHIP OF HARKINIAN'} on={!!snapshot?.connection.game} />
      <Connection label="BRIDGE REALTIME" on={!!snapshot?.connection.realtime && socketOnline} detail={connectionDetail} />
      <Connection
        label="COGNIÇÃO IA"
        on={!!snapshot?.connection.ai}
        detail={snapshot?.connection.ai_error || snapshot?.connection.ai_state}
      />
    </section>

    <UsageStrip snapshot={snapshot} />
    {simulator && <div className="notice">MODO SIMULADOR — gameplay, inputs e métricas desta sessão são sintéticos.</div>}
    {snapshot?.run_mode === 'evaluation' && snapshot?.active_champion && <div className="evaluation-notice">
      AVALIAÇÃO CONGELADA — {snapshot.active_champion.id}. PPO/RND não atualizam pesos nesta run.
    </div>}

    {error && <div className="error" role="alert">{error}</div>}
    {snapshot?.reason && snapshot.status !== 'running' && <div className="notice">{snapshot.reason}</div>}

    <div className="workspace">
      <Thought snapshot={snapshot} />
      <ControllerView snapshot={snapshot} />
    </div>

    <LearningPanel snapshot={snapshot} />

    <footer>
      <span>{snapshot?.status?.toUpperCase() ?? 'OFFLINE'}</span>
      <span>{snapshot?.elapsed_s ? `${Math.floor(snapshot.elapsed_s)}s` : '0s'} · {snapshot?.run_mode === 'evaluation' ? 'avaliação congelada' : 'política ML contínua'}</span>
    </footer>
  </main>;
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode><App /></React.StrictMode>
);
