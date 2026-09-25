import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { api, bootstrap, saveJson } from './api';
import type { CombatProfile, GameState, InputDiagnosticAction, InputDiagnosticResult, Metrics, ModelInfo, ProviderInfo, RunConfig, RunRow, Skill, Snapshot } from './types';
import './style.css';
import { BudgetSummary, RunBudgetFields } from './RunBudgetFields';
import { RealtimePanel } from './RealtimePanel';

const tabs = ['AO VIVO', 'BENCHMARKS', 'EXECUÇÕES', 'SKILLS', 'MEMÓRIA', 'CONEXÕES'];
const liveTabs = ['CONTROLE', 'COMBATE', 'TERRENO', 'ATORES', 'PROGRESSO', 'ESTADO', 'DECISÃO'] as const;
type LiveTab = typeof liveTabs[number];
const defaults: RunConfig = { provider: 'codex', model: 'gpt-6-astra', effort: null, goal: 'Complete Ocarina of Time autonomously and defeat final Ganon.', memory_mode: 'adaptive', max_calls: 5000, max_tokens: 5000000, max_cost_usd: 2, max_output_tokens: 2048, max_runtime_s: 43200, checkpoint_label: 'save-manual' };
const number = (n?: number | null) => n == null ? '—' : new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 1 }).format(n);
const dollars = (n?: number | null) => n == null ? 'Não faturado por API / desconhecido' : `$${n.toFixed(4)}`;
const clock = (n: number) => `${Math.floor(n / 3600).toString().padStart(2, '0')}:${Math.floor(n / 60 % 60).toString().padStart(2, '0')}:${Math.floor(n % 60).toString().padStart(2, '0')}`;

function Monitor({ game }: { game: GameState | null }) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState('');
  function stop() { stream.current?.getTracks().forEach(t => t.stop()); stream.current = null; if (video.current) video.current.srcObject = null; setActive(false); }
  useEffect(() => () => stream.current?.getTracks().forEach(t => t.stop()), []);
  async function capture() {
    try {
      setError('');
      const media = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 }, audio: false });
      stream.current = media;
      if (video.current) { video.current.srcObject = media; await video.current.play(); }
      media.getVideoTracks()[0].onended = stop;
      setActive(true);
    } catch (err) { stop(); setError(err instanceof Error ? err.message : String(err)); }
  }
  return <section className="monitor panel"><div className="section-head"><span>01 / MONITOR DO JOGO</span><span className="muted">VÍDEO LOCAL · NÃO ENVIADO À IA</span></div>
    <div className="screen"><video ref={video} muted playsInline className={active ? '' : 'hidden'} />{!active && <div className="screen-empty"><div className="reticle">+</div><h2>{game?.source === 'simulator' ? 'Simulador de protocolo' : 'Seu jogo, nesta tela.'}</h2><p>{game?.source === 'simulator' ? 'Não há Zelda rodando no simulador. Ele testa controles, métricas e comunicação.' : 'Abra o Ship of Harkinian e selecione sua janela. A captura permanece neste navegador.'}</p><button onClick={capture}>Selecionar janela</button></div>}</div>
    <div className="monitor-foot"><span>{game ? `${game.scene_name || `SCENE ${game.scene}`} / ROOM ${game.room} / SAMPLE ${game.seq}` : 'AGUARDANDO TELEMETRIA'}</span>{active && <button className="small" onClick={stop}>Desconectar vídeo</button>}</div>{error && <p className="error">{error}</p>}</section>;
}

function DialoguePanel({ game }: { game: GameState | null }) {
  const dialogue = game?.dialogue;
  if (!dialogue?.active) return null;
  return <section className="panel dialogue-panel">
    <div className="section-head"><span>DIÁLOGO ATIVO</span><span className="muted">TEXT {dialogue.text_id ?? '—'} · {dialogue.state}</span></div>
    <div className="dialogue-body">
      <p className="dialogue-text">{dialogue.text || 'Texto ainda sendo decodificado…'}</p>
      {dialogue.speaker && <p className="muted">SPEAKER {dialogue.speaker.description || dialogue.speaker.name || `ACTOR ${dialogue.speaker.actor_id}`} · ID {dialogue.speaker.actor_id} · CAT {dialogue.speaker.category} · {number(dialogue.speaker.distance)} u</p>}
      {dialogue.choices.length > 0 && <ol className="choice-list">{dialogue.choices.map((choice, index) =>
        <li key={`${dialogue.text_id}-${index}`} className={index === dialogue.choice_index ? 'selected-choice' : ''}>{choice || `Opção ${index + 1}`}</li>)}</ol>}
    </div>
  </section>;
}

function TerrainPanel({ game, bridge }: { game: GameState | null; bridge: Snapshot['bridge'] | undefined }) {
  if (!game) return null;
  const probes = (game.navigation_probes ?? []).filter(p => p.floor_found && p.delta_y != null);
  const traversal = game.player?.climbing_ladder ? 'LADDER' : game.player?.hanging_ledge ? 'HANGING' :
    game.player?.climbing_ledge ? 'LEDGE CLIMB' : game.player?.can_climb ? 'CLIMB AVAILABLE' :
    game.player?.can_down ? 'DOWN AVAILABLE' : 'NORMAL';
  const capabilities = game.capabilities ?? [];
  const mesh = game.navmesh;
  const affordances = game.traversal_affordances ?? [];
  const sceneExits = game.scene_exits ?? [];
  const nav = bridge?.navigation;
  const navCapable = capabilities.includes('local_navmesh');
  const probeYawV2 = capabilities.includes('probe_yaw_v2');
  const exitSurfaces = capabilities.includes('scene_exit_surfaces');
  const traversalScan = capabilities.includes('traversal_affordances_v1');
  const meshActive = navCapable && !!mesh?.step && (mesh?.cells?.length ?? 0) > 0;
  const expectedBridge = game.source === 'simulator' || (
    game.bridge_build.startsWith('rt-input-v2.') && navCapable && probeYawV2
  );
  const vector = (value?: number[] | null) => value?.length === 3 ? value.map(v => number(v)).join(' / ') : '—';
  const hex16 = (value: number) => (value & 0xFFFF).toString(16).toUpperCase().padStart(4, '0');
  const navState = !navCapable ? 'INDISPONÍVEL' : meshActive ? 'ATIVO' :
    (game.paused || game.cutscene_active || game.dialogue?.active || game.game_over_state ? 'SUSPENSO PELO JOGO' : 'SEM MALHA');
  return <div className="terrain-stack">
    {!expectedBridge && bridge?.connected && <div className="notice">
      NAVIGATION V2 NÃO CONFIRMADO — bridge recebida: <code>{game.bridge_build || 'desconhecida'}</code>.
      {!navCapable ? ' capability local_navmesh ausente.' : ''}{!probeYawV2 ? ' capability probe_yaw_v2 ausente.' : ''}
      {' '}Se o código já foi atualizado, reinstale a bridge, recompile o Shipwright e abra o novo soh.exe.
    </div>}
    {navCapable && !exitSurfaces && bridge?.connected && <div className="notice">
      SCENE EXIT SURFACES INDISPONÍVEIS — para detectar warps/thresholds sem ator de porta,
      use <code>rt-input-v2.7</code> ou superior e recompile o SoH.
    </div>}
    {navCapable && !traversalScan && bridge?.connected && <div className="notice">
      TRAVERSAL SCAN INDISPONÍVEL — para descobrir escadas, descidas, ladders e paredes escaláveis
      fora das sondas imediatas, use <code>rt-input-v2.9</code> ou superior e recompile o SoH.
    </div>
    <section className="panel">
      <div className="section-head"><span>NAVIGATION V2 / A*</span><span className="muted">{navState}</span></div>
      <div className="telemetry">
        <div><span>BRIDGE BUILD</span><strong>{game.bridge_build || '—'}</strong></div>
        <div><span>PROTOCOLO</span><strong>v{game.protocol ?? '—'}</strong></div>
        <div><span>NAVMESH</span><strong>{navState}</strong></div>
        <div><span>CELLS</span><strong>{meshActive ? `${mesh.cells.length} / 81` : '—'}</strong></div>
        <div><span>STEP</span><strong>{meshActive ? `${number(mesh.step)} u` : '—'}</strong></div>
        <div><span>RAIO LOCAL</span><strong>{meshActive ? `${number(mesh.step * mesh.half_extent)} u` : '—'}</strong></div>
        <div><span>PROBE YAW</span><strong>{probeYawV2 ? 'v2' : 'legacy / ausente'}</strong></div>
        <div><span>EXIT SURFACES</span><strong>{exitSurfaces ? 'v2.7+' : 'indisponível'}</strong></div>
        <div><span>TRAVERSAL SCAN</span><strong>{traversalScan ? `${affordances.length} candidato(s)` : 'indisponível'}</strong></div>
        <div><span>SCENE EXITS</span><strong>{sceneExits.length ? `${sceneExits.length} observada(s)` : 'nenhuma'}</strong></div>
        <div><span>A*</span><strong>{nav?.status ? nav.status.toUpperCase().replaceAll('_', ' ') : (navCapable ? 'IDLE' : 'OFF')}</strong></div>
        <div><span>SKILL</span><strong>{nav?.skill ?? '—'}</strong></div>
        <div><span>PATH</span><strong>{nav?.path_cells ? `${nav.path_cells} cells` : '—'}</strong></div>
        <div><span>TARGET</span><strong>{vector(nav?.target_position)}</strong></div>
        <div><span>WAYPOINT</span><strong>{vector(nav?.waypoint)}</strong></div>
        <div><span>PROBE</span><strong>{nav?.probe_safe == null ? '—' : nav.probe_safe ? 'SAFE' : 'BLOCKED'}</strong></div>
        <div><span>CUSTO A*</span><strong>{nav?.plan_cost == null ? '—' : number(nav.plan_cost)}</strong></div>
        <div><span>EXIT ATUAL</span><strong>{nav?.exit_index == null ? '—' : `EXIT ${nav.exit_index} · ENTRANCE 0x${hex16(nav.entrance_index ?? -1)}`}</strong></div>
      </div>
    </section>
    {affordances.length > 0 && <section className="panel actors-panel">
      <div className="section-head"><span>ROTAS VERTICAIS OBSERVADAS</span><span className="muted">TRAVERSAL AFFORDANCES</span></div>
      <div className="actors-grid">{affordances.map((item, index) =>
        <div className="actor-item" key={`${item.kind}-${index}`}>
          <span>{item.kind.replaceAll('_', ' ').toUpperCase()} · {item.direction.toUpperCase()}</span>
          <strong>APPROACH {vector(item.approach_position)} · TARGET {vector(item.target_position)} · ΔY {number(item.height_delta)} · {number(item.distance)}u · use traverse_to</strong>
        </div>)}
      </div>
    </section>}
    {sceneExits.length > 0 && <section className="panel actors-panel">
      <div className="section-head"><span>SAÍDAS DE CENA OBSERVADAS</span><span className="muted">COLLISION SURFACE</span></div>
      <div className="actors-grid">{sceneExits.map(exit =>
        <div className="actor-item" key={exit.exit_index}>
          <span>EXIT {exit.exit_index} · ENTRANCE 0x{hex16(exit.entrance_index)}</span>
          <strong>POS {vector(exit.position)} · {exit.samples} amostras · DIRECT {exit.direct_reachable ? 'SAFE' : 'WAIT'} · use traverse_exit</strong>
        </div>)}
      </div>
    </section>}
    <section className="panel actors-panel">
      <div className="section-head"><span>TERRENO / TRAVESSIA</span><span className="muted">{traversal}</span></div>
      <div className="actors-grid">{probes.map((probe, index) =>
        <div className="actor-item" key={`${probe.direction}-${probe.distance}-${index}`}>
          <span>{probe.direction.replaceAll('_', ' ')} · {number(probe.distance)} u</span>
          <strong>ΔY {number(probe.delta_y)} · FLOOR {probe.floor_type ?? '—'}{probe.wall_hit ? ` · WALL ${number(probe.wall_distance)}u · FLAGS 0x${probe.wall_flags.toString(16).toUpperCase()}` : ''}</strong>
        </div>)}
        {!probes.length && <p className="empty">Nenhum probe de piso utilizável neste snapshot.</p>}
      </div>
    </section>
  </div>;
}

function ActorsPanel({ game }: { game: GameState | null }) {
  const actors = game?.room_actors?.length ? game.room_actors : game?.nearby_actors ?? [];
  if (!game || !actors.length) return null;
  return <section className="panel actors-panel">
    <div className="section-head"><span>ATORES DA SALA</span><span className="muted">{game.room_actor_count || actors.length} ATIVOS{game.room_actors_truncated ? ' · LISTA LIMITADA A 64' : ''}</span></div>
    <div className="actors-grid">{actors.map((actor, index) =>
      <div className="actor-item" key={`${actor.actor_id}-${actor.params}-${actor.room}-${index}`}>
        <span>{actor.description || actor.name || `Actor ${actor.actor_id}`}</span>
        <strong>ID {actor.actor_id} · {actor.category_name || `CAT ${actor.category}`} · ROOM {actor.room} · PARAM {actor.params} · {number(actor.distance)} u{actor.drawn ? ' · DRAWN' : ' · OFF-CAMERA'}{actor.targeted ? ' · TARGET' : ''}</strong>
      </div>)}
    </div>
  </section>;
}

function ProgressPanel({ game }: { game: GameState | null }) {
  const p = game?.progress;
  if (!p) return null;
  const upgrades = Object.entries(p.upgrade_levels ?? {}).filter(([, level]) => level > 0);
  return <section className="panel progress-panel">
    <div className="section-head"><span>PROGRESSO OBSERVÁVEL</span><span className="muted">PAUSE / HUD</span></div>
    <div className="progress-grid">
      <div><span>QUEST / SONGS</span><strong>{p.quest_items.length ? p.quest_items.join(' · ') : 'Nenhum registrado'}</strong></div>
      <div><span>EQUIPAMENTO</span><strong>{p.equipment?.length ? p.equipment.map(item => `${item.equipped ? '● ' : ''}${item.name}`).join(' · ') : (p.owned_equipment.length ? p.owned_equipment.join(' · ') : 'Nenhum registrado')}</strong></div>
      <div><span>DUNGEON ATUAL</span><strong>{p.dungeon_items.length ? p.dungeon_items.join(' · ') : 'Sem mapa/compass/boss key'} · {p.small_keys} chaves</strong></div>
      <div><span>UPGRADES</span><strong>{upgrades.length ? upgrades.map(([name, level]) => `${name} ${level}`).join(' · ') : 'Nenhum'}</strong></div>
      <div><span>OUTROS</span><strong>{p.heart_pieces}/4 heart pieces · {p.skull_tokens} skulltulas · magia {p.magic_acquired ? (p.double_magic ? 'dupla' : 'sim') : 'não'} · defesa dupla {p.double_defense ? 'sim' : 'não'}</strong></div>
    </div>
  </section>;
}

function InventoryPanel({ game }: { game: GameState | null }) {
  if (!game?.inventory_named?.length) return null;
  return <section className="panel inventory-panel">
    <div className="section-head"><span>INVENTÁRIO OBSERVADO</span><span className="muted">{game.inventory_named.length} slots ocupados</span></div>
    <div className="inventory-grid">{game.inventory_named.map(item =>
      <div className="inventory-item" key={item.slot}><span>SLOT {item.slot} · ID {item.item_id}{item.ammo != null && item.ammo >= 0 ? ` · AMMO ${item.ammo}` : ''}</span><strong>{item.name}</strong></div>)}
    </div>
  </section>;
}

function CombatLearningPanel({ profiles, game }: { profiles: CombatProfile[]; game: GameState | null }) {
  const enemies = (game?.room_actors ?? []).filter(actor => actor.category === 5 || actor.category === 9);
  const visibleIds = new Set(enemies.map(actor => actor.actor_id));
  const unknown = enemies.filter(actor => !profiles.some(profile => profile.actor_id === actor.actor_id));
  return <section className="panel combat-learning-panel">
    <div className="section-head"><span>APRENDIZADO POR INIMIGO</span><span className="muted">{profiles.length} PERFIS · MODELO + EFFORT ATUAL</span></div>
    {unknown.length > 0 && <div className="combat-unknown">{unknown.map(actor =>
      <div key={actor.actor_uid || String(actor.actor_id) + '-' + String(actor.params)}>
        <span>SEM EXPERIÊNCIA</span>
        <strong>{actor.description || actor.name || 'Actor ' + actor.actor_id}</strong>
        <small>ID {actor.actor_id} · o primeiro episódio ainda não gerou política persistida</small>
      </div>)}</div>}
    <div className="combat-profile-grid">{profiles.map(profile => {
      const best = Object.entries(profile.best_by_state ?? {}).slice(0, 8);
      return <article key={profile.id} className={visibleIds.has(profile.actor_id) ? 'combat-profile visible' : 'combat-profile'}>
        <div className="combat-profile-head"><div><span>{visibleIds.has(profile.actor_id) ? 'PRESENTE NA SALA' : 'MEMÓRIA'}</span><h3>{profile.enemy_name || 'Actor ' + profile.actor_id}</h3></div><strong>{profile.wins}V / {profile.losses}D</strong></div>
        <div className="combat-stats"><span>EPISÓDIOS <b>{profile.encounters}</b></span><span>INCOMPLETOS <b>{profile.incomplete}</b></span><span>DANO RECEBIDO <b>{number(profile.damage_taken / 16)} ♥</b></span></div>
        {best.length ? <div className="learned-actions">{best.map(([stateName, entry]) =>
          <div key={stateName}><code>{stateName}</code><span>{entry.action}</span><small>reward {entry.mean.toFixed(2)} · n={entry.samples}</small></div>)}</div>
          : <p className="muted">Sem ações consolidadas. O próximo encontro explora apenas opções seguras e atualiza este perfil.</p>}
      </article>;
    })}</div>
    {!profiles.length && !unknown.length && <p className="empty">Nenhum inimigo observado e nenhum perfil aprendido neste namespace.</p>}
  </section>;
}

function MetricStrip({ metrics }: { metrics: Metrics | null }) {
  const items = [['CHAMADAS', number(metrics?.calls)], ['TOKENS TOTAIS', number(metrics?.total_tokens)], ['CUSTO REGISTRADO', dollars(metrics?.cost_usd)], ['MORTES', number(metrics?.deaths)], ['LATÊNCIA MÉDIA', metrics?.mean_latency_ms == null ? '—' : `${number(metrics.mean_latency_ms / 1000)} s`]];
  return <div className="metrics">{items.map(([label, value]) => <div className="metric" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>;
}

function App() {
  const [tab, setTab] = useState('AO VIVO');
  const [liveTab, setLiveTab] = useState<LiveTab>('CONTROLE');
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [controlBusy, setControlBusy] = useState(false);
  const [diagnosticBusy, setDiagnosticBusy] = useState(false);
  const [diagnosticResult, setDiagnosticResult] = useState<InputDiagnosticResult | null>(null);
  function acceptSnapshot(next: Snapshot) {
    setSnap(old => old && (old.control_generation ?? 0) > (next.control_generation ?? 0) ? old : next);
  }
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [config, setConfig] = useState<RunConfig>(defaults);
  const [rows, setRows] = useState<RunRow[]>([]);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [hint, setHint] = useState('');
  const [key, setKey] = useState('');
  const [login, setLogin] = useState<{authUrl?: string; verificationUrl?: string; userCode?: string} | null>(null);
  const [detail, setDetail] = useState<unknown>(null);
  const selected = models.find(m => m.id === config.model);
  const active = snap?.status === 'running' || snap?.status === 'paused';
  const game = snap?.bridge.state ?? null;
  const player = game?.player;
  const selectedProvider = providers.find(p => p.id === config.provider);

  async function action(fn: () => Promise<unknown>) { setBusy(true); setError(''); try { await fn(); } catch (err) { setError(err instanceof Error ? err.message : String(err)); } finally { setBusy(false); } }
  async function refreshProviders() { setProviders(await api<ProviderInfo[]>('/providers')); }
  useEffect(() => {
    let alive = true; let socket: WebSocket | undefined; let retry: ReturnType<typeof setTimeout> | undefined;
    async function connect() {
      try {
        const token = await bootstrap(); if (!alive) return;
        setReady(true);
        socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/events`);
        socket.onopen = () => socket?.send(JSON.stringify({token}));
        socket.onmessage = event => { if (alive) { setConnected(true); acceptSnapshot(JSON.parse(event.data) as Snapshot); } };
        socket.onclose = () => { if (alive) { setConnected(false); retry = setTimeout(connect, 2000); } };
        socket.onerror = () => socket?.close();
      } catch (err) { if (alive) { setError(String(err)); retry = setTimeout(connect, 3000); } }
    }
    void connect();
    return () => { alive = false; clearTimeout(retry); socket?.close(); };
  }, []);
  useEffect(() => { if (ready) void action(refreshProviders); }, [ready]);
  useEffect(() => {
    if (!ready) return;
    let current = true; setModels([]);
    api<ModelInfo[]>(`/models/${config.provider}`).then(catalog => { if (!current) return; setModels(catalog); setConfig(c => { const astra = config.provider === 'codex' ? catalog.find(m => m.id === 'gpt-6-astra') ?? catalog.find(m => m.id.toLowerCase().includes('astra') || m.name.toLowerCase().includes('astra')) : undefined; const model = catalog.find(m => m.id === c.model) ?? astra ?? catalog.find(m => m.structured_output); return {...c, model: model?.id ?? '', effort: model?.default_effort && model.efforts.includes(model.default_effort) ? model.default_effort : null}; }); }).catch(err => { if (current) setError(String(err)); });
    return () => { current = false; };
  }, [config.provider, ready, providers]);
  useEffect(() => { if (!ready) return; if (tab === 'BENCHMARKS' || tab === 'EXECUÇÕES') void action(async () => setRows(await api<RunRow[]>('/runs'))); if (tab === 'SKILLS') void action(async () => setSkills(await api<Skill[]>('/skills'))); }, [tab, ready]);
  function update<K extends keyof RunConfig>(field: K, value: RunConfig[K]) { setConfig(c => ({...c, [field]: value})); }
  async function control(command: string) {
    setControlBusy(true); setError('');
    try { acceptSnapshot(await api<Snapshot>(`/control/${command}`, {})); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setControlBusy(false); }
  }
  async function start() { acceptSnapshot(await api<Snapshot>('/runs', config)); }
  async function diagnosticInput(action: InputDiagnosticAction) {
    setDiagnosticBusy(true); setError('');
    try {
      const response = await api<{result: InputDiagnosticResult; snapshot: Snapshot}>('/diagnostics/input', {action});
      setDiagnosticResult(response.result);
      acceptSnapshot(response.snapshot);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setDiagnosticBusy(false); }
  }
  async function releaseDiagnostic() {
    setError('');
    try {
      const response = await api<{result: {released: boolean}; snapshot: Snapshot}>('/diagnostics/release', {});
      acceptSnapshot(response.snapshot);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  }

  return <div className="app"><aside><div className="brand"><span className="brand-mark">Z / AI</span><b>ZELDA<br/>AI PLAYER</b><small>AGENT LAB · 0.1</small></div><nav aria-label="Navegação principal">{tabs.map((name, index) => <button key={name} className={tab === name ? 'selected' : ''} onClick={() => { setTab(name); setDetail(null); }}><span>0{index + 1}</span>{name}</button>)}</nav><div className="sidebar-bottom"><span className={connected ? 'led on' : 'led'} />{connected ? 'RUNTIME CONECTADO' : 'RUNTIME OFFLINE'}<p>Estado estruturado.<br/>Nenhum screenshot no prompt.</p></div></aside>
    <main><header><div><p className="eyebrow">OCARINA OF TIME / SHIP OF HARKINIAN</p><h1>{tab === 'AO VIVO' ? 'Sala de controle' : tab[0] + tab.slice(1).toLowerCase()}</h1></div><div className="run-status"><span className={snap?.status === 'running' ? 'led on' : 'led'} />{snap?.status ?? 'OFFLINE'}<strong>{clock(snap?.elapsed_s ?? 0)}</strong></div></header>
      {game?.source === 'simulator' && <div className="notice">MODO SIMULADO — não é gameplay real e não pertence ao benchmark de Zelda.</div>}
      {error && <div role="alert" className="error-banner"><span>{error}</span><button aria-label="Fechar erro" onClick={() => setError('')}>×</button></div>}
      {snap?.reason && <div className="notice">Execução: {snap.reason}</div>}
      {tab === 'AO VIVO' && <><MetricStrip metrics={snap?.metrics ?? null}/><div className="cockpit"><div className="live-column">
        <Monitor game={game}/>
        <div className="live-tabs" role="tablist" aria-label="Painéis do jogo">{liveTabs.map(name =>
          <button type="button" role="tab" aria-selected={liveTab === name} className={liveTab === name ? 'active' : ''} key={name} onClick={() => setLiveTab(name)}>{name}</button>)}</div>
        <div className="live-tab-stage">
          {liveTab === 'CONTROLE' && <><RealtimePanel bridge={snap?.bridge} runtimeStatus={snap?.status} busy={diagnosticBusy || !!snap?.diagnostic_active} result={diagnosticResult ?? snap?.diagnostic ?? null} onDiagnostic={action => void diagnosticInput(action)} onRelease={() => void releaseDiagnostic()}/><DialoguePanel game={game}/></>}
          {liveTab === 'COMBATE' && <CombatLearningPanel profiles={snap?.combat_profiles ?? []} game={game}/>}
          {liveTab === 'TERRENO' && <TerrainPanel game={game} bridge={snap?.bridge}/>}
          {liveTab === 'ATORES' && <ActorsPanel game={game}/>}
          {liveTab === 'PROGRESSO' && <><ProgressPanel game={game}/><InventoryPanel game={game}/></>}
          {liveTab === 'ESTADO' && <div className="telemetry panel">
            <div><span>VIDA</span><strong>{player ? number(player.health / 16) + ' / ' + number(player.max_health / 16) + ' corações' : '—'}</strong></div>
            <div><span>RUPIAS</span><strong>{number(player?.rupees)}</strong></div>
            <div><span>POSIÇÃO NATIVA</span><strong>{player?.position.map(v => number(v)).join(' / ') ?? '—'}</strong></div>
            <div><span>BRIDGE</span><strong>{snap?.bridge.connected ? `Conectado · ${game?.bridge_build ?? 'build desconhecida'} · P${game?.protocol ?? '—'}` : 'Desconectado'}</strong></div>
            <div><span>AÇÃO CONTEXTUAL</span><strong>{game?.context_action?.label ?? '—'}</strong></div>
            <div><span>ATOR CONTEXTUAL</span><strong>{game?.context_actor ? (game.context_actor.description || game.context_actor.name || 'ID ' + game.context_actor.actor_id) : '—'}</strong></div>
            <div><span>ENTRADA</span><strong>{game?.entrance_index ?? '—'}</strong></div>
            <div><span>HORÁRIO</span><strong>{game ? (game.is_night ? 'Noite' : 'Dia') + ' · 0x' + game.day_time.toString(16).padStart(4, '0').toUpperCase() : '—'}</strong></div>
            <div><span>ATORES DESENHADOS</span><strong>{game?.nearby_actors?.length ?? 0}</strong></div>
            <div><span>CUTSCENE</span><strong>{game?.cutscene_active ? 'Ativa' : 'Não'}</strong></div>
            <div><span>PAUSE</span><strong>{game?.pause_menu?.active ? 'Página ' + game.pause_menu.page_index + ' · ' + (game.pause_menu.ready ? 'pronto' : 'transição ' + game.pause_menu.transition_state) + ' · cursor ' + (game.pause_menu.cursor_slot?.[game.pause_menu.page_index] ?? '—') : 'Fechado'}</strong></div>
            <div><span>GAME OVER</span><strong>{game?.game_over_state ? 'Estado ' + game.game_over_state : 'Não'}</strong></div>
            <div><span>OCARINA</span><strong>{game?.ocarina_mode ? 'Modo ' + game.ocarina_mode + ' · última ' + game.last_played_song : 'Inativa'}</strong></div>
          </div>}
          {liveTab === 'DECISÃO' && <>
            <section className="panel"><div className="section-head">DECISÃO E RESULTADO</div><div className="decision"><span className="eyebrow">{snap?.last_decision?.skill ?? 'SEM AÇÃO'}</span><h3>{snap?.last_decision?.goal ?? 'Pronto para uma nova execução'}</h3><p>{snap?.last_decision?.summary ?? 'O modelo recebe estado estruturado, memória de mundo e perfis aprendidos de inimigos.'}</p>{snap?.last_result && <code>{snap.last_result.status} / {snap.last_result.reason}</code>}</div></section>
            <div className="decision-grid">
              <section className="panel"><div className="section-head">EVENTOS RECENTES</div><div className="event-list">{snap?.events.slice().reverse().map((event, index) => <div className="event" key={String(event.at) + '-' + String(index)}><time>{new Date(event.at * 1000).toLocaleTimeString('pt-BR')}</time><b>{event.kind}</b><span>{JSON.stringify(event.data)}</span></div>)}{!snap?.events.length && <p className="muted">Nenhum evento registrado.</p>}</div></section>
              <section className="panel"><div className="section-head">INTERVENÇÃO HUMANA</div><form className="hint-form" onSubmit={e => {e.preventDefault(); void action(async () => {await api('/hints', {text: hint}); setHint('');});}}><label>Instrução para a próxima decisão<textarea rows={3} maxLength={400} value={hint} onChange={e => setHint(e.target.value)} placeholder="Ex.: tente outra direção antes de atacar."/></label><button disabled={!active || !hint.trim() || busy}>Enviar instrução</button><p className="muted">Fica no log e marca esta run como assistida.</p></form></section>
            </div>
          </>}
        </div>
      </div>
      <section className="panel controls"><div className="section-head">02 / AGENTE</div><form onSubmit={event => { event.preventDefault(); void action(start); }}>
        <label>Provider<select value={config.provider} onChange={e => update('provider', e.target.value)}><option value="codex">Codex · ChatGPT</option><option value="openrouter">OpenRouter · API</option>{providers.some(p => p.id === 'demo') && <option value="demo">Simulador determinístico</option>}</select></label>
        <label>Modelo<select value={config.model} onChange={e => { const model = models.find(m => m.id === e.target.value); setConfig(c => ({...c, model: e.target.value, effort: model?.default_effort ?? null})); }}><option value="">{models.length ? 'Selecione' : 'Catálogo indisponível'}</option>{models.map(m => <option value={m.id} key={m.id} disabled={!m.structured_output}>{m.name}{!m.structured_output ? ' · sem JSON estruturado' : ''}</option>)}</select></label>
        <label>Reasoning effort<select disabled={!selected?.efforts.length} value={config.effort ?? ''} onChange={e => update('effort', e.target.value || null)}><option value="">{selected?.efforts.length ? 'Padrão do provider' : 'Não anunciado / não suportado'}</option>{selected?.efforts.map(e => <option key={e} value={e}>{e}</option>)}</select></label>
        <p className="muted">{selectedProvider?.message ?? 'Verifique a conexão do provider.'}</p>
        {active ? <><div className="current-model">EM EXECUÇÃO<br/><b>{snap?.config?.model}</b><br/>{snap?.config?.effort ?? 'effort padrão'}</div><BudgetSummary config={snap?.config ?? null}/><button type="button" disabled={busy || !config.model} onClick={() => void action(async () => acceptSnapshot(await api<Snapshot>('/model', {provider: config.provider, model: config.model, effort: config.effort})))}>Aplicar modelo na próxima decisão</button>{snap?.pending_switch && <p className="notice">Troca pendente: {snap.pending_switch.model}</p>}</> : <>
        <label>Objetivo<textarea rows={3} maxLength={400} value={config.goal} onChange={e => update('goal', e.target.value)}/></label><label>Memória<select value={config.memory_mode} onChange={e => update('memory_mode', e.target.value)}><option value="adaptive">Adaptive — aprende rotas e combate por inimigo deste modelo + effort</option><option value="isolated">Zero-shot — não reutiliza experiência entre runs</option></select></label>
        <RunBudgetFields config={config} update={update}/>
        <button className="primary" disabled={busy || diagnosticBusy || !!snap?.diagnostic_active || !connected || !snap?.bridge.connected || !config.model || !selectedProvider?.connected}>Iniciar execução</button></>}
        </form>{(active || snap?.status === 'starting') && <div className="buttons">
          {active && <button disabled={controlBusy} onClick={() => void control(snap?.status === 'running' ? 'pause' : 'resume')}>{snap?.status === 'running' ? 'Pausar IA' : 'Retomar IA'}</button>}
          {active && <button onClick={() => void control('take_control')}>Assumir controle</button>}
          <button className="danger" onClick={() => void control('stop')}>Encerrar</button>
        </div>}
        <p className="muted">“Assumir controle” marca a execução como assistida. Pausar a IA não pausa o jogo.</p></section></div>
      </>}
      {(tab === 'BENCHMARKS' || tab === 'EXECUÇÕES') && <><div className="notice">Comparação exploratória. Sem save/RNG certificados e sem critério de conclusão validado, não atribuímos percentual de jogo nem ranking de quem zerou.</div><div className="toolbar"><button onClick={() => void action(async () => setRows(await api<RunRow[]>('/runs')))}>Atualizar registros</button><span>{rows.length} execuções recentes</span></div><section className="panel table-wrap"><table><thead><tr><th>Execução / modelo</th><th>Effort</th><th>Tipo</th><th>Calls</th><th>Tokens</th><th>Custo</th><th>Mortes</th><th>Estado</th><th/></tr></thead><tbody>{rows.map(r => <tr key={r.id}><td><strong>{r.config.model}</strong><small>{r.id.slice(0, 8)} · {new Date(r.created_at * 1000).toLocaleString('pt-BR')}</small></td><td>{r.config.effort ?? 'padrão'}</td><td>{r.source === 'simulator' ? 'SIMULADA' : r.mixed ? 'MISTA' : r.assisted ? 'ASSISTIDA' : 'AUTÔNOMA'}</td><td>{number(r.metrics.calls)}</td><td>{number(r.metrics.total_tokens)}</td><td>{dollars(r.metrics.cost_usd)}{r.metrics.unknown_cost_calls > 0 && <small>Custo incompleto</small>}</td><td>{r.metrics.deaths}</td><td>{r.status}</td><td><button onClick={() => void action(async () => setDetail(await api(`/runs/${r.id}`)))}>Inspecionar</button></td></tr>)}</tbody></table>{!rows.length && <p className="empty">Nenhuma execução. Inicie pelo painel ao vivo.</p>}</section>{detail !== null && <section className="panel"><div className="section-head">REGISTRO · ÚLTIMAS 200 ENTRADAS POR TIPO<button onClick={() => saveJson('zelda-run-detail.json', detail)}>Exportar recorte JSON</button></div><pre>{JSON.stringify(detail, null, 2)}</pre></section>}</>}
      {tab === 'SKILLS' && <><p className="intro">O modelo decide objetivos e skills. Navegação e combate executam localmente com feedback; fight_enemy aprende uma política separada por adversário no modo Adaptive.</p><section className="panel"><table><thead><tr><th>Skill</th><th>Função</th><th>Estado</th><th>Versão</th></tr></thead><tbody>{skills.map(s => <tr key={s.id}><td><code>{s.id}</code></td><td>{s.name}</td><td><span className={`pill ${s.status}`}>{s.status === 'implemented' ? 'Implementada' : 'Planejada'}</span></td><td>{s.version ?? '—'}</td></tr>)}</tbody></table></section></>}
      {tab === 'MEMÓRIA' && <><p className="intro">No modo Adaptive, notas, transições, trajetórias e perfis de combate por inimigo persistem por modelo + effort + versão do contrato. Rotas legadas são preservadas, mas não são executadas sem um destino escolhido. Dicas, pausas e controle humano não promovem trajetórias autônomas.</p><section className="panel"><div className="section-head">GRAFO DE MUNDO OBSERVADO</div><table><thead><tr><th>Origem</th><th>Saída observada</th><th>Destino</th><th>Spawn</th><th>Travessias</th></tr></thead><tbody>{snap?.world_edges?.map(edge => <tr key={edge.id}><td><strong>{edge.from_scene_name || `SCENE ${edge.from_scene}`}</strong><small>ROOM {edge.from_room}</small></td><td>{edge.from_position?.length === 3 ? edge.from_position.map(v => number(v)).join(' / ') : '—'}</td><td><strong>{edge.to_scene_name || `SCENE ${edge.to_scene}`}</strong><small>ROOM {edge.to_room} · ENTRANCE {edge.entrance_index}</small></td><td>{edge.to_position?.length === 3 ? edge.to_position.map(v => number(v)).join(' / ') : '—'}</td><td>{edge.traversals}</td></tr>)}</tbody></table>{!snap?.world_edges?.length && <p className="empty">Nenhuma transição autônoma observada ainda.</p>}</section><section className="panel"><div className="section-head">ROTAS APRENDIDAS</div><table><thead><tr><th>Origem</th><th>Destino</th><th>Passos</th><th>Sucessos</th><th>Falhas</th></tr></thead><tbody>{snap?.trajectories?.map(t => <tr key={t.id}><td>SCENE {t.from_scene} / ROOM {t.from_room}</td><td>SCENE {t.to_scene} / ROOM {t.to_room}</td><td>{t.actions.length}</td><td>{t.successes}</td><td>{t.failures}</td></tr>)}</tbody></table>{!snap?.trajectories?.length && <p className="empty">Nenhuma rota aprendida ainda. Uma mudança de sala/cena autônoma promove a sequência de navegação que funcionou.</p>}</section><section className="panel memory-list">{snap?.memory.map(m => <article key={m.id}><span className="eyebrow">SCENE {m.scene}</span><p>{m.note}</p><small>{new Date(m.created_at * 1000).toLocaleString('pt-BR')}</small></article>)}{!snap?.memory.length && <p className="empty">Nenhuma nota de memória nesta execução.</p>}</section></>}
      {tab === 'CONEXÕES' && <div className="connection-grid"><section className="panel"><div className="section-head">CODEX / CHATGPT</div><div className="content"><h2>Login oficial, perfil isolado.</h2><p>O app-server gerencia a autenticação. O projeto não transforma seu token do ChatGPT em uma chave de API.</p><p className="muted">{providers.find(p => p.id === 'codex')?.message}</p>{providers.find(p => p.id === 'codex')?.cli_version && <p className="muted">Codex CLI {providers.find(p => p.id === 'codex')?.cli_version} · Astra {providers.find(p => p.id === 'codex')?.astra_cli_ready ? 'compatível' : 'requer 0.153.0+'}</p>}<button disabled={busy} onClick={() => void action(async () => setLogin(await api('/auth/codex', {device: false})))}>Conectar ChatGPT</button><button disabled={busy} onClick={() => void action(async () => setLogin(await api('/auth/codex', {device: true})))}>Usar código de dispositivo</button>{login && <div className="notice"><a href={login.authUrl ?? login.verificationUrl} target="_blank" rel="noreferrer">Abrir página oficial de autenticação ↗</a>{login.userCode && <p>Código: <code>{login.userCode}</code></p>}<p>Após concluir, clique em “Verificar conexões”.</p></div>}<pre>{JSON.stringify(providers.find(p => p.id === 'codex')?.limits ?? {info: 'Limites aparecerão quando informados pelo Codex.'}, null, 2)}</pre></div></section>
      <section className="panel"><div className="section-head">OPENROUTER / API</div><form className="content" onSubmit={e => {e.preventDefault(); void action(async () => {await api('/auth/openrouter', {key}); setKey(''); await refreshProviders();});}}><h2>Cobrança por utilização.</h2><label>Chave da API<input type="password" autoComplete="off" value={key} onChange={e => setKey(e.target.value)} placeholder="sk-or-…"/></label><button disabled={busy || !key || snap?.status === 'running'}>Conectar OpenRouter</button><p className="muted">A chave digitada permanece na memória do backend até ele encerrar. Para persistência local, use OPENROUTER_API_KEY no .env, nunca no repositório.</p><p>{providers.find(p => p.id === 'openrouter')?.message}</p></form></section><div className="toolbar"><button disabled={busy} onClick={() => void action(refreshProviders)}>Verificar conexões</button></div></div>}
      <footer><span>ZELDA AI PLAYER · LABORATÓRIO LOCAL</span><span>v0.1 · Eventos reais quando conectados · Nenhum custo inventado</span></footer>
    </main></div>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
