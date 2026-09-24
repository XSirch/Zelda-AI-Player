import { useState } from 'react';
import type { RunConfig } from './types';

type BudgetField = 'max_calls' | 'max_tokens' | 'max_cost_usd' | 'max_output_tokens' | 'max_runtime_s';
const fields: Array<{ id: BudgetField; label: string; max: number; min: number; step: number }> = [
  { id: 'max_calls', label: 'Máximo de chamadas', min: 0, max: 10000, step: 1 },
  { id: 'max_tokens', label: 'Orçamento total de tokens', min: 0, max: 10000000, step: 1 },
  { id: 'max_cost_usd', label: 'Orçamento OpenRouter (USD)', min: 0, max: 1000, step: .01 },
  { id: 'max_output_tokens', label: 'Saída por resposta OpenRouter (inclui reasoning)', min: 256, max: 16384, step: 1 },
  { id: 'max_runtime_s', label: 'Duração máxima (segundos)', min: 0, max: 86400, step: 1 },
];

export function BudgetSummary({ config }: { config: RunConfig | null }) {
  if (!config) return null;
  const show = (value: number) => value === 0 ? 'Sem limite' : value.toLocaleString('pt-BR');
  return <p className="muted">Chamadas: {show(config.max_calls)} · Tokens: {show(config.max_tokens)}
    <br/>USD: {show(config.max_cost_usd)} · Duração: {show(config.max_runtime_s)}{config.max_runtime_s > 0 ? ' s' : ''}</p>;
}

export function RunBudgetFields({ config, update }: {
  config: RunConfig; update: <K extends keyof RunConfig>(field: K, value: RunConfig[K]) => void;
}) {
  const [blank, setBlank] = useState<Partial<Record<BudgetField, boolean>>>({});
  return <details><summary>Limites e identificação</summary>
    <label>Checkpoint (identificação manual)<input value={config.checkpoint_label}
      onChange={event => update('checkpoint_label', event.target.value)}/></label>
    {fields.map(field => <label key={field.id}>{field.label}
      <input required name={field.id} type="number" min={field.min} max={field.max} step={field.step}
        aria-describedby={`${field.id}-help`} value={blank[field.id] ? '' : config[field.id]}
        onChange={event => {
          const text = event.target.value;
          setBlank(old => ({ ...old, [field.id]: text === '' }));
          // An empty field is invalid; it must NOT silently become an unlimited budget.
          if (text !== '' && Number.isFinite(Number(text))) update(field.id, Number(text));
        }}/>
      <small id={`${field.id}-help`} className="muted">{field.min === 0
        ? (config[field.id] === 0 && !blank[field.id] ? 'Sem limite — este teto está desativado.' : '0 = sem limite para este campo.')
        : 'Limite por resposta; 0 não se aplica.'}</small>
    </label>)}
    <p className="muted">Os limites são independentes. Zerar tokens não desativa o teto de chamadas, custo ou duração.
      Tokens, chamadas e custos continuam registrados. Limites e cobrança do provider não são removidos.</p>
    {config.provider === 'openrouter' && config.max_cost_usd === 0 &&
      <p className="notice">Sem teto de USD no harness. O gasto dependerá do uso e dos limites da sua chave OpenRouter.</p>}
    <p className="muted">O Codex não expõe limite de saída por este adaptador. Custo não informado permanece desconhecido.</p>
  </details>;
}
