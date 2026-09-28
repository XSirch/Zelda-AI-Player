# Zelda AI Player

Jogador autônomo para **The Legend of Zelda: Ocarina of Time** no **Ship of Harkinian (SoH)**.

## Autonomy V3

Ao clicar **INICIAR**, três loops independentes trabalham em paralelo:

- **ML actor:** PPO local em PyTorch emite diretamente o analógico N64 e bits físicos dos botões. Não recebe `A = interact`, `B = attack`, `navigate_to`, `fight_enemy` ou macros equivalentes.
- **Online learner:** PPO aprende com experiência recém-coletada e **RND (Random Network Distillation)** fornece curiosidade. O learner usa uma cópia separada da rede; backprop não interrompe os inputs.
- **Cognição LLM:** Codex/ChatGPT ou OpenRouter mantém apenas objetivo/intenção de alto nível. Uma inferência lenta não interrompe o controle motor. A cognição é **sparse/event-driven**: uma chamada inicial e novas chamadas somente em eventos estratégicos (transição, progresso durável, escolha/resolução relevante de diálogo, morte/boss) ou após stuck sustentado. Não existe refresh periódico por `horizon_ms`.

A política recebe quatro frames estruturados consecutivos e contexto espacial absoluto, permitindo aprender timing e rotas específicas. O checkpoint persistente fica em `.local/ml/raw-controller-ppo-rnd-v2.pt`.

## Reward

Não existe roteiro hard-coded de Zelda no reward. Os sinais são observáveis:

- curiosidade RND;
- novas regiões, atores e eventos;
- transições de scene/room;
- mudanças de diálogo/contexto;
- progresso persistente de inventário/quest/equipamento;
- progresso em direção a alvo observado fornecido pela cognição;
- dano causado/recebido e morte;
- pequeno custo por button-mashing.

Memórias persistentes são criadas somente a partir de evidência observada, como diálogo realmente recebido, item/equipamento adquirido, morte e transições realmente atravessadas.

### Pontos de conquista vs reward PPO

Objetivos observáveis também geram uma pontuação humana separada do reward de treino. Exemplo: adquirir a **Kokiri Sword** vale **+100 pontos de conquista**, enquanto o PPO recebe um bônus escalado de **+3.0**. O reward total de um passo continua limitado a `[-5, +5]`, então a pontuação de UI não desestabiliza o treinamento. Na reward v4, curiosidade RND só paga surpresa acima do baseline e tem peso máximo pequeno; eventos genéricos e transições repetidas não geram reward.

O painel diferencia:
- pontos de conquista da run;
- updates PPO e amostras treinadas **nesta run**;
- totais persistidos no checkpoint;
- reward acumulado e média recente;
- conquistas concretas como equipamento, itens, story flags, transições novas, dano causado, inimigos e bosses derrotados.

Progresso que já existia no save ao iniciar a run é tratado como baseline e não gera conquista retroativa.

### Pressão contra ficar preso na mesma área

Além da penalidade de ficar literalmente parado, a reward v4 acompanha **expansão espacial grossa**. O mundo observado é dividido em macro-regiões locais de aproximadamente 500×160×500 unidades por scene/room. Entrar pela primeira vez numa macro-região, obter progresso durável, iniciar diálogo novo relevante, causar dano ou alcançar outro marco útil reinicia o relógio.

Circular por células pequenas, revisitar macro-regiões já conhecidas ou apertar botões enquanto continua perto do mesmo lugar **não reinicia** esse relógio. Após 60 s sem expansão/progresso entra `local_dwell`; a penalidade cresce até cerca de `-0.35` por decisão ML após mais 180 s. O breakdown ao vivo mostra `local_dwell`, e o painel mostra o tempo `sem expansão`.

Esse mesmo relógio participa do detector de stuck da cognição: aos 90 s sem expansão local, o Luna pode receber um evento `local_area_stuck`, ainda sujeito ao cooldown de 180 s entre replans por stuck.

**Frontier progress:** punição negativa sozinha não indica qual ação é melhor. A reward v4 também mantém o maior raio alcançado desde a última expansão/progresso. Aumentar esse raio pela primeira vez gera `frontier_progress`; andar em círculos no mesmo raio não paga. Entrar numa nova macro-região rende `new_macro_region +0.8`. Quando a cognição fornece um alvo observado, `intent_progress` também recebe peso maior e simétrico para aproximar/afastar.

### Champions e avaliação congelada

Quando uma run de **treino** emite o evento nativo `game_completed`, o runtime espera o último rollout/gradient update terminar, salva o checkpoint de treino e cria um snapshot imutável:

```text
.local/ml/champions/
  completion-0001.pt
  completion-0001.json
  completion-0002.pt
  completion-0002.json
  best-completion.pt
  best-completion.json
```

Cada completion guarda SHA-256 e metadados da run (tempo, updates, amostras, reward e modelo de cognição). `best-completion.pt` é apenas um alias atualizável para o menor tempo de conclusão observado; os arquivos `completion-XXXX.pt` nunca são sobrescritos.

**INICIAR** continua usando o checkpoint de treino `.local/ml/raw-controller-ppo-rnd-v2.pt`. Quando existe um champion, **AVALIAR CHAMPION** carrega por padrão o completion mais recente, desativa PPO/RND training e usa decisões determinísticas do actor. A avaliação nunca salva sobre o champion nem cria um novo champion ao terminar.

Para testar retenção desde o começo do jogo: mantenha `.local/ml/` e o banco de conhecimento, carregue um **save novo** do OoT e então clique **AVALIAR CHAMPION**. O sistema não cria/apaga saves do jogo automaticamente.

API avançada:

```text
GET  /api/champions
POST /api/evaluate              {"champion_id": null}
POST /api/evaluate              {"champion_id": "completion-0001"}
POST /api/evaluate              {"champion_id": "best"}
```

## Painel

A interface principal mostra somente:

- conexão SoH / bridge realtime / cognição;
- pensamento operacional;
- analógico e botões físicos em tempo real;
- tokens e número de chamadas da run;
- cache e reasoning tokens;
- custo API quando o provider reporta USD;
- cota restante do Codex/ChatGPT;
- **INICIAR / PARAR** e **AVALIAR CHAMPION** quando houver um completion salvo.

Para Codex autenticado por ChatGPT, o app-server fornece tokens, mas custo em USD permanece **—** quando não há cobrança API reportada. A cota é lida separadamente por `account/rateLimits/read`, sem chamada de modelo, e mostra somente as janelas realmente devolvidas pelo Codex.

## Primeira configuração

```powershell
cd C:\Projetos\Zelda-AI-Player
uv sync
uv run zelda-ai init
uv run zelda-ai auth-codex
```

Configuração padrão da cognição:

```text
ZELDA_AGENT_PROVIDER=codex
ZELDA_AGENT_MODEL=gpt-6-luna
ZELDA_AGENT_EFFORT=xhigh
```

Altere esses valores no `.env` quando necessário.

## Build do painel

```powershell
cd C:\Projetos\Zelda-AI-Player\web
npm install
npm run build
cd ..
```

## Native Bridge v3.0 (SoH)

Revisão Shipwright fixada:

```text
HarbourMasters/Shipwright
d30fc192f2eb01ceea45bd1e12de61636cafbf86
```

A Autonomy V3 usa **Native Bridge v3.0 / Adapter v3.0** com wire protocol `3`.

Instale/atualize a bridge:

```powershell
cd C:\Projetos\Zelda-AI-Player
uv run python scripts/integrate_soh.py C:\Projetos\Shipwright-AI
```

Depois recompile o SoH. No computador atual, o build usa Visual Studio 2026 com toolset v143:

```powershell
cd C:\Projetos\Shipwright-AI
git submodule update --init --recursive

& 'C:\Program Files\CMake\bin\cmake.exe' `
  -S . `
  -B "build/x64" `
  -G "Visual Studio 18 2026" `
  -T v143 `
  -A x64

& 'C:\Program Files\CMake\bin\cmake.exe' `
  --build .\build\x64 `
  --config Release `
  --target GenerateSohOtr

& 'C:\Program Files\CMake\bin\cmake.exe' `
  --build .\build\x64 `
  --config Release
```

## Executar

Terminal 1:

```powershell
cd C:\Projetos\Zelda-AI-Player
uv run zelda-ai serve
```

Terminal 2:

```powershell
cd C:\Projetos\Zelda-AI-Player
uv run zelda-ai launch-soh "C:\Projetos\Shipwright-AI\x64\Release\soh.exe"
```

Abra **http://127.0.0.1:8787**, carregue um save jogável e clique **INICIAR**.

## OpenRouter

Configure `OPENROUTER_API_KEY` no `.env` e altere `ZELDA_AGENT_PROVIDER/openrouter` + `ZELDA_AGENT_MODEL`. Não existe retry automático de chamada paga.

## Validação local obrigatória

```powershell
cd C:\Projetos\Zelda-AI-Player
uv sync
uv run pytest -q

cd web
npm run build
```

Não considere a V3 validada no Windows enquanto esses comandos não passarem após o pull/merge.

## Status

A arquitetura de aprendizado contínuo está implementada, mas uma política PPO nova começa essencialmente sem habilidade. A próxima evidência necessária é treinamento real no SoH, medindo:

1. descoberta útil de controles sem mappings semânticos;
2. melhoria de navegação após tentativas repetidas;
3. aprendizado de combate por experiência temporal;
4. retenção após restart pelo checkpoint ML;
5. continuidade de inputs durante inferência e treino;
6. neutralização imediata em **PARAR**;
7. tokens e cota do Codex atualizando sem interferir no controle.

Documentação:

- [Arquitetura](docs/ARCHITECTURE.md)
- [Bridge realtime v3](docs/REALTIME_V3.md)
- [Status](docs/STATUS.md)
- [Instruções para agentes](AGENTS.md)
