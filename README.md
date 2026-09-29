# Zelda AI Player

Jogador autônomo para **The Legend of Zelda: Ocarina of Time** no **Ship of Harkinian (SoH)**.

## Autonomy V3

Ao clicar **INICIAR**, três loops independentes trabalham em paralelo:

- **ML actor goal-conditioned:** PPO local em PyTorch continua emitindo analógico N64 e bits físicos dos botões, mas um prior geométrico camera-relative transforma `target_position`/ator/direção observados em viés da própria distribuição do stick. O PPO aprende correções residuais e todos os botões. Não existe `navigate_to`, rota de Zelda ou mapping semântico de botão.
- **Online learner:** PPO aprende com experiência recém-coletada e **RND (Random Network Distillation)** fornece curiosidade. O learner usa uma cópia separada da rede; backprop não interrompe os inputs.
- **Cognição LLM:** Codex/ChatGPT ou OpenRouter mantém apenas objetivo/intenção de alto nível. Uma inferência lenta não interrompe o controle motor. A cognição é **sparse/event-driven**: uma chamada inicial e novas chamadas somente em eventos estratégicos (transição, progresso durável, escolha/resolução relevante de diálogo, morte/boss) ou após stuck sustentado. Não existe refresh periódico por `horizon_ms`.

A política recebe quatro frames estruturados consecutivos e contexto espacial absoluto. A motor policy v3 não trata mais o guidance como uma sugestão fraca dentro da Beta: o PPO amostra um **stick residual**, e o stick realmente enviado ao jogo é uma mistura determinística entre esse residual e o guidance estruturado. Assim, quando existe um waypoint/rota com força 0,86, aproximadamente 86% do analógico executado vem do guidance e apenas 14% fica para correção/exploração aprendida.

Essa mudança altera a semântica da ação, então o treino passa a usar `.local/ml/raw-controller-ppo-rnd-v3.pt`. O antigo `raw-controller-ppo-rnd-v2.pt` é preservado, mas não é reinterpretado. A route memory em `.local/ml/route-graph-v1.json` **é preservada e reutilizada**, pois contém apenas caminhos empiricamente percorridos e é independente dos pesos do PPO.

### Goal-conditioned motor

Texto do Luna em `summary` não controla Link. Para um destino ser acionável, a cognição precisa preencher `target_position`, `target_actor_*` ou uma direção estruturada. O motor converte esse objetivo para um vetor de analógico relativo à câmera e o injeta como prior na **mesma distribuição Beta usada pelo PPO**; os log-probs continuam coerentes para treino on-policy.

O guidance v4 também usa os oito probes locais de colisão já observados pela bridge. Se o heading direto para o alvo estiver bloqueado, ele escolhe apenas um **desvio local transitável** que continue aproximadamente alinhado ao destino; isso não é A*, não cria uma rota e não contém conhecimento de Zelda. Se não existir desvio seguro, a força do prior cai para 30% e a cognição recebe `guidance_blocked` após bloqueio sustentado. Além disso, depois de 90 s sem expansão/progresso a força do alvo começa a cair, chegando a 25%, para impedir que um waypoint inacessível vire um ímã permanente.

Quando existe guidance forte de navegação, a distribuição de botões também recebe um viés de quietude sem bloquear nenhum botão. A exploração PPO agora é **annealed**: stick e botões têm coeficientes de entropia separados, e a entropia dos botões cai de forma linear até zero nas primeiras ~50 mil amostras treinadas. Há ainda um custo pela **contagem esperada de botões ativos**, em vez da antiga penalidade minúscula sobre a média das probabilidades. Isso evita que os 9 Bernoullis independentes mantenham button-mashing indefinidamente. Em `combat`, `dialogue` e `menu` o guidance de stick continua desligado quando apropriado.

Ao entrar no raio de um waypoint de movimento, o runtime emite `intent_target_reached` uma única vez e chama a cognição sparse para escolher o próximo ponto observado. Isso impede que um waypoint já atravessado continue puxando o motor para trás.

### Rotas aprendidas

A route memory v1 transforma deslocamento real em um **grafo topológico dirigido**. Aproximadamente a cada célula de 80×50×80 unidades realmente atravessada, o controlador registra um nó e conecta somente o trecho que Link de fato percorreu. Não são criadas arestas reversas automaticamente, atalhos através de paredes nem conhecimento de collision que nunca foi visitado.

Quando a cognição pede um `target_position`/ator, o controlador procura no **grafo observado** uma sequência que realmente tenha sido percorrida e que aproxime Link do objetivo. Se já existir um caminho até perto do alvo, ele reutiliza os waypoints completos; se o alvo ainda for inédito, pode reutilizar apenas um trecho parcial que produza avanço geométrico relevante. Isso permite repetir um contorno descoberto por exploração como A → B → C → D em vez de voltar a apontar em linha reta para a parede. Trechos percorridos mais vezes recebem maior confiança, e a força do guidance cresce gradualmente com essa evidência.

Uma rota parcial não pode virar outro ímã: ao chegar ao fim conhecido de um ramo que ainda não alcança o alvo, esse endpoint fica marcado como esgotado para aquele objetivo e não é reproduzido novamente até o grafo ganhar novos nós/trechos. A memória também separa rotas por scene/room, idade de Link e mundo normal/espelhado. Ela nunca inventa arestas reversas, atalhos ou o restante de uma rota não observada.

A memória é incremental e persistente entre runs. Avançar por uma rota conhecida conta como progresso operacional, mas **não zera `local_dwell`** nem reabre `frontier_progress`; assim, uma volta A → B → C → A não consegue mascarar um loop. Runs de avaliação carregam a cópia read-only congelada junto com o champion, para que uma avaliação posterior não seja alterada por rotas aprendidas depois.

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

Objetivos observáveis também geram uma pontuação humana separada do reward de treino. Exemplo: adquirir a **Kokiri Sword** vale **+100 pontos de conquista**, enquanto o PPO recebe um bônus escalado de **+3.0**. O reward total de um passo continua limitado a `[-5, +5]`, então a pontuação de UI não desestabiliza o treinamento. Na reward v7, curiosidade RND só paga surpresa acima do baseline e tem peso máximo pequeno; eventos genéricos e transições repetidas não geram reward.

O painel diferencia:
- pontos de conquista da run;
- updates PPO e amostras treinadas **nesta run**;
- totais persistidos no checkpoint;
- reward acumulado e média recente;
- conquistas concretas como equipamento, itens, story flags, transições novas, dano causado, inimigos e bosses derrotados.

Progresso que já existia no save ao iniciar a run é tratado como baseline e não gera conquista retroativa.

### Pressão contra ficar preso na mesma área

Além da penalidade de ficar literalmente parado, a reward v7 acompanha **expansão espacial grossa**. O mundo observado é dividido em macro-regiões locais de aproximadamente 500×160×500 unidades por scene/room. Entrar pela primeira vez numa macro-região, obter progresso durável, iniciar diálogo novo relevante, causar dano ou alcançar outro marco útil reinicia o relógio.

Circular por células pequenas, revisitar macro-regiões já conhecidas ou apertar botões enquanto continua perto do mesmo lugar **não reinicia** esse relógio. Após 60 s sem expansão/progresso entra `local_dwell`. Na reward v7 ele continua escalando com o tempo, mas foi reescalado de cerca de `-0.35` para no máximo `-0.02` por decisão ML. O valor antigo dominava o retorno (até ~`-3.5/s` a 10 Hz) e transformava o treino em um grande sinal negativo quase independente da ação. O relógio/stuck continua forte; apenas o gradiente PPO deixa de ser esmagado por essa taxa. O breakdown ao vivo mostra `local_dwell`, e o painel mostra o tempo `sem expansão`.

Esse mesmo relógio participa do detector de stuck da cognição: aos 90 s sem expansão local, o Luna pode receber um evento `local_area_stuck`, ainda sujeito ao cooldown de 180 s entre replans por stuck.

**Frontier progress:** punição negativa sozinha não indica qual ação é melhor. A reward v7 também mantém o maior raio alcançado desde a última expansão/progresso. Aumentar esse raio pela primeira vez gera `frontier_progress`; andar em círculos no mesmo raio não paga. Entrar numa nova macro-região rende `new_macro_region +0.8`. Quando a cognição fornece um alvo observado, `intent_progress` é simétrico para aproximar/afastar. Se o guidance executado estava bloqueado por colisão **ou reproduzindo uma rota aprendida**, esse shaping de distância reta é suprimido imediatamente: um contorno correto pode aumentar temporariamente a distância Euclidiana até o destino. Fora desses casos ele começa a desaparecer após 60 s sem expansão/progresso e chega a zero após mais 120 s. Assim, afastar-se para contornar uma parede não é punido e voltar para o mesmo waypoint obstruído não forma um reward loop.

**Coletas e baús:** a reward v6 também premia ganho observado de rupees, munição, vida e magia. O reward usa o **delta real do estado**: se a carteira, munição, vida ou magia já estiverem cheias e a coleta não aumentar o valor, o bônus é zero. Primeira aquisição de um item continua usando o milestone persistente mais forte, sem somar o bônus de ammo inicial. A bridge v3.1 emite `chest_opened` a partir do treasure flag nativo, uma única vez por baú, gerando reward e conquista próprios.

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

Cada completion guarda SHA-256 e metadados da run: tempo final, chamadas de cognição, tokens de input/output (além de cache/reasoning), custo reportado, breakdown por provider/modelo, updates, amostras, reward, modelo de cognição e estatísticas da route memory. O grafo de rotas usado naquele completion também é copiado para `completion-XXXX.routes.json` com SHA-256 próprio. `best-completion.pt` é apenas um alias atualizável para o menor tempo de conclusão observado; os arquivos individuais de policy/rotas nunca são sobrescritos.

O benchmark da run também fica no banco. As métricas ao vivo continuam derivadas das chamadas/segmentos, mas ao encerrar normalmente a run o runtime grava um snapshot final em `run_summaries`, exposto como `benchmark` em `GET /api/runs/{run_id}`. Nele, `elapsed_s` fica congelado e os campos `input_tokens`, `output_tokens`, `total_tokens`, `known_cost_usd` e `usage_by_model` preservam o consumo final. `cost_usd` permanece `null` quando o custo total não pode ser conhecido (por exemplo, Codex autenticado via ChatGPT ou uma chamada OpenRouter sem custo retornado), evitando transformar custo desconhecido em zero.

**INICIAR** continua usando o checkpoint de treino `.local/ml/raw-controller-ppo-rnd-v3.pt`. Quando existe um champion, **AVALIAR CHAMPION** carrega por padrão o completion mais recente, desativa PPO/RND training e usa decisões determinísticas do actor. A avaliação nunca salva sobre o champion nem cria um novo champion ao terminar.

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
- tempo da run ao vivo e tempo final congelado quando ela termina;
- tokens de input/output, total e número de chamadas da run;
- cache e reasoning tokens;
- custo API quando o provider reporta USD (ou custo parcial conhecido em runs mistas);
- cota restante do Codex/ChatGPT;
- route memory (nós, trechos e reusos) e indicação `rota aprendida` quando o guidance estiver reproduzindo um caminho observado;
- diagnóstico PPO ao vivo: botões esperados, percentual efetivo de guidance e quanto da fase de exploração ainda resta;
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

## Native Bridge v3.1 (SoH)

Revisão Shipwright fixada:

```text
HarbourMasters/Shipwright
d30fc192f2eb01ceea45bd1e12de61636cafbf86
```

A Autonomy V3 usa **Native Bridge v3.1 / Adapter v3.1** com wire protocol `3`.

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
4. retenção após restart pelo checkpoint ML e `route-graph-v1.json`;
5. descoberta de um contorno não reto, crescimento de nós/trechos e reutilização posterior exibida como `rota aprendida`;
6. continuidade de inputs durante inferência e treino;
7. neutralização imediata em **PARAR**;
8. tokens e cota do Codex atualizando sem interferir no controle.

Documentação:

- [Arquitetura](docs/ARCHITECTURE.md)
- [Bridge realtime v3](docs/REALTIME_V3.md)
- [Status](docs/STATUS.md)
- [Instruções para agentes](AGENTS.md)
