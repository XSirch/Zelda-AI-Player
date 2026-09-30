# Zelda AI Player

Jogador autônomo para **The Legend of Zelda: Ocarina of Time** no **Ship of Harkinian (SoH)**.

## Autonomy V3

Ao clicar **INICIAR**, três loops independentes trabalham em paralelo:

- **ML actor goal-conditioned:** PPO local em PyTorch continua emitindo analógico N64 e bits físicos dos botões, mas um prior geométrico camera-relative transforma `target_position`/ator/direção observados em viés da própria distribuição do stick. O PPO aprende correções residuais e todos os botões. Não existe `navigate_to`, rota de Zelda ou mapping semântico de botão.
- **Online learner:** PPO aprende com experiência recém-coletada e **RND (Random Network Distillation)** fornece curiosidade. O learner usa uma cópia separada da rede; backprop não interrompe os inputs.
- **Cognição LLM:** Codex/ChatGPT ou OpenRouter escolhe apenas o **próximo objetivo estratégico**. Quando a meta tem condição verificável, o runtime trava `objective + completion` até a telemetria provar conclusão. Troca de sala, stuck, waypoint local, progresso parcial e recuperação de rota não podem substituir a meta. A única exceção transitória é uma escolha semântica de diálogo, que não altera o objetivo. Não existe refresh periódico por `horizon_ms`.

A política recebe quatro frames estruturados consecutivos e contexto espacial absoluto. A motor policy v3 não trata mais o guidance como uma sugestão fraca dentro da Beta: o PPO amostra um **stick residual**, e o stick realmente enviado ao jogo é uma mistura determinística entre esse residual e o guidance estruturado. Assim, quando existe um waypoint/rota com força 0,86, aproximadamente 86% do analógico executado vem do guidance e apenas 14% fica para correção/exploração aprendida.

Essa mudança altera a semântica da ação, então o treino passa a usar `.local/ml/raw-controller-ppo-rnd-v3.pt`. O antigo `raw-controller-ppo-rnd-v2.pt` é preservado, mas não é reinterpretado. A route memory em `.local/ml/route-graph-v1.json` **é preservada e reutilizada**, pois contém apenas caminhos empiricamente percorridos e é independente dos pesos do PPO.

### Goal-conditioned motor

A cognição não fornece mais waypoints operacionais para metas rastreáveis. Ela entrega uma meta como `Obtain the Kokiri Sword` junto de um contrato de conclusão como `equipment:name=Kokiri Sword`. O runtime converte essa meta em intenção local neutra; route memory, frontiers, scene exits, atores observados e affordances físicas decidem o caminho sem mudar o objetivo. `summary` é mantido igual ao texto da meta apenas por compatibilidade do contrato.

O guidance v4 também usa os oito probes locais de colisão já observados pela bridge. Se o heading direto para o alvo estiver bloqueado, ele escolhe apenas um **desvio local transitável** que continue aproximadamente alinhado ao destino; isso não é A*, não cria uma rota e não contém conhecimento de Zelda. Se não existir desvio seguro, a força do prior cai para 30% e a cognição recebe `guidance_blocked` após bloqueio sustentado. Além disso, depois de 90 s sem expansão/progresso a força do alvo começa a cair, chegando a 25%, para impedir que um waypoint inacessível vire um ímã permanente.

Quando existe guidance forte de navegação, a distribuição de botões também recebe um viés de quietude sem bloquear nenhum botão. A exploração PPO agora é **annealed**: stick e botões têm coeficientes de entropia separados, e a entropia dos botões cai de forma linear até zero nas primeiras ~50 mil amostras treinadas. Há ainda um custo pela **contagem esperada de botões ativos**, em vez da antiga penalidade minúscula sobre a média das probabilidades. Isso evita que os 9 Bernoullis independentes mantenham button-mashing indefinidamente. Em `combat`, `dialogue` e `menu` o guidance de stick continua desligado quando apropriado.

Ao entrar no raio de um waypoint local, o controlador apenas avança sua execução local. Se houver uma meta rastreável travada, `intent_target_reached`, `world_transition`, `local_area_stuck`, `guidance_blocked` e progresso parcial **não chamam a IA para trocar a meta**. A próxima chamada estratégica ocorre quando o `ObjectiveTracker` confirma a condição de conclusão.

Quando a cognição está em `explore` sem alvo/direção estruturada, o sistema também não entrega mais o analógico inteiro ao acaso: a route memory escolhe um **frontier local observado** entre os probes de colisão transitáveis. O frontier escolhido vira um pequeno objetivo em coordenadas do mundo e fica **mantido** até ser alcançado, ficar ~2,5 s sem progresso ou expirar em ~7 s; ele não é recalculado a cada decisão ML. Direções `back/back_left/back_right` só são consideradas quando não existe nenhuma opção frontal/lateral transitável. Frontiers que travam entram em cooldown e também acumulam **falha persistente** no `route-graph-v1.json`; candidatos que já falharam perdem prioridade mesmo em runs futuras, e um sucesso posterior reduz essa penalidade. Se a mesma scene/room ficar sem expansão por ~20 s, ou acumular pressão suficiente de frontiers/arestas falhos sob uma meta rastreável, `scene_exits` passam a ter prioridade mesmo que o timer tenha sido reiniciado por reposicionamento. Nenhum mapa oculto é consultado.

### Rotas aprendidas

A route memory v1 transforma deslocamento real em um **grafo topológico dirigido**. Aproximadamente a cada célula de 80×50×80 unidades realmente atravessada, o controlador registra um nó e conecta somente o trecho que Link de fato percorreu. Não são criadas arestas reversas automaticamente, atalhos através de paredes nem conhecimento de collision que nunca foi visitado.

Quando a cognição pede um `target_position`/ator, o controlador procura no **grafo observado** uma sequência que realmente tenha sido percorrida e que aproxime Link do objetivo. Cada aresta dirigida guarda também o **ponto real de entrada** observado quando Link atravessou para a célula seguinte; replay usa esse gateway em vez do centro médio da célula, que pode cair do lado errado de um canto. Se já existir um caminho até perto do alvo, ele reutiliza os waypoints completos; se o alvo ainda for inédito, pode reutilizar apenas um trecho parcial que produza avanço geométrico relevante. Trechos percorridos mais vezes recebem maior confiança.

Uma rota parcial não pode virar outro ímã: ao chegar ao fim conhecido de um ramo que ainda não alcança o alvo, esse endpoint fica marcado como esgotado para aquele objetivo e não é reproduzido novamente até o grafo ganhar novos nós/trechos. Além disso, uma **aresta aprendida precisa continuar provando que funciona**: se por ~2,5 s ela não reduz a distância ao gateway (ou passa ~8 s sem concluir), entra em cooldown por ~20 s, acumula uma falha de confiabilidade e o planner procura outra aresta/saída/frontier. Assim, "Link já passou aqui uma vez" não significa "insista nesta direção para sempre". `local_dwell` continua sem reduzir a força de rota/frontier/scene exit; esses sinais têm seus próprios critérios de validade. O fade anti-loop fica restrito ao `target_position` direto. A memória separa rotas por scene/room, idade e mundo normal/espelhado.

A memória é incremental e persistente entre runs. Avançar por uma rota conhecida conta como progresso operacional, mas **não zera `local_dwell`** nem reabre `frontier_progress`; assim, uma volta A → B → C → A não consegue mascarar um loop.

A mesma `route-graph-v1.json` também guarda **affordances de interação aprendidas empiricamente**. Quando Link chega a uma saída/porta e existe `context_action`, o controlador não assume que "Open = A": ele testa um botão físico por vez (sem START), observa se houve transição de scene/room, diálogo ou outro efeito durável e só então grava a associação. Uma associação conhecida é reutilizada; após três falhas consecutivas ela é descartada e volta a ser explorada. Diálogo linear também tem prioridade física: enquanto `dialogue.active` estiver verdadeiro, stick/exploração ficam neutros; quando `can_advance` estiver disponível, o controlador aprende empiricamente qual botão avança o texto observando a página mudar/fechar. Choices continuam sob decisão da cognição e nunca são confirmados aleatoriamente pelo PPO. Como isso fica no route graph, o snapshot do champion congela também essas affordances. Runs de avaliação carregam a cópia read-only congelada junto com o champion.

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
- **META TRAVADA** e a condição local de conclusão, incluindo quantos replans foram ignorados enquanto a meta permaneceu ativa;
- route memory (nós, trechos, reusos e interações aprendidas), indicação `rota aprendida`, `frontier observado` ou `saída observada` conforme o guidance ativo;
- aprendizado contextual de interação, incluindo quando o bot está testando um botão físico e quantas associações já foram aprendidas;
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
