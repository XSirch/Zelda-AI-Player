# Zelda AI Player

Jogador autônomo para **The Legend of Zelda: Ocarina of Time** no **Ship of Harkinian (SoH)**.

A fundação do motor V4 corrige a projeção horizontal invertida do controle nativo e separa os fragmentos PPO nas intervenções. O lote G1 alcançou 99/100 saídas e retornos físicos com o Arquivo 2, sem chamadas de IA. O currículo de superfícies mede o aprendizado local separadamente. A campanha completa continua sem qualificação; os resultados e as limitações estão em [docs/STATUS.md](docs/STATUS.md).

## Laya base: treino e avaliação local

O Laya agora tem um piloto para atravessar saídas nativas observadas. O planejador mantém pontos de passagem no mundo, aceita aproximações parciais pela colisão atual e exige uma transição física seguida de três observações novas com Link parado no chão. Chegar perto da saída não encerra a tarefa. O objetivo rastreável continua sendo obter a Kokiri Sword durante toda a tentativa.

Os dois primeiros lotes saíram da casa em **2/3 tentativas cada**, usando somente o analógico do candidato congelado, sem mistura com o controle de referência, treino ou chamadas a provedores. Um desvio que primeiro se afastou da saída funcionou no jogo; a falha restante deixou a posição de Link sem ligações na malha observada. O adapter **3.10** testa uma malha menor somente nesse caso, preservando as verificações de corpo e piso. O lote seguinte concluiu **6/6 saídas**, com **5/5 resets** no mesmo processo; o modelo respondeu em até **37,5 ms** nesse lote. A amostra ainda é pequena e o tempo completo de reação física permanece sem medição. Resultados completos ficam no [registro do controle de portais](docs/validation/laya_portal_control_2026-10-05.json).

Este piloto continua fora do botão INICIAR. Ele avalia saídas de piso alcançáveis pela telemetria atual; ainda não comprova acesso à escada, coleta da espada ou conclusão da campanha. Para executá-lo com a base e o candidato locais existentes:

```powershell
uv run python -m zelda_ai.laya_portal_evaluation C:/Projetos/Shipwright-AI/x64/Release/soh.exe --source-home .local/qualification/g1-da24d7be4d70/seed-home --episodes 3 --candidate .local/laya/candidate-numeric-stick-mse --base .local/laya/base --python .local/laya-env/Scripts/python.exe
```

O adapter **3.9** separa o contato com uma borda escalável das ligações de caminhada. A verificação de chegada exige contato real com o chão: estar no alto durante um salto ou uma animação de subida não basta. O motor identifica os estados de natação, mergulho e movimento submerso pelas flags nativas e interrompe a política de caminhada ao entrar na água. Os controladores e o aprendizado desses movimentos ainda precisam de validação no jogo.

A avaliação de um candidato congelado agora mantém **um processo do SoH** e usa o reset normal entre os blocos de tarefas. O atalho é configurado somente na cópia de QA; cada reset exige input consumido, retorno observado ao título e um novo carregamento normal do Arquivo 2. Os comandos e as respostas antigas do modelo são liberados antes de continuar. Um reset recarrega o save de trabalho, com o progresso que o próprio jogo salvou. Os blocos reutilizados continuam pertencendo ao mesmo grupo nativo; a coleta para treino mantém processos separados para preservar a divisão entre treino, validação e teste. `--fresh-processes` permite uma avaliação com instâncias separadas.

O adapter **3.8** também verifica paredes na parte inferior do corpo antes de autorizar uma ligação de caminhada. O novo lote concluiu **84/84 caminhadas iniciadas dentro da casa**, mas deixou **16 das 100 tarefas planejadas sem execução**. O currículo evita alvos perto de saídas; nos últimos estados registrados antes das recusas, havia um caminho observado até uma saída. Outro lote concluiu **15/15 caminhadas no chão externo**, após saída e descida feitas pelo motor de referência. Esses resultados ainda não qualificam a navegação geral. A direção permanece no mundo e o analógico acompanha o ângulo de input atual do SoH; a câmera top-down continua sem validação nativa. Veja o [registro das colisões inferiores](docs/validation/nav_lower_body_2026-10-05.json).

No adapter 3.7, os dois lotes anteriores concluíram **15/15 caminhadas curtas cada**, com os mesmos pesos congelados. No segundo, o controle acompanhou 257 mudanças de orientação da câmera e teve resposta p95 de **29,8 ms**, sem respostas expiradas. O controle preso à escada continua separado. Uma tentativa maior parou em **79/80**, deixando 20 tarefas sem execução por falha de conexão no início da quinta sessão. Veja o [registro histórico das colisões e da câmera](docs/validation/nav_backfaces_2026-10-05.json).

O laboratório agora tem um perfil separado para **descida já presa à escada**. Ele recebe telemetria numérica do alvo, da ligação à escada e das orientações de Link e do input. A saída é analógico físico; uma mudança de câmera invalida uma resposta antiga, em vez de aplicar a transformação da caminhada. A aproximação ainda pertence ao controle de referência, e esse perfil continua fora do botão INICIAR.

O piloto de 05/10 treinou a base com 228 ações válidas, separadas por processo do jogo. O erro reservado caiu de 32 para 0,5 unidade. Na avaliação física, houve **0/2 descidas concluídas antes do treino** e **2/2 depois**; cada lote planejava três sessões, mas a primeira falhou durante a carga normal do save. Os exemplos contêm uma única direção, e um analógico constante foi melhor offline. A latência do candidato teve p95 de **90,8 ms**, máximo de **224,5 ms** e três respostas expiradas; o tempo completo de reação física ainda não foi medido. Esse resultado é limitado à descida avaliada. Veja [o registro de aprendizado da escada](docs/validation/laya_ladder_learning_2026-10-05.json).

Coleta e treino desse perfil usam diretórios novos e preservam o candidato de caminhada:

```powershell
uv run python -m zelda_ai.laya_ladder_curriculum C:/Projetos/Shipwright-AI/x64/Release/soh.exe --source-home .local/qualification/g1-da24d7be4d70/seed-home --sessions 6
# Substitua o caminho abaixo pelo dataset informado pela coleta.
$dadosEscada = '.local/qualification/nome-do-lote/dataset'
.local/laya-env/Scripts/python.exe -m zelda_ai.laya_training train .local/laya/base $dadosEscada .local/laya/escada-candidata-nova --numeric-telemetry --loss-mode stick_mse --steps 200 --batch-size 16 --baseline-output .local/laya/escada-antes-nova
uv run python -m zelda_ai.laya_ladder_curriculum C:/Projetos/Shipwright-AI/x64/Release/soh.exe --source-home .local/qualification/g1-da24d7be4d70/seed-home --sessions 3 --candidate .local/laya/escada-candidata-nova --base .local/laya/base --python .local/laya-env/Scripts/python.exe
```

O controle assíncrono do Laya agora preserva a direção escolhida em coordenadas do mundo e reprojeta o analógico pela orientação de input atual do SoH. Isso acompanha a câmera girando dentro de uma sala; a vista superior usa o ângulo nativo, sem depender da direção visual. Os testes automatizados cobrem esses casos. No jogo, o novo lote exercitou a rotação e concluiu **25/30 caminhadas**, abaixo das 28/30 anteriores; não demonstrou ganho de confiabilidade nem entrou em vista superior. As falhas e os limites estão no [registro de câmera](docs/validation/laya_camera_2026-10-04.json).

O treino parte do [Laya base publicado](https://huggingface.co/convaiinnovations/laya), com revisão e SHA-256 fixados, em um ambiente CUDA separado do aplicativo. Nenhum peso ou dado de trading entra no treino. O novo candidato recebe telemetria numérica limitada em uma projeção treinável ligada às camadas de decisão do Laya. O encoder congelado reutiliza apenas a representação do esquema fixo; os valores observados passam pela rede a cada decisão.

A coleta ampliada produziu 1.788 ações consumidas de 114 tarefas bem-sucedidas em 120 tentativas reais. A divisão mantém sessões nativas inteiras separadas, com quatro sessões para treino. O treino que minimiza o erro do analógico executado reduziu o erro médio offline de 15,57 para 5,63 unidades. Com o mesmo seed, os lotes reais passaram de **18/30** com classificação para **26/30** com esse treino e **28/30** após usar o contato físico com paredes na navegação local. As posições seguintes dependem do movimento de cada modelo; essa comparação não é perfeitamente pareada.

A execução local agora mantém os pontos de passagem observados em coordenadas do mundo. O recentramento da malha não arrasta o ponto adiante, e o relógio da tarefa reconhece avanço físico no desvio. Com os mesmos pesos congelados, uma avaliação maior, com seed novo e cinco sessões contínuas, concluiu **94/100 caminhadas curtas**. As seis falhas por timeout foram preservadas. Esse perfil continua restrito à geometria local observada na casa de Link; ainda não é navegação geral qualificada.

O lote histórico de 94/100 manteve zero mistura com o controle de referência e zero atualização durante a avaliação. As 3.096 respostas locais chegaram dentro da meta de **100 ms por decisão**, com percentil 95 de **28,6 ms** e máximo de **39,2 ms**. A idade máxima da saída amostrada foi de 85,6 ms nesse lote; outros lotes excederam 100 ms nessa medida. O tempo completo até a reação física ainda precisa de medição.

O candidato continua experimental e não foi conectado automaticamente ao botão **INICIAR**. A caminhada ainda não atingiu a qualificação exigida. O perfil cobre somente caminhada: não aprende botões, escadas, combate, diálogos ou a coleta da Kokiri Sword. Resultados, falhas e limites estão em [docs/STATUS.md](docs/STATUS.md), no [índice dos pontos de passagem](docs/validation/laya_corridor_2026-10-04.json), no [treino com erro físico](docs/validation/laya_stick_loss_2026-10-04.json) e no [histórico de avaliação](docs/validation/laya_runtime_2026-10-04.json). O [primeiro piloto](docs/validation/laya_base_2026-10-04.json) permanece como evidência histórica.

Preparação e execução no Windows com NVIDIA/CUDA, sem chamadas a Codex/OpenRouter:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_laya.ps1
.local/laya-env/Scripts/python.exe -m zelda_ai.laya_training prepare-base .local/laya/base

# Use diretórios de saída novos e uma cópia local existente de saves/configuração.
# Carrega o Arquivo 2 pelos controles normais e coleta apenas telemetria/recibos.
uv run python -m zelda_ai.laya_curriculum C:/Projetos/Shipwright-AI/x64/Release/soh.exe --source-home .local/qualification/g1-da24d7be4d70/seed-home --sessions 3 --tasks 20

# Substitua o nome abaixo pelo lote informado pela coleta, que já contém o dataset.
$lote = 'nome-do-lote'
.local/laya-env/Scripts/python.exe -m zelda_ai.laya_training train .local/laya/base ".local/qualification/$lote/dataset" .local/laya/candidato-novo --steps 1000 --batch-size 16 --numeric-telemetry --loss-mode stick_mse
.local/laya-env/Scripts/python.exe -m zelda_ai.laya_training benchmark .local/laya/base ".local/qualification/$lote/dataset" --candidate .local/laya/candidato-novo
uv run python -m zelda_ai.laya_curriculum C:/Projetos/Shipwright-AI/x64/Release/soh.exe --source-home .local/qualification/g1-da24d7be4d70/seed-home --sessions 3 --tasks 5 --candidate .local/laya/candidato-novo --base .local/laya/base --python .local/laya-env/Scripts/python.exe
```

O loader valida a base, a origem da biblioteca e os pesos do candidato. A avaliação usa cópias de trabalho isoladas, libera o controle ao encerrar e verifica os hashes dos artefatos congelados. Entrada inválida, dados sem recibos, mistura de sessões ou ausência de CUDA causam erro explícito. Modelos, telemetria e saves ficam em `.local/` e não são distribuídos no repositório.

Para avaliar após uma saída real da sala inicial, acrescente `--cross-initial-portal` ao comando do currículo. O motor V3 congelado faz essa preparação pelos controles normais; suas ações não entram nos resultados nem no treino do Laya. O primeiro lote confirmou três saídas da casa e **15/15 caminhadas do Laya na plataforma externa**. Todas permaneceram na altura da plataforma: descida, exploração da floresta e coleta da espada continuam sem qualificação. Veja o [registro da avaliação externa](docs/validation/laya_portal_walking_2026-10-04.json).

A investigação de descida usa `--descend-observed-ledge` junto de `--cross-initial-portal`. Essa preparação experimental recebe somente propostas de piso da colisão local atual, tenta até três regiões distintas e exige aterrissagem parada em três observações novas. As falhas ficam registradas e impedem o início do candidato naquele save de trabalho. Ações dessa preparação pertencem ao controlador de referência e ficam fora do treino do Laya. O [registro de descida](docs/validation/laya_descent_2026-10-04.json) preserva as tentativas anteriores.

O adapter 3.6 passou a expor o piso local que já era observado 180 unidades abaixo da plataforma. A caminhada agora interrompe ao prender Link à escada e entrega essa preparação a um controlador separado, que testa analógicos físicos sem botões e aceita uma direção somente após observar descida com input consumido. A calibração perde validade quando a câmera ou Link giram em relação à orientação original, inclusive numa rotação gradual. No novo lote, a preparação desceu até o chão em **3/3 sessões**, seguida de **15/15 caminhadas curtas do Laya** na altura −80. A descida é controle de referência; o candidato ainda não aprendeu escadas. A vista superior continua coberta por testes automatizados, com validação nativa pendente. Veja o [registro da escada](docs/validation/laya_ladder_2026-10-04.json).

O perfil de caminhada também libera respostas pendentes ao receber uma tarefa de queda ou escalada. A correção pela câmera vale para movimento no solo; o controle preso a uma escada exige uma política de modo própria. A consulta nativa passou a observar as duas faces das superfícies escaláveis, mantendo a primeira colisão como barreira de oclusão. Detectar essa superfície ainda exige comprovar aproximação, interação e travessia no jogo.

## Autonomy V3

Ao clicar **INICIAR**, três loops independentes trabalham em paralelo:

- **ML actor goal-conditioned:** PPO local em PyTorch emite um residual do analógico N64 e bits físicos dos botões. O analógico executado combina esse residual com a orientação geométrica observada fora da distribuição da rede. As tarefas físicas locais podem assumir o controle temporariamente e seus passos ficam fora do PPO. Não existe rota de Zelda nem associação semântica fixa de botão.
- **Online learner:** PPO aprende com experiência recém-coletada e **RND (Random Network Distillation)** fornece curiosidade. O learner usa uma cópia separada da rede; backprop não interrompe os inputs.
- **Cognição LLM:** Codex/ChatGPT ou OpenRouter escolhe apenas o **próximo objetivo estratégico**. Quando a meta tem condição verificável, o runtime trava `objective + completion` até a telemetria provar conclusão. Troca de sala, stuck, waypoint local, progresso parcial e recuperação de rota não podem substituir a meta. A única exceção transitória é uma escolha semântica de diálogo, que não altera o objetivo. Não existe refresh periódico por `horizon_ms`.

A política recebe quatro frames estruturados consecutivos e contexto espacial absoluto. A motor policy v3 não trata mais o guidance como uma sugestão fraca dentro da Beta: o PPO amostra um **stick residual**, e o stick realmente enviado ao jogo é uma mistura determinística entre esse residual e o guidance estruturado. Assim, quando existe um waypoint/rota com força 0,86, aproximadamente 86% do analógico executado vem do guidance e apenas 14% fica para correção/exploração aprendida.

Essa mudança altera a semântica da ação, então o treino passa a usar `.local/ml/raw-controller-ppo-rnd-v3.pt`. O antigo `raw-controller-ppo-rnd-v2.pt` é preservado, mas não é reinterpretado. A route memory em `.local/ml/route-graph-v1.json` **é preservada e reutilizada**, pois contém apenas caminhos empiricamente percorridos e é independente dos pesos do PPO.

A **Room Map Memory v1** fica separada em `.local/ml/room-map-v1.json`. Ela acumula apenas geometria/landmarks observados: posições reais de entrada e transição, `scene_exits`, atores `door`, células transitadas/NavMesh/probes, endpoints bloqueados e affordances verticais. Ao voltar a uma room conhecida, esse mapa pode lembrar onde Link realmente saiu antes mesmo de a saída aparecer no snapshot atual; o `route-graph-v1.json` continua sendo a autoridade sobre quais caminhos dirigidos foram de fato percorridos.

### Goal-conditioned motor

A cognição não fornece mais waypoints operacionais para metas rastreáveis. Ela entrega uma meta como `Obtain the Kokiri Sword` junto de um contrato de conclusão como `equipment:name=Kokiri Sword`. O runtime converte essa meta em intenção local neutra; route memory, frontiers, scene exits, atores observados e affordances físicas decidem o caminho sem mudar o objetivo. `summary` é mantido igual ao texto da meta apenas por compatibilidade do contrato.

O guidance v4 também usa os oito probes locais de colisão já observados pela bridge. Se o heading direto para o alvo estiver bloqueado, ele escolhe apenas um **desvio local transitável** que continue aproximadamente alinhado ao destino; isso não é A*, não cria uma rota e não contém conhecimento de Zelda. Se não existir desvio seguro, a força do prior cai para 30% e a cognição recebe `guidance_blocked` após bloqueio sustentado. Além disso, depois de 90 s sem expansão/progresso a força do alvo começa a cair, chegando a 25%, para impedir que um waypoint inacessível vire um ímã permanente.

Quando existe guidance forte de navegação, a distribuição de botões também recebe um viés de quietude sem bloquear nenhum botão. A exploração PPO agora é **annealed**: stick e botões têm coeficientes de entropia separados, e a entropia dos botões cai de forma linear até zero nas primeiras ~50 mil amostras treinadas. Há ainda um custo pela **contagem esperada de botões ativos**, em vez da antiga penalidade minúscula sobre a média das probabilidades. Isso evita que os 9 Bernoullis independentes mantenham button-mashing indefinidamente. Em `combat`, `dialogue` e `menu` o guidance de stick continua desligado quando apropriado.

Ao entrar no raio de um waypoint local, o controlador apenas avança sua execução local. Se houver uma meta rastreável travada, `intent_target_reached`, `world_transition`, `local_area_stuck`, `guidance_blocked` e progresso parcial **não chamam a IA para trocar a meta**. A próxima chamada estratégica ocorre quando o `ObjectiveTracker` confirma a condição de conclusão.

Quando a cognição está em `explore` sem alvo/direção estruturada, o sistema não entrega mais o analógico inteiro ao acaso. Primeiro procura uma **travessia vertical observada ainda não visitada** entre ladders, escadas/slope, ledges e paredes escaláveis produzidas pela collision do SoH. Essa travessia recebe autoridade determinística do analógico e não entra no PPO, porque o residual da rede não causou a ação executada. Se a engine expuser um `context_action` para subir/descer, o botão continua não sendo hard-coded: o sistema testa um botão físico por vez e só aprende a associação quando observa estado de ladder/ledge, movimento ou outra consequência real.

Sem uma travessia vertical nova, a route memory escolhe um **frontier local observado** entre os probes de colisão transitáveis. O frontier escolhido vira um pequeno objetivo em coordenadas do mundo e fica **mantido** até ser alcançado, ficar ~2,5 s sem progresso ou expirar em ~7 s; ele não é recalculado a cada decisão ML. Direções `back/back_left/back_right` só são consideradas quando não existe nenhuma opção frontal/lateral transitável. Frontiers que travam entram em cooldown e também acumulam **falha persistente** no `route-graph-v1.json`; candidatos que já falharam perdem prioridade mesmo em runs futuras, e um sucesso posterior reduz essa penalidade.

Para metas rastreáveis sem evidência local útil (sem ator-alvo, baú, NPC, switch, inimigo/boss relevante no contexto), uma saída observada ou uma transição já comprovada pode ser perseguida imediatamente, sem aguardar o timer de 20 s. Um ponto onde Link **realmente mudou de room/scene** tem precedência sobre uma saída apenas escaneada. Durante recovery, cada escape é health-checked: se ~3,5 s passarem sem progresso geométrico, aquele alvo entra em cooldown por ~20 s e outra saída/porta/travessia/frontier pode ser tentada. Uma `scene_exit` não diretamente alcançável também pode usar uma ladder/escada/ledge observada como etapa intermediária. Nenhum mapa oculto é consultado.

### Memória persistente das rooms

A `room-map-v1.json` é indexada por **mundo normal/espelhado + idade + scene + room**. Em cada room visitada ela mantém, de forma incremental:

- entradas observadas e posições de spawn;
- transições que Link realmente realizou para outra scene/room, incluindo o ponto de saída real;
- `scene_exits` observados e portas `ACTORCAT_DOOR`;
- células caminhadas e células transitáveis vistas pelo NavMesh/probes;
- endpoints de probe bloqueados/sem piso;
- escadas, ladders, ledges e paredes escaláveis observadas como traversal affordances.

Isso não é um mapa pré-carregado do OoT. Na primeira visita, o agente só conhece o que o SoH já observou. Depois que Link sai de uma room, a posição da transição vira uma evidência forte de saída. Numa visita futura, se o scan atual não mostrar a saída, o controlador pode recuperar essa posição da Room Map Memory e pedir ao route graph uma rota já percorrida até ela; se ainda não houver rota completa, usa a posição lembrada como alvo collision-aware em vez de voltar a explorar a sala do zero.

### Rotas aprendidas

A route memory v1 transforma deslocamento real em um **grafo topológico dirigido**. Aproximadamente a cada célula de 80×50×80 unidades realmente atravessada, o controlador registra um nó e conecta somente o trecho que Link de fato percorreu. Não são criadas arestas reversas automaticamente, atalhos através de paredes nem conhecimento de collision que nunca foi visitado.

Quando a cognição pede um `target_position`/ator, o controlador procura no **grafo observado** uma sequência que realmente tenha sido percorrida e que aproxime Link do objetivo. Cada aresta dirigida guarda também o **ponto real de entrada** observado quando Link atravessou para a célula seguinte; replay usa esse gateway em vez do centro médio da célula, que pode cair do lado errado de um canto. Se já existir um caminho até perto do alvo, ele reutiliza os waypoints completos; se o alvo ainda for inédito, pode reutilizar apenas um trecho parcial que produza avanço geométrico relevante. Trechos percorridos mais vezes recebem maior confiança.

Uma rota parcial não pode virar outro ímã: ao chegar ao fim conhecido de um ramo que ainda não alcança o alvo, esse endpoint fica marcado como esgotado para aquele objetivo e não é reproduzido novamente até o grafo ganhar novos nós/trechos. Além disso, uma **aresta aprendida precisa continuar provando que funciona**: se por ~2,5 s ela não reduz a distância ao gateway (ou passa ~8 s sem concluir), entra em cooldown por ~20 s, acumula uma falha de confiabilidade e o planner procura outra aresta/saída/frontier. Assim, "Link já passou aqui uma vez" não significa "insista nesta direção para sempre". `local_dwell` continua sem reduzir a força de rota/frontier/scene exit; esses sinais têm seus próprios critérios de validade. O fade anti-loop fica restrito ao `target_position` direto. A memória separa rotas por scene/room, idade e mundo normal/espelhado.

A memória é incremental e persistente entre runs. Avançar por uma rota conhecida conta como progresso operacional, mas **não zera `local_dwell`** nem reabre `frontier_progress`; assim, uma volta A → B → C → A não consegue mascarar um loop.

A mesma `route-graph-v1.json` também guarda **affordances de interação aprendidas empiricamente**. Quando Link chega a uma saída/porta e existe `context_action`, o controlador não assume que "Open = A": ele testa um botão físico por vez (sem START), observa se houve transição de scene/room, diálogo ou outro efeito durável e só então grava a associação. Uma associação conhecida é reutilizada; após três falhas consecutivas ela é descartada e volta a ser explorada. Diálogo linear também tem prioridade física: enquanto `dialogue.active` estiver verdadeiro, stick/exploração ficam neutros; quando `can_advance` estiver disponível, o controlador aprende empiricamente qual botão avança o texto observando a página mudar/fechar. No fechamento, o botão é liberado imediatamente e entra um **guard de desengate** por até ~8 s (ou até Link se afastar ~120u): o stick continua livre para sair dali, mas botões ficam suprimidos para impedir `fecha → botão aleatório → reabre o mesmo diálogo`. Essas ações sobrescritas não entram no PPO. Choices continuam sob decisão da cognição. Como isso fica no route graph, o snapshot do champion congela também essas affordances.

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
  completion-0001.routes.json
  completion-0001.room-map.json
  completion-0002.pt
  completion-0002.json
  completion-0002.routes.json
  completion-0002.room-map.json
  best-completion.pt
  best-completion.json
  best-completion.routes.json
  best-completion.room-map.json
```

Cada completion guarda SHA-256 e metadados da run: tempo final, chamadas de cognição, tokens de input/output (além de cache/reasoning), custo reportado, breakdown por provider/modelo, updates, amostras, reward, modelo de cognição e estatísticas das memórias. O grafo de rotas é copiado para `completion-XXXX.routes.json` e a Room Map Memory para `completion-XXXX.room-map.json`, ambos com SHA-256 próprio. A avaliação usa essas cópias em modo read-only; conhecimento aprendido depois do champion não vaza para o benchmark. `best-completion.pt` é apenas um alias atualizável para o menor tempo de conclusão observado; os artefatos individuais nunca são sobrescritos.

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

## Native Bridge v3.2 (SoH)

Para testar apenas o motor local, sem cognição e sem atualizar pesos ou memórias:

```powershell
uv run zelda-ai qualify-motor --probe-only
uv run zelda-ai qualify-motor --save-slot 2 --seconds 120
```

`--save-slot` autoriza a seleção daquele arquivo existente por inputs físicos. O adaptador permite confirmar apenas esse arquivo e bloqueia comandos nos modos de copiar, apagar e criar nomes. O harness usa o perfil instrumentado `instrumented_local_v1`, verifica mudança real de sala seguida de estado jogável e salva observações/recibos em `.local/qualification/`. Um episódio bem-sucedido não aprova G1. O código não cria provedores de cognição nem usa o simulador.

Para executar o lote automático G1 no Windows, com os assets locais presentes junto do executável:

```powershell
uv run zelda-ai qualify-g1 C:/Projetos/Shipwright-AI/x64/Release/soh.exe --source-home C:/Projetos/Shipwright-AI/x64/Release --save-slot 2 --episodes 100 --seed 1042027 --seconds 120
```

O supervisor cria uma cópia local dos saves e da configuração, congela política e memórias e inicia seus próprios processos SoH. A cada par de episódios, carrega o save pelos controles normais, testa a saída da sala e uma nova travessia a partir do destino. As posições de preparação vêm somente das conexões dirigidas da malha de colisão observada; nenhuma posição ou câmera é escrita na memória do jogo. Os mapeamentos de teclado e gamepad ficam desativados apenas nas cópias de QA, enquanto a bridge controla o jogo. O orçamento de 120 s inclui reinício, carga, preparação, motor e verificação. Os arquivos originais são verificados por SHA-256.

Cada execução recebe um diretório novo em `.local/qualification/g1-*/`, com manifesto, relatório atualizado após cada tentativa, preparação, observações e recibos. Um lote menor, configurado por `--episodes`, serve como piloto e termina sem aprovação G1. O gate exige 100 tentativas, ao menos 99 travessias verificadas, variação efetivamente observada de posição/câmera, movimentos consumidos nas quatro direções e nenhuma atualização dos pesos. A qualificação vale para o perfil e os contextos medidos; ela não demonstra aprendizagem nova nem conclusão da campanha. O comando retorna código 2 quando o gate não é aprovado.

O lote físico de 2026-10-04 aprovou esse perfil com **99/100**, incluindo uma falha de retorno. Os resultados e limites estão em [docs/STATUS.md](docs/STATUS.md). Para auditar um lote local e exportar somente o índice público:

```powershell
uv run python scripts/export_g1.py .local/qualification/g1-3a53157b3632 --output docs/validation/g1_2026-10-04.json
```

### Currículo local de superfícies

O comando abaixo coleta demonstrações com inputs consumidos pelo SoH, treina uma rede separada por imitação e compara a mesma inicialização antes/depois em novas tentativas. O aluno controla o analógico diretamente, sem mistura com a referência. O treino não entra no PPO da campanha e o candidato fica em um checkpoint imutável, sem promoção automática.

```powershell
uv run zelda-ai train-local-surfaces C:/Projetos/Shipwright-AI/x64/Release/soh.exe --source-home C:/Projetos/Shipwright-AI/x64/Release --save-slot 2 --demonstrations 24 --evaluation 30 --retention 10 --seed 4102031
```

São três famílias medidas separadamente: subida de superfície curta observada, descida e chegada a uma célula observada após uma tentativa que falhou por ausência de movimento. O preparo relocaliza Link pela geometria atual e usa apenas controles normais nas cópias de QA. Esse perfil não cobre ladders, lofts, mira, combate nem retomada após toda espécie de colisão. O objetivo estratégico permanece estável durante as tentativas locais. Os relatórios completos ficam locais; `scripts/export_surfaces.py` audita recibos, estados, objetivos e hashes antes de exportar o índice público.

`--evaluation` deve ser divisível por três. O padrão serve para desenvolvimento; 300 tentativas antes e 300 depois fornecem 100 por família, sem agregar modos ruins com bons. Passar um piloto não aprova G2. Nenhum desses comandos instancia provedores de IA ou altera os pesos, memórias e saves originais.

O lote medido de 2026-10-04 passou de **0/30 para 21/30** após o treino. Separadamente: subida **2/10**, descida **9/10**, retomada após ausência de movimento **10/10**. A seleção/preparação de superfícies de subida ainda falhou em sete tentativas reservadas. O candidato ficou sem promoção, e a saída da casa passou nas dez regressões. O índice auditado está em [docs/validation/surface_learning_2026-10-04.json](docs/validation/surface_learning_2026-10-04.json). Isso demonstra aprendizado limitado do controle local; não qualifica navegação geral, ladders ou campanha.

No Windows com MSVC, os testes de transporte nativo também podem ser executados com `powershell -ExecutionPolicy Bypass -File scripts/verify_native.ps1`. Esse resultado complementa os testes pytest que exigem g++ ou clang++.

Revisão Shipwright fixada:

```text
HarbourMasters/Shipwright
d30fc192f2eb01ceea45bd1e12de61636cafbf86
```

A Autonomy V3 e a fundação V4 usam **Native Bridge v3.2 / Adapter v3.2** com wire protocol `3`.

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
