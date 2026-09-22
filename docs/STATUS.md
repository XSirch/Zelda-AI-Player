# Status de implementação

Milestone 0.2 em desenvolvimento, 22/09/2026. Este arquivo distingue implementação, teste e proposta.

## Autonomy v1 — transições, diálogo e controles sistêmicos (22/09/2026)

- O contrato foi elevado para `state-v2/skills-v1/trajectory-v1/prompt-v4`. Mudanças de `scene`, `room` ou `scene_epoch` agora invalidam imediatamente a ação corrente, liberam a lease de input e forçam nova observação/planejamento.
- A bridge decodifica o `MessageContext.msgBufDecoded` usando a própria `Message_TTS_Decode` do Shipwright fixado. O estado inclui texto, speaker observado, estado da textbox, escolhas e cursor de escolha; o runtime não chama o modelo enquanto o texto ainda está sendo impresso.
- A percepção inclui `OnSetDoAction`, target atual e no máximo 24 atores realmente desenhados na sala atual, ordenados por distância e limitados a 1400 unidades. Não é uma enumeração de todos os atores da cena e não publica atores ocultos por padrão.
- Pause menu, game-over e ocarina passaram a ter estado estruturado. Há primitives locais para abrir/fechar pause, mover/confirmar/cancelar cursor, atribuir item selecionado a C, confirmar continue e tocar as 12 músicas normais a partir da tabela de notas do próprio SoH.
- Movimento/combate curto ganhou `camera_center`, `roll`, `backflip`, `sidestep` e `jump_attack`. Um score local de stuck invalida trajetórias candidatas e pede replanejamento quando ações falham repetidamente.
- O painel mostra diálogo/choices, ação contextual, atores desenhados, entrance, cutscene, pause, game-over e ocarina.
- **Validação desta mudança:** contratos e testes de regressão foram adicionados ao branch, porém a suíte Python, o build Vite e o build SoH/Windows ainda não foram executados nesta revisão porque o ambiente de autoria não conseguiu resolver `github.com` para materializar o checkout. Portanto, esta revisão ainda não é certificada como compilada/jogável.
- Aplicação local desta revisão exige atualizar o Zelda-AI-Player, executar novamente `scripts/integrate_soh.py` no checkout Shipwright fixado, reconfigurar/recompilar o SoH e reiniciar o backend. Diferentemente do hotfix abaixo, há alterações em `ZeldaAiBridge.cpp`.


## Hotfix: bridge desconectado ao carregar o save (22/09/2026)

- Causa reproduzida: o adaptador nativo envia `ItemEquips.buttonItems[8]` (B, três C-buttons e quatro slots do D-pad), mas `GameState.equipped` aceitava no máximo quatro. Os pacotes de menu eram válidos; os de gameplay eram rejeitados e o heartbeat expirava mantendo a última cena `-1`.
- Corrigido o limite para oito, preservando compatibilidade com listas antigas de quatro e mantendo a rejeição de listas maiores. Não há truncamento silencioso dos itens.
- `Bridge.status()` agora inclui `last_validation_error` com nomes conhecidos dos campos e tipos de erro. O backend registra mudanças desse diagnóstico, sem valores recebidos, tokens ou nomes arbitrários de campos. Um novo estado válido limpa o diagnóstico; pacotes rejeitados não renovam o heartbeat.
- Validação deste hotfix: a falha foi reproduzida antes da correção; **13 testes de regressão passaram** depois, incluindo transição menu/gameplay, oito slots, limites, recuperação, autenticação, replay, isolamento de instância e ida/volta de comandos por UDP em loopback. Python 3.13.5 e Pydantic 2.13.4 no ambiente de teste. A suíte completa, o frontend e o SoH/Windows não foram executados novamente nesta correção.
- Aplicação: encerrar o backend com Ctrl+C, atualizar o clone de Zelda-AI-Player e executar `uv run zelda-ai serve`, sem `--demo`. Não exige recompilar o SoH nem refazer o login. Manter a mesma pasta local de dados/token. Se for necessário reabrir o jogo, usar `launch-soh` a partir desse mesmo clone.
- Diagnóstico sem expor credenciais: `Invoke-RestMethod http://127.0.0.1:8787/api/status | ConvertTo-Json -Depth 8`. Inspecionar `bridge.connected`, `bridge.rejected_packets` e `bridge.last_validation_error`.
- O usuário informou que compilou e abriu o SoH no Windows após a correção local de `view.at` para `view.lookAt`. Este hotfix altera apenas Python/testes/documentação e não sobrescreve aquela alteração nativa local. Gameplay/autonomia ainda não estão certificados.

| Componente | Implementação | Validação realizada |
|---|---|---|
| Estado, decisões e segurança de entrada | v2 implementado: diálogo/menu/atores/transições | Testes novos escritos; execução desta revisão pendente |
| Backend, runs, memória, custos e segmentos | Implementado | SQLite e HTTP, ciclo demo |
| Codex auth/model/effort/decisões | Adaptador implementado | JSON-RPC com subprocesso falso; sem conta real |
| OpenRouter | Adaptador implementado | MockTransport, falhas/usage; sem chave real |
| Controle C++ com lease | Implementado | Compilado e executado com g++ |
| Integração SoH/padmgr | Bridge Autonomy v1 implementada | Hooks/estruturas verificados no upstream fixado; recompilação SoH/Windows pendente |
| Painel React/Vite | Telemetria Autonomy v1 adicionada | Build/typecheck desta revisão pendente |
| Memória de experiência | Notas + trajetórias persistentes por modelo/effort | Rotas são promovidas em transições autônomas e reaplicadas localmente; validação real no SoH pendente |
| Locomoção adaptativa | Giro + replay + primitives de esquiva + stuck detector | Implementado; teste real no SoH pendente |
| Diálogo, pause, game-over e ocarina | Estado estruturado + primitives locais | Implementado; teste real no SoH pendente |
| Mira, navegação espacial com colisão e combate composto | Planejados | Não implementados |
| Conclusão de dungeons/jogo certificada | Planejada | Sem métrica percentual inventada |

## Próxima validação local: gate de integração, não uma política de bloqueio

1. Instalar dependências, gerar lockfiles, buildar e revisar o painel no navegador Windows.
2. Instalar/autenticar Codex em `.local/codex`, carregar catálogo, executar uma decisão pequena sem movimentação perigosa. Conferir usage e effort nos eventos.
3. Compilar o Shipwright fixado com o bridge, abrir save de teste, conferir estado/ACK e executar uma skill curta. Testar pause, fechamento do backend e retorno do controle manual.
4. Fazer uma chamada OpenRouter com chave limitada, conferindo o custo real no provider. Não repetir automaticamente chamadas que falharam depois de faturadas.

## Backlog de gameplay

**M2 — percepção utilizável:** diálogo decodificado, ações contextuais, transições, estado de pause/game-over/ocarina e atores desenhados estão implementados no código. Restam geometria navegável/colisão observável, identificação semântica segura de entidades, mapa mundial persistente e validação real de que o filtro não vaza informação oculta.

**M3 — controle de longa duração:** pause/menu genérico, músicas e primitives de combate curto foram adicionados. Restam navegação espacial até entidades/saídas, mira calibrada de arco/Hookshot, `equip_item` por identidade semântica, combate composto parametrizado, manipulação de puzzles e critérios observáveis por encontro.

**M4 — aprendizado e benchmark certificado:** saves/checkpoints e RNG reproduzíveis, orçamento de treino, memória por modelo, seleção/promoção de skills candidatas em ambiente separado, curvas de aprendizagem, milestones de dungeons e detector de conclusão validado. Não misturar treino, dicas humanas e avaliação cega.

**M5 — produto:** filas de experimentos, gráficos tempo/tokens/custo, exportação integral paginada, vídeo gravado opcional, replay de eventos e visão sob demanda com contabilização. Nunca enviar vídeo ao modelo por padrão.

## Limites conhecidos

- Para assistir, o usuário escolhe a janela no diálogo de captura do navegador; não é uma transmissão pública automática.
- Pausar a IA não pausa o motor do jogo. Sem skill defensiva ativa, Link pode sofrer dano enquanto espera uma resposta.
- Só há uma instância de jogo e uma run ativa por backend.
- Eventos UDP não certificam conclusão de quests. O texto de diálogo agora é decodificado, mas isso não equivale a um detector certificado de progresso de quest.
- Budget local de tokens é verificado entre chamadas; o limite em USD usa reserva estimada. Cancelamentos podem deixar faturamento pendente, que deve ser auditado no provider.
- A lista `nearby_actors` contém somente atores marcados como desenhados no frame/sala, limitada por distância e quantidade; ainda precisa de validação visual no SoH para confirmar a observabilidade pretendida.
- Menu genérico não é ainda uma skill semântica de equipamento: o modelo vê IDs/cursor e opera primitives.
- Modelos OpenRouter sem JSON estruturado estão desabilitados nesta primeira versão.
- Não há CI hospedado nem gasto de GitHub Actions; os testes são locais.


## Adaptive trajectory learning (22/09/2026)

- `memory_mode=adaptive` é o padrão para novas runs no painel; `isolated` continua disponível para zero-shot/benchmark limpo.
- O runtime grava até 96 ações locais de navegação e promove a sequência quando ocorre uma mudança válida de scene/room.
- Rotas são isoladas por provider + modelo + effort + versão do contrato e persistem no SQLite entre runs e reinícios.
- Antes de gastar uma nova chamada de modelo, o runtime procura uma rota conhecida compatível com scene/room, posição e yaw de spawn e tenta reproduzi-la localmente.
- Sucessos/falhas de replay alteram a prioridade da rota. Rotas com repetidas falhas deixam de ser selecionadas.
- Ações que falham são removidas do traço candidato. Dicas humanas e `Assumir controle` invalidam o traço atual para impedir que assistência contamine aprendizado autônomo.
- O painel MEMÓRIA exibe origem, destino, número de passos, sucessos e falhas das trajetórias.
