# Status de implementação

Milestone 0.1, 22/09/2026. Este arquivo distingue implementação, teste e proposta.

| Componente | Implementação | Validação realizada |
|---|---|---|
| Estado, decisões e segurança de entrada | Implementado | Testes Python |
| Backend, runs, memória, custos e segmentos | Implementado | SQLite e HTTP, ciclo demo |
| Codex auth/model/effort/decisões | Adaptador implementado | JSON-RPC com subprocesso falso; sem conta real |
| OpenRouter | Adaptador implementado | MockTransport, falhas/usage; sem chave real |
| Controle C++ com lease | Implementado | Compilado e executado com g++ |
| Integração SoH/padmgr | Código e instalador implementados | Assinaturas verificadas no upstream; não compilado dentro do SoH |
| Painel React/Vite | Código implementado | Sintaxe TS/TSX; npm bloqueado por DNS, sem build/typecheck completo/revisão visual |
| Memória de experiência | Notas persistentes implementadas | Isolamento/deduplicação testados |
| Mira, navegação e combate adaptativo | Planejados | Não implementados |
| Conclusão de dungeons/jogo certificada | Planejada | Sem métrica percentual inventada |

## Próxima validação local: gate de integração, não uma política de bloqueio

1. Instalar dependências, gerar lockfiles, buildar e revisar o painel no navegador Windows.
2. Instalar/autenticar Codex em `.local/codex`, carregar catálogo, executar uma decisão pequena sem movimentação perigosa. Conferir usage e effort nos eventos.
3. Compilar o Shipwright fixado com o bridge, abrir save de teste, conferir estado/ACK e executar uma skill curta. Testar pause, fechamento do backend e retorno do controle manual.
4. Fazer uma chamada OpenRouter com chave limitada, conferindo o custo real no provider. Não repetir automaticamente chamadas que falharam depois de faturadas.

## Backlog de gameplay

**M2 — percepção utilizável:** texto de diálogo decodificado, ações contextuais, identificação de atores visíveis com filtro de sala/visibilidade/oclusão, geometria navegável limitada, estado de menus e mapa aprendido. Testes de observabilidade para não vazar puzzles/baús/flags ocultos.

**M3 — controle de longa duração:** feedback de navegação, câmera, mira calibrada de arco/Hookshot, equipar itens via menus, estratégias de combate parametrizadas, logs de acerto/falha e calibração. Skills devem possuir critérios observáveis de sucesso e testes por encontro.

**M4 — aprendizado e benchmark certificado:** saves/checkpoints e RNG reproduzíveis, orçamento de treino, memória por modelo, seleção/promoção de skills candidatas em ambiente separado, curvas de aprendizagem, milestones de dungeons e detector de conclusão validado. Não misturar treino, dicas humanas e avaliação cega.

**M5 — produto:** filas de experimentos, gráficos tempo/tokens/custo, exportação integral paginada, vídeo gravado opcional, replay de eventos e visão sob demanda com contabilização. Nunca enviar vídeo ao modelo por padrão.

## Limites conhecidos

- Para assistir, o usuário escolhe a janela no diálogo de captura do navegador; não é uma transmissão pública automática.
- Pausar a IA não pausa o motor do jogo. Sem skill defensiva ativa, Link pode sofrer dano enquanto espera uma resposta.
- Só há uma instância de jogo e uma run ativa por backend.
- Eventos UDP não certificam conclusão de quests. IDs de mensagens não substituem diálogos para raciocínio de puzzles.
- Budget local de tokens é verificado entre chamadas; o limite em USD usa reserva estimada. Cancelamentos podem deixar faturamento pendente, que deve ser auditado no provider.
- Modelos OpenRouter sem JSON estruturado estão desabilitados nesta primeira versão.
- Não há CI hospedado nem gasto de GitHub Actions; os testes são locais.
