# Arquitetura 0.1

```text
SoH custom build -- UDP 127.0.0.1:8766 --> Bridge Python
     ^                                      |
     | input com lease                      v
     +------------------------ Runtime / skills / SQLAlchemy
                                           |
                         +-----------------+----------------+
                         |                                  |
                 Codex app-server                    OpenRouter HTTP
                 ChatGPT auth                        API key
                         |                                  |
                         +------ JSON Decision validado ----+
                                           |
                              FastAPI HTTP + WebSocket
                                           |
                                   React / TypeScript
                              vídeo local separado do prompt
```

## Decisão, estado e cadência

`models.py` define o contrato Autonomy v2 (`state-v3/skills-v2/trajectory-v2/prompt-v7`). O bridge envia snapshots limitados a 5 Hz; esses pacotes não são chamadas de IA. Cada inferência recebe um estado compacto, objetivo, último resultado, cinco eventos e até oito memórias relevantes. Um único modelo/uma única skill opera por vez. Não enviamos todo o histórico de jogo nem screenshots.

Primitives motoras continuam curtas e limitadas. Controladores compostos locais executam navegação até posição/ator, follow, conversa/interação, exploração, manipulação, mira, facing/escudo, combate genérico, equipamento, ocarina, diálogo linear e game-over sem uma chamada de modelo por frame. O planner escolhe subobjetivos e estratégias; sucesso de interação/combate exige evidência observável.

Dados expostos: posição/orientação/colisão/água, vida/magia/rupias, inventário possuído com nomes e munição aplicável, equipamento/progresso visível no pause, câmera, cena nomeada/sala/entrance/dia-noite, diálogo, ação contextual e ator associado, target, `nearby_actors` como subconjunto desenhado/próximo (24 atores/1400 unidades) e `room_actors` com até 64 atores ativos da sala atual e atores globais da sala, independentemente de câmera/renderização. Cada ator inclui categoria semântica, room, ID/params, posição, distância e flag `drawn`; `room_actor_count` e `room_actors_truncated` tornam qualquer corte explícito. Não há solução de puzzles, escrita direta em HP/inventário, navmesh global nem eventChkInf/flags ocultos de roteiro deliberadamente expostos. O estado ainda não é classificado como benchmark humano-observável estrito: contém coordenadas exatas e IDs internos.

## Controle nativo

O instalador insere `ZeldaAiBridge_OverrideInput` antes do cálculo original das bordas dos botões em `padmgr.c`, apenas no controle 0. A duração usa relógio monotônico real, não frames de renderização. Uma lease expira em até 500 ms; perda de conexão devolve o input original ao jogador.

Pacotes precisam conter token, instance_id, scene_epoch, base_seq e seq. Comandos de mundos diferentes, antigos ou repetidos são descartados. Mudança observada de scene ou room incrementa o epoch de controle, libera a lease vigente e gera evento de replanejamento. O token usa loopback; não exponha a porta como serviço remoto. A lista de eventos nativos é uma janela de 16 entradas com IDs e deduplicação, não um transporte lossless para auditoria de conclusão do jogo.

## Providers

O OpenRouter recebe `response_format=json_schema` com o mesmo contrato do Codex `turn/start.outputSchema`. O Codex usa uma thread efêmera por decisão para limitar o histórico; a memória útil é externa. Seu perfil isolado usa login oficial e não lê `auth.json` na aplicação.

Reasoning bruto não é armazenado nem mostrado. `summary` é uma explicação operacional curta solicitada no JSON e seus tokens contam na saída. `Usage` admite null; não substituímos custo/tokens desconhecidos por medição zero. Cancelamentos podem impedir contabilização final e pausam futuras chamadas.

Modelos podem conhecer Zelda do pré-treino; memória vazia não implica desconhecimento do jogo. Comparações são exploratórias até que versão de skills, saves, RNG, modalidade de estado e critérios de vitória sejam certificados.

## Persistência e segurança

SQLite local com WAL; SQLAlchemy permite uma URL alternativa com o driver instalado pelo operador. Não requer PostgreSQL ou Docker para começar. Tabelas: runs, segments, calls, events, memories, trajectories e world_edges. Reiniciar o backend marca runs anteriores como interrompidas; não retoma inferências pagas automaticamente.

HTTP aceita apenas hosts locais. Mutação exige nonce de sessão, e origens web externas são rejeitadas. WebSocket autentica pela primeira mensagem, não por segredo em URL. Chave OpenRouter digitada no painel é mantida apenas em memória. Arquivos locais de credenciais, saves, ROMs e banco ficam fora do Git.

## Fontes técnicas usadas

- [Codex App Server](https://developers.openai.com/codex/app-server/): JSON-RPC, autenticação, models e eventos de tokens.
- [Codex auth](https://developers.openai.com/codex/auth/).
- [Codex config reference](https://developers.openai.com/codex/config-reference/).
- [OpenRouter reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
- [OpenRouter usage](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
- [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).
- [Shipwright fixado](https://github.com/HarbourMasters/Shipwright/tree/d30fc192f2eb01ceea45bd1e12de61636cafbf86): GameInteractor_HookTable.h, ShipInit.hpp, padmgr.c e CMakeLists.txt.
