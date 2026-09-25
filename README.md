# Zelda AI Player

Harness local para agentes jogarem **The Legend of Zelda: Ocarina of Time** no **Ship of Harkinian (SoH)** usando estado estruturado do jogo, controle nativo, Codex/ChatGPT ou OpenRouter e painel administrativo web.

## Status atual

**Realtime Input Foundation v2.5 está na `main`.** O projeto já separa planejamento por modelo do controle motor local, possui observações rápidas, confirmação de consumo de input no engine, identidade por instância de ator e budgets independentes. Ainda **não está certificado como capaz de zerar OoT autonomamente**.

O próximo trabalho principal continua sendo Navigation V2 (geometria/navmesh/A*). O combate agora aprende uma política separada para cada tipo de inimigo observado; adapters específicos de animação/boss continuam sendo refinamentos futuros.

Revisão SoH fixada:

```text
HarbourMasters/Shipwright
d30fc192f2eb01ceea45bd1e12de61636cafbf86
```

Diretórios Windows atuais:

```text
C:\Projetos\Zelda-AI-Player
C:\Projetos\Shipwright-AI
```

## Implementado

- **Backend:** Python/FastAPI, SQLite/SQLAlchemy, WebSocket e histórico de runs, segmentos, chamadas, eventos, memória, trajetórias e grafo de mundo.
- **Painel React/TypeScript:** provider/modelo/effort, budgets, iniciar/pausar/retomar/encerrar, assumir controle, skills, memória e benchmarks. O AO VIVO usa abas logo abaixo do vídeo para Controle, Combate, Terreno, Atores, Progresso, Estado e Decisão, evitando uma página vertical gigante.
- **Providers:** Codex app-server com login ChatGPT isolado e OpenRouter por API. Não existe retry pago automático.
- **Bridge V2:** UDP autenticado em localhost com snapshots rápidos para controle e snapshots completos aproximadamente a cada 200 ms para dados mais pesados.
- **Input scheduler nativo:** setpoints contínuos separados de sequências discretas, deduplicação, owner epochs, watchdog monotônico e receipts de `accepted`, `consumed` e `completed`.
- **Hook no consumo do controle:** sequências avançam nas leituras que efetivamente consomem `Input`, não em timers do Python nem em frames de renderização.
- **Percepção estruturada:** pose, yaw, câmera, scene/room, colisão, terreno, diálogo, inventário, equipamento, targeting, atores da sala inclusive off-camera, game-over, cutscene e ocarina.
- **Navigation V2:** o bridge gera a cada snapshot completo uma grade walkable 9×9 centrada em Link a partir de `BgCheck`; cada aresta exige piso contínuo, desnível caminhável, corredor sem parede e margem lateral para o corpo. O Python usa A* em links recíprocos, não corta quinas, recentra a malha conforme Link avança e valida o próximo heading novamente nas 16 sondas rápidas antes de enviar analógico. `navigate_to`, aproximação, follow e exploração usam a malha; `move` primitivo é recusado quando a sonda indica parede/vazio/desnível inseguro.
- **Actor UID:** inimigos iguais deixam de ser identificados apenas por `actor_id`; cada vida/spawn observado recebe identidade própria.
- **Journal de eventos:** eventos não confirmados podem ser reenviados e gaps são explicitamente detectados.
- **Skills locais:** navegação curta, porta, traverse, follow, interação, exploração, manipulação, mira, equipamento/menu e músicas.
- **Aprendizado de combate por inimigo:** no modo Adaptive, cada modelo+effort mantém perfis isolados por classe de inimigo. O controlador aprende quais ações funcionam em estados como distância, aproximação/recuo do inimigo, orientação, dano recente, ameaça e lock, atualiza a política por tentativa e erro durante a luta e persiste encontros, vitórias, derrotas, dano recebido e melhores ações. O LLM recebe esse resumo nas decisões seguintes; o loop motor continua local para não introduzir latência de inferência em cada golpe. Pausa/intervenção humana/dica tainta a run para combate: a IA pode continuar usando perfis anteriores, mas novos episódios daquela run não são promovidos como aprendizado autônomo.
- **Parada independente do provider:** stop/take-control revoga o input antes de aguardar cleanup de inferência ou validação de modelo.
- **Diagnóstico local de input:** A/B, Z-target, frente, ré, backflip e stress A/B ×20 rodam sem provider, sem benchmark e sem memória; exibem latência Python→consumo P50/P95/P99, fila nativa e edges observados. Backflip/sidestep usam o camera input yaw nativo do OoT para converter direção relativa ao Link em analógico relativo à câmera, seguram Z + direção até o próprio Player_ProcessControlStick reportar a direção esperada, então geram o edge de A; verificam piso na direção Link-relative e só confirmam sucesso quando o engine reporta HOPPING com a direção esperada (backflip = 2, left = 1, right = 3). Movimento recusa quando o probe não comprova piso seguro.
- **Aprendizado versionado:** dados anteriores são preservados; traces falhos/intervenções não são promovidos como experiência autônoma.

> `consumed` significa que o input chegou ao consumidor do jogo. Não significa automaticamente que um golpe acertou, uma esquiva teve efeito ou uma animação terminou.

## Budgets: `0 = sem limite`

| Campo | Valor 0 |
| --- | --- |
| `max_calls` | sem teto de chamadas no harness |
| `max_tokens` | sem teto acumulado de tokens no harness |
| `max_cost_usd` | sem teto de USD do harness |
| `max_runtime_s` | sem limite de duração da run |

Exemplo sem tetos impostos pelo harness:

```json
{
  "max_calls": 0,
  "max_tokens": 0,
  "max_cost_usd": 0,
  "max_runtime_s": 0,
  "max_output_tokens": 2048
}
```

`max_output_tokens` é um limite **por resposta OpenRouter** e continua positivo. `max_cost_usd = 0` não torna o OpenRouter gratuito; apenas desativa o teto adicional do Zelda AI Player.

## Atualizar a main

```powershell
cd C:\Projetos\Zelda-AI-Player

git switch main
git fetch origin
git pull --ff-only origin main

uv sync
uv run pytest -q
```

Se houver alterações locais importantes, preserve-as antes do pull com commit próprio ou `git stash push -u`. Não use `reset --hard` apenas para atualizar.

## Build do painel

```powershell
cd C:\Projetos\Zelda-AI-Player\web
npm install
npm run build
cd ..
```

Painel de produção: **http://127.0.0.1:8787**

## Integrar e recompilar o SoH

O `soh.exe` oficial não contém esta bridge. É necessário recompilar o checkout fixado.

### 1. Confirmar o upstream

```powershell
git -C C:\Projetos\Shipwright-AI rev-parse HEAD
```

Esperado:

```text
d30fc192f2eb01ceea45bd1e12de61636cafbf86
```

Checkout do zero:

```powershell
git clone --recursive https://github.com/HarbourMasters/Shipwright.git C:\Projetos\Shipwright-AI
git -C C:\Projetos\Shipwright-AI checkout d30fc192f2eb01ceea45bd1e12de61636cafbf86
git -C C:\Projetos\Shipwright-AI submodule update --init --recursive
```

### 2. Instalar a bridge V2

```powershell
cd C:\Projetos\Zelda-AI-Player
uv run python scripts/integrate_soh.py C:\Projetos\Shipwright-AI
```

O integrador instala `ZeldaAiBridge.cpp`, `ZeldaAiBridge.h`, `InputScheduler.hpp` e `ActorRegistry.hpp`. Ele valida o `padmgr.c`, reconhece a integração anterior, guarda backup fora do glob do CMake e aborta em alterações desconhecidas.

### 3. Reconfigurar CMake

No computador atual, o projeto foi configurado com **Visual Studio 2026 (generator VS 18)** mantendo o toolset **v143** para a revisão fixada do Shipwright. O componente MSVC v143 precisa estar instalado no Visual Studio Installer.

```powershell
cd C:\Projetos\Shipwright-AI
git submodule update --init --recursive

& 'C:\Program Files\CMake\bin\cmake.exe' `
  -S . `
  -B "build/x64" `
  -G "Visual Studio 18 2026" `
  -T v143 `
  -A x64
```

### 4. Gerar assets e compilar Release

```powershell
& 'C:\Program Files\CMake\bin\cmake.exe' `
  --build .\build\x64 `
  --config Release `
  --target GenerateSohOtr

& 'C:\Program Files\CMake\bin\cmake.exe' `
  --build .\build\x64 `
  --config Release
```

Localize o executável:

```powershell
Get-ChildItem C:\Projetos\Shipwright-AI\build\x64 -Recurse -Filter soh.exe |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName, LastWriteTime
```

## Executar com a bridge autenticada

Terminal 1:

```powershell
cd C:\Projetos\Zelda-AI-Player
uv run zelda-ai serve
```

Terminal 2:

```powershell
cd C:\Projetos\Zelda-AI-Player
uv run zelda-ai launch-soh "C:\Projetos\Shipwright-AI\build\x64\Release\soh.exe"
```

Use o caminho real do seu build. O `launch-soh` injeta token e porta no ambiente do processo.

No painel, confirme **BRIDGE Conectado**, **BRIDGE V2**, event gaps = 0 e um save jogável carregado. Um executável antigo aparecerá como bridge legado/sem RT.

## Codex / ChatGPT

```powershell
where.exe codex
codex --version
```

Se precisar apontar explicitamente o executável:

```powershell
$codex = (Get-Command codex.exe -CommandType Application).Source
$env:ZELDA_CODEX_COMMAND = $codex
[Environment]::SetEnvironmentVariable("ZELDA_CODEX_COMMAND", $codex, "User")
```

Depois reinicie o backend.

## OpenRouter

A chave pode ser informada no painel ou por `OPENROUTER_API_KEY` em um `.env` local. Uso/custo ausente permanece desconhecido. Não há retry automático de chamada paga.

## Testes e validação atual

```powershell
cd C:\Projetos\Zelda-AI-Player
uv run pytest -q
```

Na primeira execução Windows da V2 após o merge, foram reportados:

```text
156 passed
17 skipped
1 failed
```

A única falha era de **encoding do próprio teste**: `Path.read_text()` usou `cp1252` no Windows/Python 3.14 ao ler TSX em UTF-8. A revisão seguinte passa `encoding="utf-8"` explicitamente nas leituras/escritas de fixtures textuais relevantes. Rode a suíte novamente para registrar o resultado final.

Os dois warnings observados nessa execução (Starlette/httpx e serialização Pydantic em um fixture) não causaram essa falha, mas permanecem candidatos a limpeza.

Também valide:

```powershell
cd C:\Projetos\Zelda-AI-Player\web
npm run build
```

## Ainda pendente

- navmesh/A* global derivado da geometria do jogo;
- plataformas/obstáculos dinâmicos;
- adapters específicos de animação/abertura para inimigos e bosses;
- validação de convergência dos perfis de combate ao longo de múltiplos encontros;
- replay de trajetória certificado e vinculado ao objetivo atual;
- isolamento do motor realtime em processo separado de SQLite/UI;
- validação repetida da medição Python→consumo p50/p95/p99 no SoH/Windows;
- certificação de uma run completa até o Ganon.

Documentação:

- [Realtime Input Foundation](docs/REALTIME_V2.md)
- [Status](docs/STATUS.md)
- [Arquitetura](docs/ARCHITECTURE.md)
- [Instruções para agentes](AGENTS.md)
