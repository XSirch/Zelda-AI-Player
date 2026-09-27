# Zelda AI Player

Jogador autônomo para **The Legend of Zelda: Ocarina of Time** no **Ship of Harkinian (SoH)**.

## Autonomy V3

O fluxo principal deixou de ser um LLM escolhendo `skills`. Ao clicar **INICIAR**, três loops independentes passam a trabalhar em paralelo:

- **ML actor:** PPO local em PyTorch emite diretamente o analógico N64 e os bits físicos dos botões. Ele não recebe `A = interact`, `B = attack`, `navigate_to`, `fight_enemy` ou macros equivalentes.
- **Online learner:** PPO atualiza a política com experiência recém-coletada e **RND (Random Network Distillation)** fornece curiosidade para explorar estados novos. O learner usa uma cópia separada da rede; backprop não para os inputs do jogo.
- **Cognição LLM:** Codex/ChatGPT ou OpenRouter mantém apenas o objetivo/intenção de alto nível a partir do estado estruturado. Uma inferência lenta não interrompe o controle motor.

A política recebe quatro frames estruturados consecutivos (~400 ms no ritmo atual de decisão), permitindo aprender timing de movimento e combate. O checkpoint da rede fica em `.local/ml/raw-controller-ppo-rnd-v1.pt` e é reaproveitado entre runs.

### Reward

Não existe roteiro hard-coded de Zelda no reward. Os sinais vêm de observações verificáveis:

- curiosidade RND;
- nova região visitada;
- novos atores/eventos;
- mudança de scene/room;
- mudança de diálogo/contexto;
- progresso persistente de inventário/quest/equipamento;
- progresso em direção a um alvo observado quando a cognição fornece um;
- dano causado/recebido e morte;
- pequeno custo por button-mashing.

## Painel

O painel foi reduzido ao que é útil durante uma run:

- conexão do SoH, bridge realtime e cognição;
- texto operacional do que a IA está tentando fazer;
- analógico e botões físicos pressionados em tempo real;
- tokens da run;
- tokens de cache/reasoning;
- custo API quando o provider realmente informa USD;
- cota restante do Codex/ChatGPT, lida de `account/rateLimits/read`;
- **INICIAR / PARAR**.

Para Codex autenticado por ChatGPT, tokens são contabilizados pela telemetria do app-server, mas custo em USD fica como **—** quando o provider não reporta cobrança API. A cota mostra apenas as janelas efetivamente devolvidas pelo Codex; não presume que uma janela de 5 horas exista.

## Primeira configuração

```powershell
cd C:\Projetos\Zelda-AI-Player
uv sync
uv run zelda-ai init
uv run zelda-ai auth-codex
```

Depois, inicie backend e SoH:

```powershell
uv run zelda-ai serve
uv run zelda-ai launch-soh "C:\Projetos\Shipwright-AI\build\x64\Release\soh.exe"
```

Abra **http://127.0.0.1:8787** e clique **INICIAR**.

### Configuração da cognição

Por padrão:

```text
ZELDA_AGENT_PROVIDER=codex
ZELDA_AGENT_MODEL=gpt-6-astra
ZELDA_AGENT_EFFORT=
```

Esses valores podem ser alterados no `.env`. O painel permanece deliberadamente sem seletor de skills/modelo.

## Bridge SoH

A bridge V2 continua sendo a camada de percepção e controle de baixa latência: UDP local autenticado, snapshots rápidos, journal de eventos, actor state, diálogo, inventário/progresso, probes de terreno, autosave e input scheduler nativo com lease/watchdog.

Revisão Shipwright atualmente fixada:

```text
HarbourMasters/Shipwright
d30fc192f2eb01ceea45bd1e12de61636cafbf86
```
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

- grafo global/cross-room e special links explícitos para saltos, ladders e puzzles;
- obstáculos que existem apenas como actor collider e validação adicional de plataformas móveis;
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
