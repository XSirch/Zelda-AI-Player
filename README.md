# Zelda AI Player

Laboratório local para agentes jogarem **Ocarina of Time no Ship of Harkinian**, com estado estruturado, Codex/ChatGPT, OpenRouter e painel administrativo.

**Estado: milestone 0.3 / Autonomy v2 em desenvolvimento — harness de gameplay autônomo implementado, ainda não certificado como capaz de zerar o jogo.** A v2 adiciona percepção semântica, diálogo automático, navegação/combate/mira/equipamento compostos, grafo de mundo aprendido, recuperação de game-over e detector explícito de vitória. A bridge desta revisão precisa ser recompilada e validada no SoH/Windows antes de considerar essas capacidades operacionais. Não há ROM, assets de Zelda ou credenciais no repositório.

## O que existe nesta versão

- FastAPI/Python, persistência local SQLite/SQLAlchemy, telemetria WebSocket e histórico de runs/segmentos/chamadas/eventos.
- Painel React/TypeScript: monitor local, provider/modelo/effort, limites, iniciar/pausar/retomar/encerrar, assumir controle, instruções humanas, memória, skills e comparação de registros.
- Codex app-server por JSON-RPC/stdio, login oficial ChatGPT em perfil isolado, catálogo dinâmico e contabilização dos tokens informados pela CLI. Não usa OAuth como API key.
- OpenRouter por API, catálogo dinâmico, JSON estruturado, reasoning effort quando anunciado e custo efetivamente retornado. Sem retry pago automático.
- Bridge C++ para SoH: telemetria a 5 Hz por UDP autenticado em localhost, input analógico/botões por leases de até 500 ms, proteção contra replay/comandos stale e preempção em mudança de scene/room.
- Percepção estruturada: cena nomeada, room/entrance, dia/noite, pose/colisão/água, diálogo decodificado, ação contextual associada ao ator quando disponível, target, atores desenhados priorizados por relevância, inventário com nomes/munição aplicável, equipamento/progresso visível no pause, cutscene, game-over e ocarina. Screenshots continuam fora do prompt normal.
- Skills compostas locais: navegação até posição/ator, follow, conversa/interação com evidência, exploração, manipulação, combate genérico, mira com feedback, orientação/escudo, equipamento de C-buttons e gear, menu, game-over e as 12 músicas normais. O modelo escolhe objetivo/skill; correções de frame ficam no runtime.
- Diálogo linear é transcrito e avançado localmente sem gastar uma inferência por página. Game-over é salvo/continuado localmente. A derrota do Ganon final emite `game_completed` e encerra a run como `completed`.
- Memória persistente de notas/hipóteses, trajetórias e grafo de mundo observado. Cada transição realmente atravessada registra origem/saída, destino/spawn e entrance; não é preenchida a partir de walkthrough oculto. No modo Adaptive, sequências autônomas que conseguem mudar de sala/cena são persistidas por modelo + effort + versão do contrato e reaplicadas localmente em runs futuras antes de gastar outra inferência. **Não há treinamento de pesos nem geração automática de código de combate nesta versão.**

## Testar o painel e o ciclo sem jogo ou créditos

Requisitos: Python 3.12+, `uv`, Node 22.16+ e npm. Execute na raiz do clone:

```powershell
git clone https://github.com/XSirch/Zelda-AI-Player.git
cd Zelda-AI-Player
uv sync
cd web
npm install
npm run build
cd ..
uv run zelda-ai serve --demo
```

Abra **http://127.0.0.1:8787**. Em **AO VIVO**, selecione **Simulador determinístico** e inicie. O banner SIMULADO é permanente; não é gameplay nem um modelo de IA. O contador de tokens/custo do simulador é zero porque não chama provider algum.

Para desenvolvimento do frontend, mantenha `uv run zelda-ai serve --demo` em um terminal e execute `npm run dev` em `web/` em outro. Acesse **http://127.0.0.1:5173**. O Vite faz proxy de API e WebSocket para o backend.

> A instalação npm não foi concluída no ambiente de autoria por indisponibilidade de DNS. A sintaxe TS/TSX foi verificada, mas o typecheck completo, build Vite e inspeção visual do painel **não foram executados**. Não há lockfile inventado. Gere e versione os lockfiles após resolver dependências no ambiente local.

## Conectar o SoH real

O executável oficial sem modificações não publica este protocolo. É necessário compilar uma cópia separada do Shipwright com o adaptador incluído. Use uma extração de jogo obtida legitimamente; este projeto não fornece nem baixa ROMs.

```powershell
# Exemplo: clone separado, fora de Zelda-AI-Player
git clone --recursive https://github.com/HarbourMasters/Shipwright.git D:\Projetos\Shipwright-AI
git -C D:\Projetos\Shipwright-AI checkout d30fc192f2eb01ceea45bd1e12de61636cafbf86
git -C D:\Projetos\Shipwright-AI submodule update --init --recursive

# Na raiz de Zelda-AI-Player
uv run python scripts/integrate_soh.py D:\Projetos\Shipwright-AI
```

O instalador confere o commit e o blob de `padmgr.c`, copia apenas nossos três arquivos C++ e aplica um ponto de input antes do cálculo nativo de press/release. É idempotente; não dá `reset --hard`, não deleta assets e não reescreve o upstream arbitrariamente.

Siga as instruções de build do [Shipwright na revisão fixada](https://github.com/HarbourMasters/Shipwright/tree/d30fc192f2eb01ceea45bd1e12de61636cafbf86). Reconfigure o CMake depois de instalar o bridge, pois novos arquivos foram adicionados. **O build completo do SoH não foi executado aqui.**

> **Autonomy v2 altera novamente o código nativo da bridge.** Se você já tinha compilado uma revisão anterior, execute novamente `uv run python scripts/integrate_soh.py D:\Projetos\Shipwright-AI`, reconfigure o CMake e recompile o SoH antes de testar diálogo/transições/menu/ocarina.

Depois de compilar, abra dois terminais na raiz deste projeto:

```powershell
# Terminal 1: backend real, sem --demo
uv run zelda-ai serve

# Terminal 2: substitua pelo caminho real do executável compilado
uv run zelda-ai launch-soh "D:\Projetos\Shipwright-AI\build\CAMINHO_REAL\soh.exe"
```

`launch-soh` injeta o token local e a porta no ambiente do processo. Abra/carregue um save manualmente. O painel precisa mostrar **BRIDGE Conectado** e um estado jogável antes de autorizar uma run. Depois de iniciar a run, o objetivo padrão é completar OoT e derrotar o Ganon final sem intervenção humana. A seleção inicial de save ainda não é controlada pela IA.

## Autenticação e modelos

### Codex / ChatGPT

Instale a CLI oficial do Codex e deixe o executável `codex` no PATH. Em **CONEXÕES → Conectar ChatGPT**, conclua o login na página oficial e clique em **Verificar conexões**. O código de dispositivo é uma alternativa quando suportado pela CLI.

A autenticação fica em `.local/codex`, sob gerenciamento da CLI. Por isolamento, o projeto **não copia** as credenciais do seu perfil global; será necessário autenticar esse perfil uma vez. Alternativa no PowerShell:

```powershell
$env:CODEX_HOME = Join-Path (Get-Location) ".local\codex"
codex login
```

O subprocesso trabalha num diretório vazio, sem carregar os projetos/MCPs pessoais, com ferramentas de shell e pesquisa desabilitadas e sandbox de leitura. O modelo de gameplay deve somente devolver o JSON da decisão. Não há bloqueios de worktree impostos ao agente que desenvolve o projeto.

Os modelos e efforts vêm de `model/list`, não de nomes inventados. O consumo usa os limites/créditos da sua conta Codex, e **não é representado como custo zero ou como fatura estimada da API OpenAI**. A disponibilidade e a quota dependem da sua conta. O limite de saída configurado no painel aplica-se ao OpenRouter; o adaptador Codex não oferece um teto de saída por chamada.

### OpenRouter

Informe a chave mascarada em **CONEXÕES** (memória do processo) ou copie `.env.example` para `.env` e preencha `OPENROUTER_API_KEY` no seu computador. Nesta versão só são selecionáveis modelos que anunciam `structured_outputs`.

Não presumimos que todos aceitam os mesmos efforts. O seletor utiliza o metadado `reasoning.supported_efforts`. Custo desconhecido ou tokens ausentes ficam explicitamente pendentes e impedem novas inferências na run. Falhas, cancelamentos ou respostas truncadas podem ter sido cobrados; não repetimos automaticamente a chamada.

## Vídeo não consome tokens

Em **Selecionar janela**, escolha a janela do SoH no diálogo do navegador. O vídeo é apenas um `MediaStream` local, exibido em `<video>`, sem upload ao backend ou ao modelo. O modelo recebe somente texto/estado. Fechar a captura não encerra a run; pausar a IA não pausa o jogo.

## Custos e comparação

- Totais = tokens de entrada + saída. Cache e reasoning são subconjuntos, nunca somados novamente.
- Há limites de chamadas, tokens, duração e uma reserva conservadora antes de cada chamada OpenRouter. Uma chamada em andamento pode ultrapassar um orçamento de tokens; a reserva de USD não é garantia contratual do provider. Para teto de cobrança, configure também limite na chave do provider.
- Trocas de modelo/effort são aplicadas **entre decisões** e criam segmentos. A run passa a ser mista.
- Dicas e controle humano marcam a run como assistida. Partidas simuladas nunca são rotuladas como SoH.
- O fingerprint atual identifica revisão e estado inicial observado, **não** um save state/RNG certificado. Não produzimos um percentual de conclusão ou ranking de quem zerou.
- O núcleo, contrato de decisão e skills são compartilhados. Codex e OpenRouter usam transportes e envelopes diferentes; não alegamos equivalência perfeita dos harnesses internos.
- A tela de inspeção/exportação contém as últimas 200 entradas por tipo. O banco local guarda o histórico integral.

## Testes

```powershell
uv run pytest -q
cd web
npm run build
```

A suíte histórica havia passado antes da Autonomy v2. Para esta revisão, novos testes de contratos/controladores/grafo/diálogo/game-over/conclusão foram escritos, mas **não puderam ser executados no ambiente de autoria porque o checkout via GitHub continua bloqueado por DNS**. O build Vite e a recompilação completa do SoH/Windows também permanecem pendentes. O teste isolado de `InputLease.hpp` não substitui compilar o adaptador dentro do SoH.

Veja [docs/STATUS.md](docs/STATUS.md) para fronteiras do milestone e próximo trabalho, [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para os contratos e [AGENTS.md](AGENTS.md) para desenvolvimento.
