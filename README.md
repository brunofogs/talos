# Talos

Motor de alocação que monta carteiras de investimento **por meta** para
investidores brasileiros de baixa e média renda, a partir da prateleira de
produtos de uma instituição parceira.

> **Aviso importante.** O Talos é uma **simulação**. Os dados em `exemplos/`
> são **fictícios** e as premissas são simplificadas. Nada aqui é
> recomendação de investimento, e nenhum retorno é prometido.
>
> Os limites do FGC (R$ 250 mil por conglomerado e R$ 1 milhão a cada 4 anos
> por CPF) ficam em `exemplos/premissas.json` e **precisam ser conferidos no
> site do FGC antes de qualquer uso real**.

## O que você precisa

- Python 3.10 ou mais novo ([python.org/downloads](https://www.python.org/downloads/)).
- Nenhum compilador, conta ou chave de API. Tudo roda sem internet depois
  de instalado.

## Instalação

Abra um terminal **dentro da pasta `talos`** e siga os passos do seu sistema.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[teste]"
```

Se o PowerShell bloquear o `Activate.ps1`, rode antes (vale só para aquela
janela):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

No Prompt de Comando (cmd), a ativação é `.venv\Scripts\activate.bat`.

### macOS e Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[teste]"
```

Depois de ativado, o comando `python` passa a apontar para o ambiente virtual
em qualquer sistema. Para sair do ambiente, digite `deactivate`.

## Como rodar

```bash
python -m talos --versao
python -m talos recomendar --cliente exemplos/cliente.json --prateleira exemplos/prateleira.csv
python -m talos validar --cliente exemplos/cliente.json --prateleira exemplos/prateleira.csv
```

O `recomendar` imprime, para cada meta, a situação (atingível, difícil ou sem
solução), o retorno necessário e o esperado, a chance de sucesso, a faixa de
valores no fim do prazo, a carteira e as explicações. Também grava um
registro de auditoria em JSON na pasta `saidas/`.

O `validar` compara, nos mesmos cenários, a carteira do Talos com duas
referências simples: 100% do CDI e pesos iguais entre os produtos que servem
para cada meta. Mostra retorno esperado, resultado nos 5% piores cenários e
chance de sucesso lado a lado, com uma conclusão por meta.

Opções:

| Opção | O que faz |
|---|---|
| `--premissas` e `--curva` | outros arquivos de premissas e curva (padrão: os da pasta da prateleira) |
| `--cenarios N` | quantidade de cenários (padrão: a das premissas, 5.000) |
| `--semente N` | outra semente para os cenários (padrão: a das premissas) |
| `--reamostragem` | usa a reamostragem de Michaud (bem mais lento) |
| `--saidas PASTA` | onde gravar o registro de auditoria |
| `--acoes ARQUIVO` | lista de ações da corretora (padrão: `acoes.csv` na pasta da prateleira, se existir) |
| `--estrategia` | `menor_risco` (padrão) ou `crescimento` |

Se algum arquivo faltar ou tiver dados inválidos, o programa explica o
problema em português e termina com código 2.

## Ações individuais

Além da prateleira, o Talos pode usar uma **lista de ações** (`acoes.csv`). Essa
lista precisa vir da **área de análise da corretora**: o Talos não escolhe quais
ações entram, só decide quanto colocar em cada uma, conforme o perfil e as
metas do cliente. As regras:

- ações só entram em metas de 3 anos ou mais e só para perfis que aceitam o
  risco delas (risco 4 ou 5);
- cada ação fica com no máximo 10% da meta, e a renda variável somada respeita
  o teto do perfil;
- quem trabalha numa empresa da lista não recebe a ação dela, e o setor do
  trabalho continua limitado;
- nenhum grupo (fora o Tesouro Nacional) passa de 50% da meta, e nenhum produto
  entra com menos de 2% da meta.

| Coluna do `acoes.csv` | O que é |
|---|---|
| `ticker`, `empresa`, `conglomerado`, `setor` | identificação; `conglomerado` vazio = a própria empresa |
| `risco` | de 1 a 5, na escala da prateleira |
| `beta` | quanto a ação acompanha a bolsa (1 = igual à bolsa) |
| `volatilidade_propria_aa` | oscilação própria da empresa, além da bolsa, ao ano |
| `alfa_aa` | retorno esperado além do que a bolsa explica (ex.: a partir do preço-alvo da análise) |
| `aplicacao_minima` | preço de uma ação (mercado fracionário) |

As ações de `exemplos/acoes.csv` são **fictícias** (tickers terminados em
`-F`).

> **Atenção:** indicar ações reais a investidores é atividade regulada pela CVM
> (analista de valores mobiliários e consultoria). Com a lista real da
> corretora, use o Talos internamente (linha de comando ou `servir`), sob a
> responsabilidade da área de análise, e **não** publique essa versão em
> página aberta.

**Estratégias** (`--estrategia`, `?estrategia=` na API ou a escolha na página):

- `menor_risco` (padrão): a carteira de menor perda nos piores cenários que
  atinge a meta. Ações só entram se ajudarem a reduzir o risco.
- `crescimento`: a carteira que mais cresce nos cenários comuns (a média da
  metade pior dos cenários), dentro do risco aceito pelo perfil. Ela não
  persegue as ações que só vão bem em poucos cenários de sorte.

## API para corretoras

O Talos também funciona como um serviço web (API em JSON), para o sistema de
uma corretora pedir recomendações. Em um terminal, ligue o serviço:

```bash
python -m talos servir
```

Em outro terminal, faça o papel da corretora. O comando abaixo chama a API
com o cliente de exemplo e mostra a resposta exatamente como ela chega:

```bash
python -m talos testar-api
```

| Rota | O que faz |
|---|---|
| `GET /saude` | confirma que o serviço está no ar e mostra a versão |
| `GET /prateleira` | produtos, perfis aceitos e data das premissas |
| `POST /recomendacoes` | recebe o cliente no formato do `cliente.json` e devolve a carteira de cada meta, as estatísticas e as explicações; grava a auditoria em `saidas/` |
| `POST /validacoes` | mesmo corpo; devolve a comparação com 100% do CDI e com pesos iguais |

Nas rotas `POST`, a URL aceita `?cenarios=N&semente=N`. Erros voltam em
português, no formato `{"erro": "..."}`, com o código HTTP adequado (400 para
dados inválidos, 404 para rota inexistente, 405 para método errado, 413 para
corpo acima de 1 MB). Os cálculos entram numa fila e rodam um de cada vez.

Opções do `servir`: `--porta` (padrão 8000), `--host` (padrão 127.0.0.1),
`--dados` (pasta com prateleira, premissas e curva; padrão `exemplos`) e
`--saidas`. Opções do `testar-api`: `--url`, `--cliente` e `--cenarios`.

> **Segurança:** o serviço usa só a biblioteca padrão do Python, atende
> apenas neste computador por padrão e **não tem autenticação nem HTTPS**.
> Serve para demonstração e integração local. Antes de expor à internet ou
> à rede da corretora, é preciso colocar autenticação, HTTPS e limites de
> uso na frente dele.

## Como rodar os testes

```bash
python -m pytest
```

Os mesmos testes rodam automaticamente no GitHub Actions em Windows, macOS e
Linux, com Python 3.10 e 3.12 (arquivo `.github/workflows/testes.yml`).

## Protótipo web

**Link:** https://brunofogs.github.io/talos/ (dados fictícios; na primeira abertura, o motor leva de 20 a 40 segundos para carregar).

A pasta `prototipo/` tem uma página que roda o **mesmo código Python** do
Talos dentro do navegador (com Pyodide), sem servidor: dá para editar o
perfil e as metas do cliente e ver as carteiras, a chance de sucesso e as
explicações.

Para testar no seu computador (o primeiro comando baixa uma única vez
cerca de 35 MB do Python do navegador):

```bash
python prototipo/montar.py
python -m http.server 8000 --directory prototipo/dist
```

Depois abra `http://localhost:8000` no navegador.

No GitHub, o workflow `.github/workflows/pagina.yml` roda os testes, monta a
página (com `--cdn`: o Python do navegador vem do jsDelivr, bem mais rápido) e
publica no GitHub Pages a cada push na branch `main`. Para isso,
em **Settings → Pages**, escolha **Source: GitHub Actions**.

## Organização

| Pasta | Conteúdo |
|---|---|
| `src/talos/` | código do motor e da API |
| `prototipo/` | página web de demonstração e o script que a monta |
| `tests/` | testes automáticos |
| `exemplos/` | cliente, prateleira, curva de juros e premissas (fictícios) |
| `saidas/` | recomendações geradas (não vão para o git) |

## Arquivos de entrada

Todos são **fictícios** e começam com um aviso dizendo isso (campo `_aviso`
nos JSON, linha iniciada por `#` nos CSV). Linhas com `#` nos CSV são
ignoradas. Números usam **ponto** como separador decimal, e taxas e
alíquotas são frações (0.15 = 15%).

| Arquivo | Conteúdo |
|---|---|
| `cliente.json` | perfil (`conservador`, `moderado` ou `arrojado`), renda, dívidas caras e metas (valor atual, valor-alvo, prazo em meses, se é reserva de emergência) |
| `prateleira.csv` | um produto por linha: emissor, conglomerado, indexador e taxa, vencimento e carência em meses (vazio = sem vencimento), liquidez diária (`sim`/`não`), tributação, classe, risco de 1 a 5, cobertura do FGC e aplicação mínima |
| `curva_di.csv` | vértices da curva de DI futuro: dias úteis e taxa ao ano |
| `acoes.csv` | opcional: lista de ações aprovada pela análise da corretora (veja "Ações individuais") |
| `premissas.json` | juro real, tabelas de IR, come-cotas, limites do FGC, limites de cada perfil, parâmetros do otimizador (limites por produto, por ação e por grupo, peso mínimo) e dos cenários |

Na prateleira, o significado de `taxa` depende do `indexador`: `cdi` é
fração do CDI (1.10 = 110% do CDI); `selic` e `ipca` são o spread ao ano;
`prefixado` é a taxa ao ano; `variavel` não usa a taxa (o retorno vem dos
cenários).

## Estado atual

As 8 etapas do CLAUDE.md estão concluídas, mais a regra de emprego, o protótipo
web, a API para corretoras (`servir` e `testar-api`) e as ações individuais com
a estratégia de crescimento.
